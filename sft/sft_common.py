"""Masked-diffusion SFT on GSM8K: data formatting, the noising collator and the loss (adapted from d1's SFT trainer).

Target format (the generator continues from the prompt's trailing "<reasoning>"):

    prompt   : MATH_SYSTEM_PROMPT + question -> chat template -> "<reasoning>"
    response : \\n{rationale}\\n</reasoning>\\n<answer>\\n\\boxed{answer}\\n</answer> + EOS, EOS-filled to a multiple
               of the decoding block length (32) so that the model learns to stop

GSM8K's calculator annotations (<<48/2=24>>) are stripped from the rationale. Nothing is truncated: a row longer
than max_length is dropped and counted.

Loss: absorbing-state masked diffusion on the response only (the prompt is never noised), weighted by 1/t and
normalised by the number of real (non-prompt, non-padding) response tokens.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from transformers import DefaultDataCollator, Trainer

from divr.prompts import MATH_SYSTEM_PROMPT, REASONING_PREFIX

CALC_ANNOTATION = re.compile(r"<<[^>]*>>")


def split_answer_field(answer_field: str):
    """GSM8K's `answer` is `{rationale}\\n#### {number}`: (rationale without calculator annotations, number, n stripped)."""
    rationale = answer_field.split("####")[0]
    n_calc = len(CALC_ANNOTATION.findall(rationale))
    m = re.search(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)", answer_field)
    gold = m.group(1).replace(",", "").strip() if m else ""
    return CALC_ANNOTATION.sub("", rationale).strip(), gold, n_calc


def preprocess_gsm8k(rows, tokenizer, max_length: int = 640, eos_fill_block: int = 32):
    """Tokenise rows to (input_ids, prompt_lengths, real_lengths); no padding, no truncation."""
    out, kept, n_dropped, n_calc_total = [], [], 0, 0
    for ex in rows:
        rationale, gold, n_calc = split_answer_field(ex["answer"])
        n_calc_total += n_calc
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": MATH_SYSTEM_PROMPT + "\n\n" + str(ex["question"]).strip()}],
                                               add_generation_prompt=True, tokenize=False) + REASONING_PREFIX
        response = f"\n{rationale}\n</reasoning>\n<answer>\n\\boxed{{{gold}}}\n</answer>"
        enc = tokenizer(prompt + response, add_special_tokens=False)["input_ids"]
        if len(enc) > max_length:
            n_dropped += 1
            continue
        p_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if eos_fill_block:
            enc = enc + [tokenizer.eos_token_id]
            enc += [tokenizer.eos_token_id] * (-(len(enc) - p_len) % eos_fill_block)
        kept.append(len(enc))
        out.append({"input_ids": torch.tensor(enc, dtype=torch.long), "prompt_lengths": torch.tensor(p_len),
                    "real_lengths": torch.tensor(len(enc))})
    stats = {"n_kept": len(out), "n_dropped_too_long": n_dropped, "max_length": max_length,
             "n_calc_annotations_stripped": n_calc_total,
             "kept_len_mean": float(sum(kept) / max(len(kept), 1)), "kept_len_max": max(kept) if kept else 0}
    return out, stats


def freeze_eval_noise(examples, eps: float = 1e-3):
    """A fixed noise level per held-out example, stratified over the schedule, so eval_loss is comparable across checkpoints."""
    n = len(examples)
    for i, e in enumerate(examples):
        e["t"] = torch.tensor((i + 0.5) / n * (1 - eps), dtype=torch.float32)
    return examples


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DiffusionCollator(DefaultDataCollator):
    """Applies the forward (masking) process to the response and marks the supervised positions."""

    def __init__(self, tokenizer, mask_token_id: int, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id

    def forward_process(self, input_ids, t_in=None, eps: float = 1e-3):
        B, N = input_ids.shape
        t = torch.rand((B,), device=input_ids.device) if t_in is None else t_in
        t = ((1 - eps) * t + eps)[:, None].repeat(1, N)
        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        return torch.where(mask_indices, self.mask_token_id, input_ids), t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        noisy, t, mask_indices = self.forward_process(input_ids, batch.get("t"))
        idx = torch.arange(input_ids.shape[1]).unsqueeze(0)
        prompt_mask = idx < batch.pop("prompt_lengths").view(-1, 1)
        real_mask = idx < batch.pop("real_lengths").view(-1, 1)       # pad == eos for LLaDA, so padding is tracked by length
        response_mask = real_mask & ~prompt_mask
        noisy[prompt_mask] = input_ids[prompt_mask]
        labels[~(mask_indices & response_mask)] = -100
        batch["input_ids"] = noisy.long()
        batch["labels"] = labels
        batch["t"] = t
        batch["num_real_tokens"] = response_mask.sum()
        return batch


class DiffusionTrainer(Trainer):
    """Absorbing-state diffusion loss, normalised over real response tokens. `shift` applies Dream's logit shift."""

    shift = False

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        labels = inputs.pop("labels")
        t = inputs.pop("t")
        n_real = inputs.pop("num_real_tokens")
        outputs = model(input_ids=inputs["input_ids"])
        logits = outputs.logits
        if self.shift:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1), reduction="none").view(labels.shape)
        loss = (ce / t).sum() / n_real.clamp_min(1)
        if model.training and self.state.global_step % max(self.args.logging_steps, 1) == 0:
            with torch.no_grad():
                n_sup = (labels != -100).sum().clamp_min(1)
                self.log({"unscaled_loss": (ce.sum() / n_sup).item(), "supervised_tokens": int(n_sup.item()),
                          "real_tokens": int(n_real.item())})
        return (loss, outputs) if return_outputs else loss


def enable_llada_activation_checkpointing(model) -> None:
    """LLaDA inherits OLMo's per-layer checkpointing, which the HF gradient_checkpointing flag does not reach."""
    target = getattr(model, "model", model)
    fn = getattr(target, "set_activation_checkpointing", None)
    if fn is None:
        raise SystemExit("set_activation_checkpointing is missing on this model")
    fn("whole_layer")
