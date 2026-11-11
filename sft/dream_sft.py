#!/usr/bin/env python3
"""GSM8K SFT of Dream-v0-Instruct-7B with the LLaDA recipe (appendix table of the paper).

Same data, format, loss and hyper-parameters as sft.llada_sft; Dream-specific: mask id 151666, Qwen module names,
and the one-position logit shift (output i predicts token i + 1), so the loss reads the prediction for position i
from output i - 1, the same convention as divr.family.logits_of at decoding time.

    python -m sft.dream_sft --output_dir checkpoints/dream-gsm8k-sft
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer, TrainingArguments

from divr.data import load_gsm8k_split
from divr.family import DREAM
from sft.sft_common import DiffusionCollator, DiffusionTrainer, SFTDataset, freeze_eval_noise, preprocess_gsm8k


class DreamTrainer(DiffusionTrainer):
    shift = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=128); ap.add_argument("--lora_alpha", type=int, default=256)
    ap.add_argument("--max_length", type=int, default=1024, help="Qwen's tokenizer splits digits, so above LLaDA's 640")
    ap.add_argument("--eos_fill_block", type=int, default=32)
    ap.add_argument("--split_seed", type=int, default=42); ap.add_argument("--n_holdout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--n_eval", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    train_rows, hold_rows, split_meta = load_gsm8k_split(args.split_seed, args.n_holdout)
    processed, stats = preprocess_gsm8k(train_rows, tok, args.max_length, args.eos_fill_block)
    eval_processed = freeze_eval_noise(preprocess_gsm8k(hold_rows, tok, args.max_length, args.eos_fill_block)[0])
    if args.limit:
        processed = processed[:args.limit]
    if args.n_eval:
        eval_processed = eval_processed[:args.n_eval]
    assert tok.eos_token_id in DREAM.eos_ids, (tok.eos_token_id, DREAM.eos_ids)
    print(f"[dream_sft] {len(processed)} train / {len(eval_processed)} held out; len mean {stats['kept_len_mean']:.0f} max {stats['kept_len_max']}", flush=True)

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias="none",
                                             target_modules=list(DREAM.lora_modules), task_type=TaskType.CAUSAL_LM))
    model.print_trainable_parameters()
    steps_per_epoch = max(1, len(processed) // (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, max_grad_norm=1.0, lr_scheduler_type="linear", warmup_steps=50,
        bf16=True, seed=args.seed, logging_steps=20,
        save_strategy="steps", save_steps=steps_per_epoch, eval_strategy="steps", eval_steps=steps_per_epoch,
        per_device_eval_batch_size=args.per_device_train_batch_size, save_total_limit=None,
        report_to=[], remove_unused_columns=False, prediction_loss_only=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, dataloader_num_workers=2)
    trainer = DreamTrainer(model=model, args=targs, train_dataset=SFTDataset(processed), eval_dataset=SFTDataset(eval_processed),
                           data_collator=DiffusionCollator(tokenizer=tok, mask_token_id=DREAM.mask_id))
    t0 = time.time()
    trainer.train()
    ev = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    meta = {"model": args.model, "split": split_meta, "data_stats": stats, "n_train": len(processed), "n_eval": len(eval_processed),
            "final_eval": ev, "hours": round((time.time() - t0) / 3600, 2), **vars(args), "lora_target_modules": list(DREAM.lora_modules)}
    json.dump(meta, open(pathlib.Path(args.output_dir) / "train_meta.json", "w"), indent=1, default=str)
    print(f"[dream_sft] done in {meta['hours']} h; held-out eval {ev}", flush=True)


if __name__ == "__main__":
    main()
