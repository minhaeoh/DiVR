"""Model-family constants: LLaDA (Base / Instruct) and Dream.

Dream (Qwen2.5-derived, Dream-org/Dream-v0-Instruct-7B) differs from LLaDA in four ways that the
decoders and the state builder need: its mask / end-of-turn / pad token ids, the ChatML template
(handled by the tokenizer), LoRA module names (Qwen), and a one-position shift of the logits (the
output at position i predicts the token at i + 1). LLaDA decoding runs under fp16 autocast to match
d1's evaluation numerics; Dream runs under bf16.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Family:
    name: str
    mask_id: int
    eos_ids: tuple           # any of these ends the response (early stop)
    fill_id: int             # what the canvas is filled with after the stop
    stop_strings: tuple      # cut the decoded text at the first of these before grading
    lora_modules: tuple      # verifier LoRA targets
    shift: bool              # logits[:, i] predicts position i + 1
    autocast_dtype: torch.dtype


LLADA = Family("llada", 126336, (126081,), 126081, ("<|endoftext|>",),
               ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj"), False, torch.float16)
DREAM = Family("dream", 151666, (151645, 151643), 151643, ("<|im_end|>", "<|endoftext|>"),
               ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), True, torch.bfloat16)


def get_family(model_name: str) -> Family:
    return DREAM if "dream" in model_name.lower() else LLADA


def logits_of(model, x: torch.Tensor, fam: Family, sl: tuple[int, int] | None = None) -> torch.Tensor:
    """Per-position next-token logits of canvas x under the family's numerics, shift applied.

    With sl = (s0, e0) only positions s0:e0 are returned, read from the unshifted logits instead of a full
    shifted copy (a 16-row batch on a long prompt would otherwise exceed GPU memory). The values are identical
    to logits_of(model, x, fam)[:, s0:e0]."""
    with torch.autocast(device_type="cuda", dtype=fam.autocast_dtype):
        logits = model(x).logits
    if sl is not None:
        s0, e0 = sl
        if not fam.shift:
            return logits[:, s0:e0]
        return logits[:, s0 - 1:e0 - 1] if s0 > 0 else torch.cat([logits[:, :1], logits[:, :max(e0 - 1, 0)]], dim=1)
    if fam.shift:
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


def argmax_of(model, x: torch.Tensor, fam: Family) -> torch.Tensor:
    """logits_of(model, x, fam).argmax(-1), shifting the argmax indices rather than the logits."""
    with torch.autocast(device_type="cuda", dtype=fam.autocast_dtype):
        am = model(x).logits.argmax(-1)
    if fam.shift:
        am = torch.cat([am[:, :1], am[:, :-1]], dim=1)
    return am


def cut(text: str, fam: Family) -> str:
    """The decoded text up to the first end token."""
    for s in fam.stop_strings:
        text = text.split(s)[0]
    return text
