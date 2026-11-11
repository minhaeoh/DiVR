"""GSM8K correctness reward for d1's diffu-GRPO trainer that uses this paper's numeric grader.

d1's `correctness_reward_func` pays 2.0 only when the text inside <answer>...</answer> equals the gold answer
string exactly. The untuned LLaDA-8B-Base writes sentences inside the tags, so that reward is almost never paid
and the policy does not move. For the Base run of the paper, the reward below replaces it: the first <answer>
block (last \\boxed inside it), else a "#### N" line, else the last \\boxed, else the last number, compared
numerically. d1's four format rewards are kept unchanged. The prompt of that run also ends with the assistant
prefix "<reasoning>", passed to the trainer as an opening assistant turn so that the rendered prompt equals the
one used everywhere else in the paper.

Usage with the d1 repository (https://github.com/dllm-reasoning/d1): copy this file next to
diffu-grpo/reward_func.py and, in diffu_grpo_train.py, replace `correctness_reward_func` in the GSM8K reward list
with `d1_lenient_reward.correctness_reward_func_lenient`. This file is self-contained.
"""
from __future__ import annotations

import re

_TERM = re.compile(r"<\|endoftext\|>|<\|eot_id\|>")
_BOX = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*)\}")
_ANS = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _num(s):
    m = _NUM.search(str(s).replace(",", ""))
    try:
        return float(m.group()) if m else None
    except ValueError:
        return None


def gsm8k_lenient_extract(text: str):
    m = _TERM.search(text)
    seg = text[: m.start()] if m else text
    cand = None
    a = _ANS.search(seg)
    if a:
        bx = _BOX.findall(a.group(1))
        cand = bx[-1] if bx else a.group(1)
    if cand is None:
        m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", seg)
        cand = m.group(1) if m else None
    if cand is None:
        bx = _BOX.findall(seg)
        cand = bx[-1] if bx else None
    if cand is None:
        nums = _NUM.findall(seg.replace(",", ""))
        cand = nums[-1] if nums else None
    return cand


def gsm8k_lenient_correct(text: str, gt: str) -> bool:
    cand = gsm8k_lenient_extract(text)
    p, t = (_num(cand) if cand is not None else None), _num(gt)
    return p is not None and t is not None and abs(p - t) < 1e-4


def correctness_reward_func_lenient(prompts, completions, answer, step=None, run_name=None, **kwargs) -> list[float]:
    """Drop-in replacement for d1's correctness_reward_func (2.0 for a correct final answer, else 0.0)."""
    responses = [completion[0]["content"] for completion in completions]
    return [2.0 if gsm8k_lenient_correct(r, a) else 0.0 for r, a in zip(responses, answer)]
