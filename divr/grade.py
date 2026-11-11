"""Graders. A response is cut at its first end token, then graded by the task's rule:

  GSM8K      the first <answer> block (last \\boxed inside it), else a "#### N" line, else the last \\boxed,
             else the last number; compared numerically.
  MATH-500   the first <answer> block (last \\boxed inside it), else the last \\boxed; MATH equivalence (d1).
  Countdown  the boxed expression must use each given number exactly once and evaluate to the target.
  code       the program passes its tests (code_eval).

`vote_key` is the per-task answer key used by majority / weighted voting.
"""
from __future__ import annotations

import re
from math import comb

from divr import code_eval
from divr.math_equiv import is_equiv
from divr.prompts import CODE_SPLITS

TERM = re.compile(r"<\|endoftext\|>|<\|eot_id\|>|<\|im_end\|>")
BOX = re.compile(r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*)\}")
ANS = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.DOTALL)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
MATH_SPLITS = ("math500",)


def num(s):
    m = NUM.search(str(s).replace(",", ""))
    try:
        return float(m.group()) if m else None
    except ValueError:
        return None


def segment(text: str) -> str:
    m = TERM.search(text)
    return text[:m.start()] if m else text


def gsm8k_candidate(text: str):
    seg = segment(text)
    cand = None
    a = ANS.search(seg)
    if a:
        bx = BOX.findall(a.group(1))
        cand = bx[-1] if bx else a.group(1)
    if cand is None:
        m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", seg)
        cand = m.group(1) if m else None
    if cand is None:
        bx = BOX.findall(seg)
        cand = bx[-1] if bx else None
    if cand is None:
        nums = NUM.findall(seg.replace(",", ""))
        cand = nums[-1] if nums else None
    return cand


def gsm8k_correct(text: str, gt: str) -> int:
    cand = gsm8k_candidate(text)
    p, t = (num(cand) if cand is not None else None), num(gt)
    return int(p is not None and t is not None and abs(p - t) < 1e-4)


def math_candidate(text: str):
    seg = segment(text)
    a = ANS.search(seg)
    if a:
        bx = BOX.findall(a.group(1))
        return (bx[-1] if bx else a.group(1)).strip() or None
    bx = BOX.findall(seg)
    return bx[-1].strip() if bx else None


def math_correct(text: str, gt: str) -> int:
    cand = math_candidate(text)
    return int(bool(cand) and is_equiv(cand, gt))


def countdown_expression(text: str):
    seg = segment(text)
    a = ANS.search(seg)
    bx = BOX.findall(a.group(1)) if a else BOX.findall(seg)
    return bx[-1].strip() if bx else None


def countdown_correct(text: str, gt: str) -> int:
    """gt = "n1,n2,n3|target"."""
    nums, target = gt.split("|")
    nums = sorted(int(x) for x in nums.split(","))
    eq = countdown_expression(text)
    if not eq or not re.fullmatch(r"[\d+\-*/().\s]+", eq) or sorted(int(n) for n in re.findall(r"\d+", eq)) != nums:
        return 0
    try:
        return int(abs(eval(eq, {"__builtins__": None}, {}) - float(target)) < 1e-5)
    except Exception:
        return 0


def is_correct(split: str, text: str, gt: str) -> int:
    if split in CODE_SPLITS:
        return code_eval.correct(split, text, gt)
    if split == "countdown":
        return countdown_correct(text, gt)
    if split in MATH_SPLITS:
        return math_correct(text, gt)
    return gsm8k_correct(text, gt)


def vote_key(split: str, text: str):
    """The answer key that groups equivalent responses for voting."""
    if split in CODE_SPLITS:
        return code_eval.answer_key(text)
    if split == "countdown":
        e = countdown_expression(text)
        return re.sub(r"\s+", "", e) if e else None
    if split in MATH_SPLITS:
        return math_candidate(text)
    v = num(gsm8k_candidate(text) or "")
    return None if v is None else round(v, 4)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
