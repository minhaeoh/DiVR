"""Problem sets, as (pid, question, ground truth) rows.

  train / dev     GSM8K train split, partitioned once by seed into 6,973 training and 500 held-out questions
                  (the SFT generator's split; the verifier's 2,000 training questions are a seeded subset of train)
  test            GSM8K test (1,319)
  math500         MATH-500 (HuggingFaceH4/MATH-500)
  countdown       three-number Countdown tasks (Jiayi-Pan/Countdown-Tasks-3to4, train partition); gt = "nums|target"
  humaneval, mbpp, mbpp_train   see code_eval

With --data_dir, a file <data_dir>/<split>.jsonl of [pid, question, gt] rows is read instead of the Hub.
"""
from __future__ import annotations

import json
import pathlib
import random

from divr import code_eval
from divr.prompts import CODE_SPLITS

SPLITS = ("train", "dev", "test", "math500", "countdown") + CODE_SPLITS


def gold_number(answer_field: str) -> str:
    """The text after `####` in a GSM8K answer (the grader compares numerically, so commas may stay)."""
    return answer_field.split("####")[-1].strip()


def load_gsm8k_split(seed: int = 42, n_holdout: int = 500):
    """GSM8K train, partitioned once into (train, holdout) by seed. The same partition serves SFT and the verifier."""
    from datasets import load_dataset
    rows = list(load_dataset("openai/gsm8k", "main", split="train"))
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    hold, train = sorted(idx[:n_holdout]), sorted(idx[n_holdout:])
    return [rows[i] for i in train], [rows[i] for i in hold], {"seed": seed, "n_train": len(train), "n_holdout": len(hold)}


def problems(split: str, n: int = 0, seed: int = 0, data_dir: str | None = None) -> list[tuple[int, str, str]]:
    """(pid, question, gt) rows of a split; n > 0 takes a fixed random subset (seeded), except for test / dev."""
    if data_dir:
        f = pathlib.Path(data_dir) / f"{split}.jsonl"
        if f.exists():
            rows = [tuple(json.loads(x)) for x in open(f)]
            if n and split not in ("test", "dev"):
                random.Random(seed).shuffle(rows)
                rows = sorted(rows[:n])
            return rows
    from datasets import load_dataset
    if split == "math500":
        rows = [(i, r["problem"], r["answer"]) for i, r in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test"))]
    elif split == "countdown":
        d = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
        rows = [(i, f"Numbers: {list(r['nums'])}\nTarget: {r['target']}", f"{','.join(map(str, r['nums']))}|{r['target']}")
                for i, r in enumerate(d) if len(r["nums"]) == 3]
    elif split in CODE_SPLITS:
        rows = code_eval.rows(split)
    elif split == "test":
        rows = [(i, r["question"], gold_number(r["answer"])) for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))]
        return rows
    else:
        train, hold, _ = load_gsm8k_split(42, 500)
        src = train if split == "train" else hold
        rows = [(i, r["question"], gold_number(r["answer"])) for i, r in enumerate(src)]
        if split == "dev":
            return rows
    if n:
        random.Random(seed).shuffle(rows)
        rows = sorted(rows[:n])
    return rows
