"""HumanEval / MBPP: loaders and a test-execution grader.

Splits: "humaneval" (164; openai/openai_humaneval test), "mbpp" (500; google-research-datasets/mbpp "full" test,
task ids 11-510), "mbpp_train" (464; "full" train 374 + validation 90, the code verifier's training problems).
The question of a row is what the model sees; the ground truth is a JSON string with the tests.

The prompt ends by opening a ```python block, so the response is the code until the closing fence. A decode is
correct when the program (HumanEval: the prompt + the response; MBPP: the setup code + the response) passes its
tests in a subprocess with a 10 s timeout and a 2 GB memory limit. Model-written code is executed as-is: run it
where that is acceptable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

TIMEOUT, MEM_BYTES = 10, 2 << 30


def rows(split: str) -> list[tuple[int, str, str]]:
    """(pid, question, gt) from the HF datasets."""
    from datasets import load_dataset
    if split == "humaneval":
        d = load_dataset("openai/openai_humaneval", split="test")
        return [(i, r["prompt"], json.dumps({"prompt": r["prompt"], "test": r["test"], "entry_point": r["entry_point"]}))
                for i, r in enumerate(d)]
    parts = ["test"] if split == "mbpp" else ["train", "validation"]
    out = []
    for part in parts:
        for r in load_dataset("google-research-datasets/mbpp", "full", split=part):
            q = f"Task: {r['text'].strip()}\nTests:\n" + "\n".join(r["test_list"])
            out.append((int(r["task_id"]), q, json.dumps({"test_list": r["test_list"], "setup": r["test_setup_code"] or ""})))
    return sorted(out)


def extract_code(text: str) -> str:
    """The response up to the closing fence (the prompt opened the block)."""
    return text.split("```")[0].strip("\n")


def answer_key(text: str):
    """A voting key for code: hash of the whitespace-normalized program."""
    code = re.sub(r"\s+", " ", extract_code(text)).strip()
    return hashlib.md5(code.encode()).hexdigest()[:12] if code else None


def program(split: str, text: str, gt: str) -> str:
    g = json.loads(gt)
    code = extract_code(text)
    if split == "humaneval":
        return g["prompt"] + "\n" + code + "\n\n" + g["test"] + f"\n\ncheck({g['entry_point']})\n"
    return (g.get("setup") or "") + "\n" + code + "\n\n" + "\n".join(g["test_list"]) + "\n"


def _limits():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    except Exception:
        pass
    os.setsid()


def run_ok(prog: str, timeout: float = TIMEOUT) -> int:
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "prog.py")
        with open(f, "w") as fh:
            fh.write(prog)
        try:
            p = subprocess.Popen([sys.executable, "-W", "ignore", f], cwd=d, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, preexec_fn=_limits, env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"})
        except Exception:
            return 0
        try:
            return int(p.wait(timeout=timeout) == 0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, 9)
            except Exception:
                p.kill()
            p.wait()
            return 0


def correct(split: str, text: str, gt: str) -> int:
    if not extract_code(text).strip():
        return 0
    return run_ok(program(split, text, gt))
