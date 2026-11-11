#!/usr/bin/env python3
"""Score the candidate pools with a verifier, and evaluate selection (best-of-k).

    python -m divr.score score --name divr_gsm8k --epoch 1 --split test --out_dir outputs
    python -m divr.score summary --name divr_gsm8k --split test --out_dir outputs

score    for every sample of <out_dir>/samples/<split>_t0.7_k4*.jsonl, the verifier's score at each block
         boundary 0..eos_block (the last one is the finished decode) -> <out_dir>/scores/<name>_e<epoch>_<split>.jsonl
summary  pooled AUROC of the final score; within-question pair accuracy (a right decode outscores a wrong one of
         the same question) at the final state and after blocks 1-4; best-of-k by final score, score-weighted
         vote, majority vote, one random sample, pass@k.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import statistics as st

import torch

from divr.data import SPLITS, problems
from divr.family import get_family
from divr.grade import vote_key
from divr.prompts import build_prompt
from divr.sample import GEN
from divr.train_verifier import build_model, load_samples, make_head, pooled, state_ids


def score(args):
    vdir = pathlib.Path(args.out_dir) / "verifiers" / args.name
    meta = json.load(open(vdir / "meta.json"))
    lora = str(vdir / f"epoch{args.epoch}" / "lora") if meta["mode"] == "lora" else None
    tok, model = build_model(meta["mode"], meta["model"], meta.get("adapter"), lora_dir=lora)
    model.eval()
    head = make_head(meta.get("hidden_size", 4096)).cuda()
    head.load_state_dict(torch.load(vdir / f"epoch{args.epoch}" / "head.pt", map_location="cuda", weights_only=True))
    head.eval()
    by = load_samples(args.out_dir, args.split, args.sample_tag, args.gen_length)
    qs = {pid: q for pid, q, _ in problems(args.split, 0, data_dir=args.data_dir)}
    mask_id = get_family(meta["model"]).mask_id
    sdir = pathlib.Path(args.out_dir) / "scores"
    sdir.mkdir(parents=True, exist_ok=True)
    fn = sdir / f"{args.name}_e{args.epoch}{args.sample_tag}_{args.split}.jsonl"
    done = {json.loads(x)["pid"] for x in open(fn)} if fn.exists() else set()
    with open(fn, "a") as out:
        for n, pid in enumerate(sorted(by)):
            if pid in done:
                continue
            p_ids = tok(build_prompt(tok, args.split, qs[pid]))["input_ids"]
            states = [(i, b) for i, r in enumerate(by[pid]) for b in range(r["eos_block"] + 1)]
            vals = []
            for c in range(0, len(states), args.chunk):
                lst = [state_ids(p_ids, by[pid][i]["resp"], b, args.gen_length, mask_id) for i, b in states[c:c + args.chunk]]
                with torch.no_grad():
                    f = pooled(model, torch.tensor(lst, device="cuda"), len(p_ids), grad=False)
                    vals += torch.sigmoid(head(f).squeeze(-1)).tolist()
            for i, r in enumerate(by[pid]):
                ks = [k for k, (ii, _) in enumerate(states) if ii == i]
                out.write(json.dumps({"pid": pid, "s": r["s"], "correct": r["correct"], "answer": vote_key(args.split, r["text"]),
                                      "eos_block": r["eos_block"], "values": [round(vals[k], 5) for k in ks]}) + "\n")
            out.flush()
            if (n + 1) % 100 == 0:
                print(f"[score] {args.split} {n + 1}/{len(by)}", flush=True)


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def summary(args):
    sdir = pathlib.Path(args.out_dir) / "scores"
    files = sorted(glob.glob(str(sdir / f"{args.name}_e*{args.sample_tag}_{args.split}.jsonl")))
    res = {}
    for f in files:
        rs = [json.loads(x) for x in open(f)]
        by = collections.defaultdict(list)
        for r in rs:
            by[r["pid"]].append(r)
        fin = lambda r, b=None: r["values"][-1] if b is None else r["values"][min(b, len(r["values"]) - 1)]   # noqa: E731
        pairs = {b: [] for b in (None, 0, 1, 2, 3)}
        for pid in by:
            for b in pairs:
                a = auroc([fin(r, b) for r in by[pid] if r["correct"]], [fin(r, b) for r in by[pid] if not r["correct"]])
                if a is not None:
                    pairs[b].append(a)
        sel = {"best_of_k": [], "weighted_vote": [], "majority_vote": [], "random_sample": [], "pass_at_k": []}
        for pid, S in by.items():
            sel["best_of_k"].append(max(S, key=fin)["correct"])
            votes, wv = collections.Counter(), collections.defaultdict(float)
            for r in S:
                if r["answer"] is not None:
                    votes[r["answer"]] += 1
                    wv[r["answer"]] += fin(r)
            right = {r["answer"] for r in S if r["correct"]}
            sel["majority_vote"].append(int(bool(votes) and votes.most_common(1)[0][0] in right))
            sel["weighted_vote"].append(int(bool(wv) and max(wv, key=wv.get) in right))
            sel["random_sample"].append(st.mean(r["correct"] for r in S))
            sel["pass_at_k"].append(int(bool(right)))
        out = {"n_questions": len(by), "n_samples": len(rs),
               "sample_acc": round(100 * st.mean(r["correct"] for r in rs), 1),
               "auroc_pooled_final": round(auroc([fin(r) for r in rs if r["correct"]], [fin(r) for r in rs if not r["correct"]]) or 0, 3),
               "within_question_pair_acc": {("final" if b is None else f"block{b}"): round(st.mean(v), 3) if v else None
                                            for b, v in pairs.items()},
               "n_mixed_questions": len(pairs[None]),
               **{k: round(100 * st.mean(v), 1) for k, v in sel.items()}}
        res[pathlib.Path(f).stem] = out
    (sdir / f"summary_{args.name}{args.sample_tag}_{args.split}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "summary"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--gen_length", type=int, default=GEN)
    ap.add_argument("--sample_tag", default="", help="the divr.sample --tag of the pools")
    ap.add_argument("--chunk", type=int, default=16, help="states per forward")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()
    {"score": score, "summary": summary}[args.mode](args)


if __name__ == "__main__":
    main()
