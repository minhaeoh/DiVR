#!/usr/bin/env python3
"""Train a DiVR verifier on a generator's own decodes.

A state is one block boundary of one decode: the prompt, the first 32(b+1) response tokens, and masks out to
the canvas length -- the canvas as decoding sees it when block b has just been finished. Its label is whether
that decode's final answer is right (every prefix carries the outcome of its trajectory). The verifier is
a LoRA on the generator's frozen backbone plus a two-layer head on the mean last-layer state over the
response positions, masks included.

  --mode lora    the verifier of the paper: LoRA (r 32, alpha 64) + head
  --mode frozen  head-only control (the generator's unchanged hidden states)

Loss: class-balanced binary cross-entropy over all states, plus a within-question pairwise ranking term
(at the same block, a decode that ends right should outscore one that ends wrong). Each step takes one
question: its k samples at `blocks_per_step` random block indices.

    python -m divr.train_verifier --mode lora --name divr_gsm8k --model GSAI-ML/LLaDA-8B-Base --adapter <sft adapter> \\
        --split train --n_problems 2000 --holdout 100 --out_dir outputs
    python -m divr.train_verifier --mode lora --name divr_code --model ... --adapter ... --split mbpp_train --holdout 64 --out_dir outputs

The generator's samples are read from <out_dir>/samples (divr.sample with the same --split / --tag).
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from divr.data import problems
from divr.family import get_family
from divr.prompts import build_prompt
from divr.sample import BLOCK, GEN, file_tag

NO_ADAPTER = ("", "none", "base", None)


def make_head(dim: int = 4096) -> nn.Module:
    """MLP on the pooled last hidden state; dim = the model's hidden size (LLaDA 4096, Dream 3584)."""
    return nn.Sequential(nn.Linear(dim, 1024), nn.SiLU(), nn.Linear(1024, 1)).float()


def hidden_size(model) -> int:
    return model.get_input_embeddings().weight.shape[1]


def load_samples(out_dir: str, split: str, sample_tag: str = "", gen_length: int = GEN, temperature: float = 0.7, k: int = 4) -> dict[int, list[dict]]:
    pattern = str(pathlib.Path(out_dir) / "samples" / f"{split}_t{temperature:g}_k{k}{file_tag(gen_length, sample_tag)}_shard*.jsonl")
    by = collections.defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        for x in open(f):
            r = json.loads(x)
            by[r["pid"]].append(r)
    if not by:
        raise SystemExit(f"no samples match {pattern}")
    return by


def state_ids(prompt_ids: list[int], resp: list[int], b: int, gen_length: int, mask_id: int) -> list[int]:
    """The canvas right after block b is finished: blocks <= b committed, the rest masked."""
    n = BLOCK * (b + 1)
    return prompt_ids + resp[:n] + [mask_id] * (gen_length - n)


def build_model(mode: str, model_name: str, adapter: str | None, lora_dir: str | None = None, trainable: bool = False):
    """(tokenizer, model). The generator (model_name with its adapter merged in, if any), then in lora mode the
    verifier LoRA (fresh, or loaded from lora_dir). With the verifier LoRA disabled the weights are the generator's."""
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    if adapter not in NO_ADAPTER:
        model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.bfloat16).merge_and_unload().cuda().eval()
    if mode == "lora":
        if lora_dir:
            model = PeftModel.from_pretrained(model, lora_dir, is_trainable=trainable, torch_dtype=torch.bfloat16)
        else:
            model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                                                     target_modules=list(get_family(model_name).lora_modules)))
    for n, p in model.named_parameters():
        p.requires_grad_(trainable and "lora_" in n)
    return tok, model


def pooled(model, ids: torch.Tensor, prompt_len: int, grad: bool) -> torch.Tensor:
    """Mean of the last hidden layer over the response positions, float32 (B, hidden)."""
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(input_ids=ids, output_hidden_states=True).hidden_states[-1]
    return h[:, prompt_len:].float().mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="lora", choices=["frozen", "lora"])
    ap.add_argument("--name", required=True, help="verifier name; saved under <out_dir>/verifiers/<name>")
    ap.add_argument("--model", required=True, help="the generator's HF model id")
    ap.add_argument("--adapter", default="none", help="the generator's LoRA adapter (merged under the verifier LoRA), or none")
    ap.add_argument("--split", default="train", help="the samples to train on (divr.sample --split)")
    ap.add_argument("--n_problems", type=int, default=2000, help="the divr.sample --n_problems of that split")
    ap.add_argument("--sample_tag", default="", help="the divr.sample --tag of the generator's samples")
    ap.add_argument("--gen_length", type=int, default=GEN)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--blocks_per_step", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4, help="questions per optimizer step")
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_lora", type=float, default=1e-4)
    ap.add_argument("--lambda_pair", type=float, default=1.0)
    ap.add_argument("--holdout", type=int, default=100, help="hold out the last N questions; report their pair accuracy per epoch")
    ap.add_argument("--limit", type=int, default=0, help="questions per epoch (smoke tests)")
    ap.add_argument("--max_states", type=int, default=0, help="cap states per step (memory)")
    ap.add_argument("--act_ckpt", action="store_true", help="activation checkpointing (512-token states)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    out = pathlib.Path(args.out_dir) / "verifiers" / args.name
    out.mkdir(parents=True, exist_ok=True)

    by = load_samples(args.out_dir, args.split, args.sample_tag, args.gen_length)
    qs = {pid: q for pid, q, _ in problems(args.split, args.n_problems, data_dir=args.data_dir)}
    pids = sorted(by)
    held = pids[len(pids) - args.holdout:] if args.holdout else []
    pids = pids[:len(pids) - args.holdout] if args.holdout else pids
    labels = [r["correct"] for rs in by.values() for r in rs]
    pos = sum(labels) / len(labels)
    w_pos, w_neg = 0.5 / pos, 0.5 / (1 - pos)
    mixed = sum(1 for rs in by.values() if 0 < sum(r["correct"] for r in rs) < len(rs))
    print(f"[verifier] {len(pids)} questions, {len(labels)} decodes, share right {pos:.3f}, mixed-outcome questions {mixed}", flush=True)

    tok, model = build_model(args.mode, args.model, args.adapter, trainable=args.mode == "lora")
    mask_id = get_family(args.model).mask_id
    if args.act_ckpt:
        llada = [m for m in model.modules() if type(m).__name__ == "LLaDAModel"]
        if llada:
            llada[0].set_activation_checkpointing("whole_layer")
        else:
            model.gradient_checkpointing_enable()
    head = make_head(hidden_size(model)).cuda()
    groups = [{"params": head.parameters(), "lr": args.lr_head}]
    lora = [p for p in model.parameters() if p.requires_grad]
    if lora:
        groups.append({"params": lora, "lr": args.lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    n_per_epoch = min(len(pids), args.limit) if args.limit else len(pids)
    total = args.epochs * math.ceil(n_per_epoch / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / total))
    json.dump({**vars(args), "share_right": pos, "n_questions": len(pids), "n_decodes": len(labels), "mixed_questions": mixed,
               "lora_modules": list(get_family(args.model).lora_modules), "hidden_size": hidden_size(model),
               "head": f"{hidden_size(model)}-1024-1 SiLU"}, open(out / "meta.json", "w"), indent=1)
    holdout_acc = {}

    step, t0 = 0, time.time()
    log = collections.defaultdict(list)
    for ep in range(args.epochs):
        order = pids[:]
        rng.shuffle(order)
        order = order[:n_per_epoch]
        for j, pid in enumerate(order):
            rs = by[pid]
            p_ids = tok(build_prompt(tok, args.split, qs[pid]))["input_ids"]
            top = max(r["eos_block"] for r in rs)
            bs = rng.sample(range(top + 1), min(args.blocks_per_step, top + 1))
            ids, y, grp = [], [], []
            for b in bs:
                for r in rs:
                    ids.append(state_ids(p_ids, r["resp"], min(b, r["eos_block"]), args.gen_length, mask_id))
                    y.append(float(r["correct"]))
                    grp.append(b)
            if args.max_states:
                ids, y, grp = ids[:args.max_states], y[:args.max_states], grp[:args.max_states]
            y = torch.tensor(y, device="cuda")
            feats = pooled(model, torch.tensor(ids, device="cuda"), len(p_ids), grad=bool(lora))
            s = head(feats).squeeze(-1)
            bce = F.binary_cross_entropy_with_logits(s, y, reduction="none")
            loss = (bce * torch.where(y > 0, w_pos, w_neg)).mean()
            pair, acc = [], []
            for b in bs:                                   # right vs wrong decodes of this question, same block
                k = torch.tensor([g == b for g in grp], device="cuda")
                sp, sn = s[k & (y > 0)], s[k & (y == 0)]
                if len(sp) and len(sn):
                    d = sp[:, None] - sn[None, :]
                    pair.append(F.softplus(-d).mean())
                    acc.append(float((d > 0).float().mean()))
            if pair:
                loss = loss + args.lambda_pair * torch.stack(pair).mean()
                log["pair_acc"].append(sum(acc) / len(acc))
            (loss / args.accum).backward()
            log["loss"].append(float(loss))
            if (j + 1) % args.accum == 0 or j + 1 == len(order):
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if (j + 1) % 100 == 0:
                print(f"[verifier] ep {ep} {j + 1}/{len(order)} loss {sum(log['loss'][-100:]) / 100:.4f} "
                      f"pair_acc(train, last 100 mixed) {sum(log['pair_acc'][-100:]) / max(1, len(log['pair_acc'][-100:])):.3f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
        d = out / f"epoch{ep}"
        d.mkdir(exist_ok=True)
        torch.save(head.state_dict(), d / "head.pt")
        if lora:
            model.save_pretrained(d / "lora")
        print(f"[verifier] saved {d}", flush=True)
        if held:                                     # held-out within-question pair accuracy at the final state
            model.eval()
            ok = []
            with torch.no_grad():
                for pid in held:
                    rs = by[pid]
                    if not 0 < sum(r["correct"] for r in rs) < len(rs):
                        continue
                    p_ids = tok(build_prompt(tok, args.split, qs[pid]))["input_ids"]
                    lst = [state_ids(p_ids, r["resp"], r["eos_block"], args.gen_length, mask_id) for r in rs]
                    v = head(pooled(model, torch.tensor(lst, device="cuda"), len(p_ids), grad=False)).squeeze(-1).tolist()
                    ok += [v[i] > v[j] for i in range(len(rs)) for j in range(len(rs)) if rs[i]["correct"] and not rs[j]["correct"]]
            model.train()
            holdout_acc[str(ep)] = round(sum(ok) / max(1, len(ok)), 4)
            json.dump(holdout_acc, open(out / "holdout.json", "w"), indent=1)
            print(f"[verifier] epoch {ep} held-out pair accuracy {holdout_acc[str(ep)]:.3f} ({len(ok)} pairs, {len(held)} questions)", flush=True)


if __name__ == "__main__":
    main()
