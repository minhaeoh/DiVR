#!/usr/bin/env python3
"""Sample complete responses from a generator, graded by their final answer.

Two uses: the verifier's training data (the generator's own decodes of its training questions, labelled
by outcome) and the best-of-k candidate pools of the evaluation sets.

The decoder is d1's low-confidence remasking (256-token canvas, 32-token blocks, 2 tokens per step), with
tokens sampled from softmax(logits / T) inside the current block (T = 0 reproduces greedy decoding), and an
early stop once every row of the batch has committed an end token in a finished block.

    python -m divr.sample --model GSAI-ML/LLaDA-8B-Base --adapter <sft adapter> --split train --n_problems 2000 \\
        --k 4 --temperature 0.7 --shard 0 --n_shards 4 --out_dir outputs
    python -m divr.sample --model GSAI-ML/LLaDA-8B-Instruct --split test --k 4 --out_dir outputs

MATH-500 uses a 512-token canvas (--gen_length 512); its files carry _g512 in their names.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import torch
import torch.nn.functional as F

from divr.data import SPLITS, problems
from divr.family import Family, LLADA, cut, get_family, logits_of
from divr.grade import is_correct
from divr.prompts import build_prompt

GEN, BLOCK = 256, 32


def num_transfer(mask_count: int, steps: int) -> list[int]:
    """Tokens committed at each of `steps` steps of a block (d1's schedule)."""
    base, rem = mask_count // steps, mask_count % steps
    return [base + (1 if i < rem else 0) for i in range(steps)]


def file_tag(gen_length: int, tag: str = "") -> str:
    return tag + (f"_g{gen_length}" if gen_length != GEN else "")


def sample_file(out_dir: str, split: str, temperature: float, k: int, gen_length: int, tag: str, shard: int, n_shards: int) -> pathlib.Path:
    return pathlib.Path(out_dir) / "samples" / f"{split}_t{temperature:g}_k{k}{file_tag(gen_length, tag)}_shard{shard}of{n_shards}.jsonl"


def load_generator(model_name: str, adapter: str | None):
    """(tokenizer, model): the HF model with an optional LoRA adapter (unmerged) on top."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
    if adapter and adapter not in ("none", "base"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, torch_dtype=torch.bfloat16).cuda().eval()
    return tok, model


@torch.no_grad()
def sample_decode(model, x, prompt_len, temperature, gen: torch.Generator, gen_length: int = GEN, steps: int | None = None,
                  fam: Family = LLADA):
    """d1's sampler on canvas x (B, L): tokens sampled at `temperature` (0 = argmax), early stop at the end token."""
    steps = steps or gen_length // 2
    nb, spb = gen_length // BLOCK, steps // (gen_length // BLOCK)
    mask_id, eos_ids = fam.mask_id, torch.tensor(fam.eos_ids, device=x.device)
    for b in range(nb):
        s0, e0 = prompt_len + b * BLOCK, prompt_len + (b + 1) * BLOCK
        sched = num_transfer(int((x[0, s0:e0] == mask_id).sum()), spb)
        for i in range(spb):
            logits = logits_of(model, x, fam)[:, s0:e0].float()
            if temperature > 0:                        # argmax(l / T + Gumbel) ~ softmax(l / T)
                u = torch.rand(logits.shape, generator=gen, device=logits.device, dtype=torch.float64)
                x0 = (logits.double() / temperature - torch.log(-torch.log(u))).argmax(-1)
            else:
                x0 = logits.argmax(-1)
            conf = F.softmax(logits, -1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(x[:, s0:e0] == mask_id, conf, torch.full_like(conf, -float("inf")))
            if sched[i] > 0:
                idx = conf.topk(sched[i], dim=1).indices
                x[:, s0:e0].scatter_(1, idx, x0.gather(1, idx))
        if bool(torch.isin(x[:, prompt_len:e0], eos_ids).any(1).all()):
            x[:, e0:] = torch.where(x[:, e0:] == mask_id, torch.full_like(x[:, e0:], fam.fill_id), x[:, e0:])
            break
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="generator HF id (LLaDA or Dream)")
    ap.add_argument("--adapter", default="none", help="generator LoRA adapter directory, or none")
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--gen_length", type=int, default=GEN, help="canvas length; steps = length / 2")
    ap.add_argument("--n_problems", type=int, default=0, help="a fixed random subset of the split (0 = all)")
    ap.add_argument("--limit", type=int, default=0, help="first N problems only (smoke tests)")
    ap.add_argument("--k", type=int, default=4, help="samples per problem")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch_problems", type=int, default=2, help="problems per batch (x k samples)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix naming the generator, e.g. _inst")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()
    probs = problems(args.split, args.n_problems, data_dir=args.data_dir)
    if args.limit:
        probs = probs[:args.limit]
    probs = probs[args.shard::args.n_shards]
    GL, ST = args.gen_length, args.gen_length // 2
    fam = get_family(args.model)
    if fam.name == "dream" and args.batch_problems != 1:
        print("[sample] Dream: batch_problems forced to 1 (no attention mask for left padding)", flush=True)
        args.batch_problems = 1
    fn = sample_file(args.out_dir, args.split, args.temperature, args.k, GL, args.tag, args.shard, args.n_shards)
    fn.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["pid"] for x in open(fn)} if fn.exists() else set()
    probs = [p for p in probs if p[0] not in done]
    print(f"[sample] {args.split}: {len(probs)} problems to go -> {fn}", flush=True)
    tok, model = load_generator(args.model, args.adapter)
    gen = torch.Generator(device="cuda").manual_seed(args.seed * 1000 + args.shard)
    with open(fn, "a") as out:
        for j in range(0, len(probs), args.batch_problems):
            batch = probs[j:j + args.batch_problems]
            enc = tok([build_prompt(tok, args.split, q) for _, q, _ in batch for _ in range(args.k)],
                      padding_side="left", padding="longest", return_tensors="pt")["input_ids"].cuda()
            pl = enc.shape[1]
            x = torch.cat([enc, torch.full((enc.shape[0], GL), fam.mask_id, dtype=torch.long, device="cuda")], 1)
            x = sample_decode(model, x, pl, args.temperature, gen, GL, ST, fam)
            for r in range(x.shape[0]):
                pid, q, gt = batch[r // args.k]
                resp = x[r, pl:].tolist()
                text = cut(tok.decode(resp, skip_special_tokens=False), fam)
                eos = min([resp.index(e) for e in fam.eos_ids if e in resp] + [GL])
                out.write(json.dumps({"split": args.split, "pid": pid, "s": r % args.k, "t": args.temperature,
                                      "gt": gt, "resp": resp, "text": text,
                                      "correct": is_correct(args.split, text, gt),
                                      "eos_block": eos // BLOCK, "gen": GL}) + "\n")
            out.flush()
            if (j // args.batch_problems) % 10 == 0:
                print(f"[sample] shard {args.shard}: {j + len(batch)}/{len(probs)}", flush=True)


if __name__ == "__main__":
    main()
