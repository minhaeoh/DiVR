#!/usr/bin/env python3
"""DiVR-guided block re-decoding.

Per problem, two decodes in the same process so that they are paired:
  greedy  plain block decoding (d1's low-confidence remasking, 32-token blocks, 16 steps per block)
  guided  the same decode, but after each finished block the verifier scores the canvas; if the score is
          below `theta` and the regeneration budget is not spent, the block is remasked and K alternatives
          are sampled at temperature T from the same prefix; the original and the alternatives are scored
          and the highest-scoring canvas is kept. Decoding then continues.
The generator is never changed: the verifier LoRA is disabled for every decoding forward.

    python -m divr.guided_decode run --name divr_gsm8k --epoch 1 --split test --theta 0.5 --k_alt 4 --max_redos 3 \\
        --shard 0 --n_shards 4 --out_dir outputs
    python -m divr.guided_decode summary --tag divr_gsm8k_e1_th0.5_k4_rb1 --split test --out_dir outputs

Cost accounting: `greedy_forwards` and `guided_forwards` count decoding forwards weighted by batch rows;
the summary adds one verifier read per decoded block and K reads per regeneration.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import pathlib
import re

import torch
import torch.nn.functional as F

from divr.data import SPLITS, problems
from divr.family import cut, get_family, logits_of
from divr.grade import is_correct, mcnemar
from divr.prompts import build_prompt
from divr.sample import BLOCK, GEN, file_tag, num_transfer
from divr.train_verifier import build_model, make_head


class Runner:
    """A generator with its verifier: block decoding, scoring, and the guided loop."""

    def __init__(self, out_dir: str, name: str, epoch: int, gen_length: int = GEN, generator: str = "same"):
        self.gen_length, self.steps = gen_length, gen_length // 2
        vdir = pathlib.Path(out_dir) / "verifiers" / name
        meta = json.load(open(vdir / "meta.json"))
        self.fam = get_family(meta["model"])
        self.gen_model = None
        if generator == "base":                      # the untuned model (no adapter) decodes; the verifier only scores
            from transformers import AutoModel
            self.gen_model = AutoModel.from_pretrained(meta["model"], trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        self.lora = meta["mode"] == "lora"
        lora_dir = str(vdir / f"epoch{epoch}" / "lora") if self.lora else None
        self.tok, self.model = build_model(meta["mode"], meta["model"], meta.get("adapter"), lora_dir=lora_dir)
        self.model.eval()
        self.head = make_head(meta.get("hidden_size", 4096)).cuda()
        self.head.load_state_dict(torch.load(vdir / f"epoch{epoch}" / "head.pt", map_location="cuda", weights_only=True))
        self.head.eval()
        self.forwards = 0                              # decoding forwards x rows
        self.conf = None                               # baselines: commit-time confidence of each placed token

    def gen_ctx(self):
        return self.model.disable_adapter() if self.lora else contextlib.nullcontext()

    @torch.no_grad()
    def decode_block(self, x, pl, b, temperature, gen):
        """Block b of canvas x (rows share the prompt); tokens sampled if temperature > 0."""
        s0, e0 = pl + b * BLOCK, pl + (b + 1) * BLOCK
        spb = self.steps // (self.gen_length // BLOCK)
        mask_id = self.fam.mask_id
        sched = num_transfer(int((x[0, s0:e0] == mask_id).sum()), spb)
        for i in range(spb):
            if self.gen_model is not None:
                logits = logits_of(self.gen_model, x, self.fam)[:, s0:e0].float()
            else:
                with self.gen_ctx():
                    logits = logits_of(self.model, x, self.fam)[:, s0:e0].float()
            self.forwards += x.shape[0]
            if temperature > 0:
                u = torch.rand(logits.shape, generator=gen, device=logits.device, dtype=torch.float64)
                x0 = (logits.double() / temperature - torch.log(-torch.log(u))).argmax(-1)
            else:
                x0 = logits.argmax(-1)
            conf = F.softmax(logits, -1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            conf = torch.where(x[:, s0:e0] == mask_id, conf, torch.full_like(conf, -float("inf")))
            if sched[i] > 0:
                idx = conf.topk(sched[i], dim=1).indices
                x[:, s0:e0].scatter_(1, idx, x0.gather(1, idx))
                if self.conf is not None:
                    self.conf[:, s0:e0].scatter_(1, idx, conf.gather(1, idx))
        return x

    @torch.no_grad()
    def value(self, x, pl):
        """The verifier's score of each canvas row."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = self.model(input_ids=x, output_hidden_states=True).hidden_states[-1]
        return torch.sigmoid(self.head(h[:, pl:].float().mean(1)).squeeze(-1))

    def run(self, prompt_ids, theta, k_alt, temperature, rollback, max_redos, gen):
        """theta = 0 gives the plain greedy decode; a list gives one threshold per block (the last extends)."""
        pl = len(prompt_ids)
        mask_id, eos_ids = self.fam.mask_id, torch.tensor(self.fam.eos_ids, device="cuda")
        x = torch.tensor([prompt_ids + [mask_id] * self.gen_length], device="cuda")
        redos, values = 0, []
        th = theta if isinstance(theta, (list, tuple)) else [theta]
        for b in range(self.gen_length // BLOCK):
            x = self.decode_block(x, pl, b, 0.0, gen)
            v = float(self.value(x, pl)[0])
            theta_b = th[min(b, len(th) - 1)]
            if theta_b > 0 and v < theta_b and redos < max_redos:
                lo = max(0, b - rollback + 1)
                base = x.clone()
                base[:, pl + lo * BLOCK: pl + (b + 1) * BLOCK] = mask_id
                alt = base.repeat(k_alt, 1)
                for bb in range(lo, b + 1):
                    alt = self.decode_block(alt, pl, bb, temperature, gen)
                va = self.value(alt, pl)
                j = int(va.argmax())
                if float(va[j]) > v:
                    x, v = alt[j:j + 1].clone(), float(va[j])
                redos += 1
            values.append(round(v, 4))
            if bool(torch.isin(x[0, pl:pl + (b + 1) * BLOCK], eos_ids).any()):
                break
        return self.tok.decode(x[0, pl:].tolist(), skip_special_tokens=False), redos, values


def run_tag(args) -> str:
    thetas = [float(t) for t in args.theta.split(",")]
    tag = f"{args.name}_e{args.epoch}_th{'-'.join(f'{t:g}' for t in thetas)}_k{args.k_alt}_rb{args.rollback}{file_tag(args.gen_length)}"
    tag += f"_gen{args.generator}" if args.generator != "same" else ""
    tag += f"_T{args.temperature:g}" if args.temperature != 0.7 else ""
    tag += f"_s{args.seed}" if args.seed else ""
    return tag


def run(args):
    r = Runner(args.out_dir, args.name, args.epoch, args.gen_length, args.generator)
    probs = problems(args.split, args.n_problems, data_dir=args.data_dir)
    if args.limit:
        probs = probs[:args.limit]
    probs = probs[args.shard::args.n_shards]
    thetas = [float(t) for t in args.theta.split(",")]
    out_dir = pathlib.Path(args.out_dir) / "guided"
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{run_tag(args)}_{args.split}_shard{args.shard}of{args.n_shards}.jsonl"
    done = {json.loads(x)["pid"] for x in open(fn)} if fn.exists() else set()
    gen = torch.Generator(device="cuda").manual_seed(args.seed * 1000 + args.shard)
    with open(fn, "a") as out:
        for n, (pid, q, gt) in enumerate(probs):
            if pid in done:
                continue
            p_ids = r.tok(build_prompt(r.tok, args.split, q))["input_ids"]
            f0 = r.forwards
            g_text, _, g_vals = r.run(p_ids, 0.0, args.k_alt, args.temperature, args.rollback, args.max_redos, gen)
            f1 = r.forwards
            v_text, redos, v_vals = r.run(p_ids, thetas, args.k_alt, args.temperature, args.rollback, args.max_redos, gen)
            out.write(json.dumps({"pid": pid, "gt": gt,
                                  "greedy_correct": is_correct(args.split, cut(g_text, r.fam), gt),
                                  "guided_correct": is_correct(args.split, cut(v_text, r.fam), gt),
                                  "redos": redos, "greedy_values": g_vals, "guided_values": v_vals,
                                  "greedy_forwards": f1 - f0, "guided_forwards": r.forwards - f1,
                                  "guided_text": cut(v_text, r.fam)}) + "\n")
            out.flush()
            if (n + 1) % 25 == 0:
                print(f"[guided] {args.split} shard {args.shard}: {n + 1}/{len(probs)}", flush=True)


def summary(args):
    out_dir = pathlib.Path(args.out_dir) / "guided"
    rows = [json.loads(x) for f in sorted(glob.glob(str(out_dir / f"{args.tag}_{args.split}_shard*.jsonl"))) for x in open(f)]
    g = [r["greedy_correct"] for r in rows]
    v = [r["guided_correct"] for r in rows]
    fx = sum(1 for a, b in zip(g, v) if b and not a)
    br = sum(1 for a, b in zip(g, v) if a and not b)
    res = {"tag": args.tag, "split": args.split, "n": len(rows), "greedy_acc": round(100 * sum(g) / len(g), 1),
           "guided_acc": round(100 * sum(v) / len(v), 1), "fixed": fx, "broken": br, "p": round(mcnemar(fx, br), 3),
           "share_with_redo": round(sum(r["redos"] > 0 for r in rows) / len(rows), 3),
           "mean_redos": round(sum(r["redos"] for r in rows) / len(rows), 2),
           "compute_vs_greedy": round(sum(r["guided_forwards"] for r in rows) / sum(r["greedy_forwards"] for r in rows), 2)}
    m = re.search(r"_k(\d+)_", args.tag)
    k = int(m.group(1)) if m else 4
    reads = sum(len(r["guided_values"]) + k * r["redos"] for r in rows)
    res["compute_incl_reads_vs_greedy"] = round((sum(r["guided_forwards"] for r in rows) + reads) / sum(r["greedy_forwards"] for r in rows), 2)
    (out_dir / f"summary_{args.tag}_{args.split}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary"])
    ap.add_argument("--name", default="")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--gen_length", type=int, default=GEN, help="canvas length (MATH-500: 512)")
    ap.add_argument("--n_problems", type=int, default=0, help="a fixed random subset of the split (Countdown: 500)")
    ap.add_argument("--generator", default="same", choices=["same", "base"], help="base: the untuned model decodes, the verifier scores")
    ap.add_argument("--theta", default="0.5", help="regenerate when the block score is below this; a comma-separated list gives one per block")
    ap.add_argument("--k_alt", type=int, default=4, help="alternatives per regeneration")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--rollback", type=int, default=1, help="blocks regenerated per event (ending at the current one)")
    ap.add_argument("--max_redos", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="summary: the run tag, <name>_e<epoch>_th<theta>_k<k>_rb<rollback>[_g512]")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()
    {"run": run, "summary": summary}[args.mode](args)


if __name__ == "__main__":
    main()
