#!/usr/bin/env python3
"""Baselines in the same harness (same prompt, generator, decoding law, grader, and cost accounting).

A. The guided loop of guided_decode.py with another signal (`run --method guided --score conf|svf`):
     conf  the generator's own commitment confidence, exp(mean log p) of the tokens committed so far
     svf   prompted self-verification: the generator judges the partial solution (Prism's judge prompt),
           score = p(Yes) / (p(Yes) + p(No)) read at a mask token in the assistant turn
   `--score value` reproduces DiVR (for a paired comparison in one process).
B. SR-SMC (Luo et al., 2026; `run --method smc --particles 4 --temperature 1.0`): N particles decoded in
   parallel; each particle's log-weight accumulates log p of the tokens it commits; after every block, if the
   effective sample size is below N/2, multinomial resampling and the weights reset; the output is the particle
   with the highest total log-confidence.
C. Prism (Bai et al., 2026; `run --method prism`): hierarchical trajectory search with self-verified pruning,
   ported from the released hts_sampler.py with its GSM8K settings (N 16, final K 4, survivor k 2, pruning window
   [10%, 60%) of the first block's steps, interval 3, decay 1.8, temperature 0.7). The K finished trajectories are
   ranked by self-verification (the released code's answer, `method_correct`); the majority vote over them (the
   paper's answer) is `majority_correct`.
D. Selection on the existing candidate pools (`select --signal loglik|svf`): best-of-k by re-read sequence
   likelihood or by self-verification.

    python -m divr.baselines run --name divr_gsm8k --epoch 1 --split test --method smc --particles 4 --temperature 1.0 --out_dir outputs
    python -m divr.baselines run --name divr_gsm8k --epoch 1 --split test --method prism --out_dir outputs
    python -m divr.baselines summary --tag <tag> --split test --out_dir outputs
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib

import torch
import torch.nn.functional as F

from divr.data import SPLITS, problems
from divr.family import argmax_of, cut, logits_of
from divr.grade import is_correct, mcnemar, vote_key
from divr.guided_decode import Runner
from divr.prompts import CODE_SPLITS, build_prompt
from divr.sample import BLOCK, GEN, file_tag, num_transfer

SVF_MATH = """You are an expert mathematician and competition judge. Your task is to evaluate a proposed mathematical solution for a given problem based on its logical rigor and accuracy.

[Math Problem]
{prompt}
[/Math Problem]

[Proposed Mathematical Solution]
{solution}
[/Proposed Mathematical Solution]

**Analysis Steps:**
1. Reasoning Validity: Are the logical steps and mathematical properties applied correctly?
2. Calculation Accuracy: Are the intermediate calculations or algebraic manipulations accurate?
3. Goal Alignment: Does the current reasoning path directly lead toward the final answer required by the problem?

**Conclusion**: Based on your analysis, is this solution path sound and likely to result in the correct final answer? Answer with a single word: Yes or No.
**Answer:**"""
SVF_CODE = """You are an expert code reviewer. Your task is to evaluate whether a proposed Python solution correctly solves the task.

[Task]
{prompt}
[/Task]

[Proposed Solution]
{solution}
[/Proposed Solution]

**Conclusion**: Based on your analysis, is the solution likely to be fully correct? Answer with a single word: Yes or No.
**Answer:**"""


class Baselines(Runner):
    """The guided Runner plus the alternative signals and the two search methods."""

    def __init__(self, out_dir, name, epoch, gen_length=GEN, generator="same"):
        super().__init__(out_dir, name, epoch, gen_length, generator)
        ids = lambda t: [self.tok.encode(s, add_special_tokens=False)[-1] for s in t]  # noqa: E731
        self.yes_ids, self.no_ids = sorted(set(ids(["Yes", " Yes", "YES"]))), sorted(set(ids(["No", " No", "NO"])))
        self.reads = 0                                   # signal forwards (verifier / judge / likelihood), x rows

    # ---------------------------------------------------------------- generator forwards (verifier LoRA disabled)
    def gen_logits(self, x, sl=None):
        if self.gen_model is not None:
            return logits_of(self.gen_model, x, self.fam, sl)
        with self.gen_ctx():
            return logits_of(self.model, x, self.fam, sl)

    def gen_argmax(self, x):
        if self.gen_model is not None:
            return argmax_of(self.gen_model, x, self.fam)
        with self.gen_ctx():
            return argmax_of(self.model, x, self.fam)

    # ---------------------------------------------------------------- signals
    def conf_score(self, conf, pl, upto):
        """exp(mean log p) over the committed response positions [pl, upto) of each row."""
        c = conf[:, pl:upto].clamp_min(1e-9)
        placed = conf[:, pl:upto] > 0
        return torch.exp((torch.log(c) * placed).sum(1) / placed.sum(1).clamp_min(1))

    def svf(self, question, texts, split):
        """p(Yes) of the generator judging each partial or finished solution (one forward per text)."""
        tpl = SVF_CODE if split in CODE_SPLITS else SVF_MATH
        out = []
        for t in texts:
            sol = t if len(t) <= 2000 else t[:500] + "\n...[truncated]...\n" + t[-1500:]
            msgs = [{"role": "user", "content": tpl.format(prompt=question.strip(), solution=sol.strip())}]
            ids = self.tok(self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))["input_ids"]
            x = torch.tensor([ids + [self.fam.mask_id]], device="cuda")
            lg = self.gen_logits(x)[0, -1].float()
            self.reads += 1
            y, n = lg[self.yes_ids].max(), lg[self.no_ids].max()
            out.append(float(torch.softmax(torch.stack([y, n]), 0)[0]))
        return torch.tensor(out, device="cuda")

    def loglik(self, x, pl):
        """Re-read likelihood: mean log p(token | full canvas) over the committed response positions."""
        lp = F.log_softmax(self.gen_logits(x)[:, pl:].float(), -1).gather(-1, x[:, pl:].unsqueeze(-1)).squeeze(-1)
        self.reads += x.shape[0]
        keep = (x[:, pl:] != self.fam.mask_id) & (x[:, pl:] != self.fam.fill_id)
        return (lp * keep).sum(1) / keep.sum(1).clamp_min(1)

    def text(self, x, pl, upto=None):
        ids = x[0, pl:upto].tolist() if upto else x[0, pl:].tolist()
        return cut(self.tok.decode([t for t in ids if t != self.fam.mask_id], skip_special_tokens=False), self.fam)

    # ---------------------------------------------------------------- A. the guided loop with any signal
    def guided(self, question, split, prompt_ids, score, theta, k_alt, temperature, max_redos, gen):
        pl = len(prompt_ids)
        mask_id, eos_ids = self.fam.mask_id, torch.tensor(self.fam.eos_ids, device="cuda")
        x = torch.tensor([prompt_ids + [mask_id] * self.gen_length], device="cuda")
        self.conf = torch.zeros_like(x, dtype=torch.float32)
        redos, values = 0, []

        def read(xx, upto):
            if score == "value":
                self.reads += xx.shape[0]
                return self.value(xx, pl)
            if score == "conf":
                return self.conf_score(self.conf, pl, upto)
            return self.svf(question, [self.text(xx[i:i + 1], pl, upto) for i in range(xx.shape[0])], split)

        for b in range(self.gen_length // BLOCK):
            x = self.decode_block(x, pl, b, 0.0, gen)
            upto = pl + (b + 1) * BLOCK
            v = float(read(x, upto)[0])
            if theta > 0 and v < theta and redos < max_redos:
                x_conf = self.conf.clone()
                base, base_c = x.clone(), self.conf.clone()
                base[:, pl + b * BLOCK: upto] = mask_id
                base_c[:, pl + b * BLOCK: upto] = 0
                alt, self.conf = base.repeat(k_alt, 1), base_c.repeat(k_alt, 1)
                alt = self.decode_block(alt, pl, b, temperature, gen)
                va = read(alt, upto)
                j = int(va.argmax())
                if float(va[j]) > v:
                    x, v, self.conf = alt[j:j + 1].clone(), float(va[j]), self.conf[j:j + 1].clone()
                else:
                    self.conf = x_conf
                redos += 1
            values.append(round(v, 4))
            if bool(torch.isin(x[0, pl:upto], eos_ids).any()):
                break
        self.conf = None
        return self.text(x, pl), redos, values

    # ---------------------------------------------------------------- B. SR-SMC
    def smc(self, prompt_ids, particles, temperature, gen):
        pl = len(prompt_ids)
        mask_id, eos_ids = self.fam.mask_id, torch.tensor(self.fam.eos_ids, device="cuda")
        x = torch.tensor([prompt_ids + [mask_id] * self.gen_length], device="cuda").repeat(particles, 1)
        self.conf = torch.zeros_like(x, dtype=torch.float32)
        log_w = torch.zeros(particles, device="cuda")
        resamples = 0
        for b in range(self.gen_length // BLOCK):
            s0, e0 = pl + b * BLOCK, pl + (b + 1) * BLOCK
            x = self.decode_block(x, pl, b, temperature, gen)
            log_w = log_w + torch.log(self.conf[:, s0:e0].clamp_min(1e-9)).sum(1)
            if particles > 1:
                w = torch.exp(log_w - log_w.max()); w = w / w.sum()
                if 1.0 / (w ** 2).sum() < 0.5 * particles:
                    k = torch.multinomial(w, particles, replacement=True, generator=gen)
                    x, self.conf = x[k].clone(), self.conf[k].clone()
                    log_w.zero_(); resamples += 1
            if bool(torch.isin(x[:, pl:e0], eos_ids).any(1).all()):
                break
        tot = torch.log(self.conf[:, pl:].clamp_min(1e-9)) * (self.conf[:, pl:] > 0)
        i = int(tot.sum(1).argmax())
        text = self.text(x[i:i + 1], pl)
        self.conf = None
        return text, resamples

    # ---------------------------------------------------------------- C. Prism (hierarchical search + self-verification)
    def prism(self, question, split, prompt_ids, gen, N=16, final_k=4, survivor_k=2, start_pct=0.1, end_pct=0.6,
              interval=3, decay=1.8, temperature=0.7, remask=2):
        pl = len(prompt_ids)
        mask_id, eos_ids = self.fam.mask_id, torch.tensor(self.fam.eos_ids, device="cuda")
        spb = self.steps // (self.gen_length // BLOCK)
        x = torch.tensor([prompt_ids + [mask_id] * self.gen_length], device="cuda").repeat(N, 1)
        self.conf = torch.zeros_like(x, dtype=torch.float32)
        ts, tr = int(spb * start_pct), int(spb * end_pct)
        next_allowed, prunes = ts, []

        def structure(t):                               # Prism's structure bonus (math task)
            s = 0.0
            if "\\boxed{" in t: s += 0.1
            if "The answer is" in t: s += 0.05
            if len(t.strip()) < 10: return -0.1
            if t.count("Step") > 15: s -= 0.2
            return s

        for b in range(self.gen_length // BLOCK):
            s0, e0 = pl + b * BLOCK, pl + (b + 1) * BLOCK
            sched = num_transfer(int((x[0, s0:e0] == mask_id).sum()), spb)
            for i in range(spb):
                if b == 0 and ts <= i < tr and i >= next_allowed:
                    target = max(final_k, math.ceil(N * decay ** -(i - ts)))
                    if x.shape[0] > target:
                        am = self.gen_argmax(x); self.reads += x.shape[0]
                        rough = torch.where(x == mask_id, am, x)
                        texts = [self.text(rough[j:j + 1], pl) for j in range(x.shape[0])]
                        sc = self.svf(question, texts, split).tolist()
                        cands = sorted([{"s": sc[j] + structure(texts[j]), "i": j,
                                         "key": texts[j].strip().replace(" ", "").replace("\n", "")[:200]} for j in range(x.shape[0])],
                                       key=lambda c: -c["s"])
                        sel, seen = [], set()
                        for c in cands:
                            if len(sel) >= survivor_k: break
                            if c["key"] not in seen: sel.append(c["i"]); seen.add(c["key"])
                        for c in cands:
                            if len(sel) >= survivor_k: break
                            if c["i"] not in sel: sel.append(c["i"])
                        x, self.conf = self._branch(x, self.conf, sel, target, pl, remask, gen)
                        prunes.append({"step": i, "width": int(x.shape[0])})
                        next_allowed = i + 1 + interval
                x = self._step(x, s0, e0, sched[i], temperature, gen)
            while bool((x[:, s0:e0] == mask_id).any()):      # re-masked children may need extra steps to fill the block
                x = self._step(x, s0, e0, 2, temperature, gen)
            if bool(torch.isin(x[:, pl:e0], eos_ids).any(1).all()):
                break
        texts = [self.text(x[j:j + 1], pl) for j in range(x.shape[0])]
        sc = self.svf(question, texts, split).tolist()
        order = sorted(range(len(texts)), key=lambda j: -(sc[j] + structure(texts[j])))
        self.conf = None
        return texts[order[0]], texts, [round(sc[j], 4) for j in order], prunes

    @torch.no_grad()
    def _step(self, x, s0, e0, k, temperature, gen):
        """One block-decoding step for all rows, recording commit-time confidences."""
        logits = self.gen_logits(x, (s0, e0)).float()
        self.forwards += x.shape[0]
        if temperature > 0:
            u = torch.rand(logits.shape, generator=gen, device=logits.device, dtype=torch.float64)
            x0 = (logits.double() / temperature - torch.log(-torch.log(u))).argmax(-1)
        else:
            x0 = logits.argmax(-1)
        conf = F.softmax(logits, -1).gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        masked = x[:, s0:e0] == self.fam.mask_id
        conf = torch.where(masked, conf, torch.full_like(conf, -float("inf")))
        for r in range(x.shape[0]):
            kk = min(int(k), int(masked[r].sum()))
            if kk > 0:
                idx = conf[r].topk(kk).indices
                x[r, s0 + idx] = x0[r, idx]
                self.conf[r, s0 + idx] = conf[r, idx]
        return x

    def _branch(self, x, conf, parents, target, pl, remask, gen):
        """Parents survive; extra copies re-mask `remask` of their lowest-confidence committed tokens."""
        n = len(parents); base, rem = target // n, target % n
        xs, cs = [], []
        for i, p in enumerate(parents):
            count = base + (1 if i < rem else 0)
            if count == 0: continue
            xs.append(x[p:p + 1]); cs.append(conf[p:p + 1])
            placed = ((x[p, pl:] != self.fam.mask_id)).nonzero(as_tuple=True)[0]
            for _ in range(count - 1):
                xx, cc = x[p:p + 1].clone(), conf[p:p + 1].clone()
                if len(placed):
                    pool = min(remask * 3, len(placed))
                    low = conf[p, pl + placed].topk(pool, largest=False).indices
                    pick = low[torch.randperm(pool, generator=gen, device="cuda")[:min(remask, pool)]]
                    pos = pl + placed[pick]
                    xx[0, pos] = self.fam.mask_id; cc[0, pos] = 0.0
                xs.append(xx); cs.append(cc)
        return torch.cat(xs), torch.cat(cs)


def majority(texts, split):
    votes = collections.Counter(a for a in (vote_key(split, t) for t in texts) if a is not None)
    if not votes:
        return texts[0]
    top = votes.most_common(1)[0][0]
    return next(t for t in texts if vote_key(split, t) == top)


def run(args):
    r = Baselines(args.out_dir, args.name, args.epoch, args.gen_length, args.generator)
    probs = problems(args.split, args.n_problems, data_dir=args.data_dir)
    if args.limit:
        probs = probs[:args.limit]
    probs = probs[args.shard::args.n_shards]
    if args.method == "guided":
        tag = f"bl-{args.score}_th{args.theta:g}_k{args.k_alt}_T{args.temperature:g}"
    elif args.method == "smc":
        tag = f"bl-smc_n{args.particles}_T{args.temperature:g}"
    else:
        tag = f"bl-prism_n{args.prism_n}_k{args.prism_k}_T{args.temperature:g}"
    tag += file_tag(args.gen_length) + (f"_gen{args.generator}" if args.generator != "same" else "") + f"_{args.name}_e{args.epoch}"
    tag += f"_s{args.seed}" if args.seed else ""
    out_dir = pathlib.Path(args.out_dir) / "guided"
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"{tag}_{args.split}_shard{args.shard}of{args.n_shards}.jsonl"
    done = {json.loads(l)["pid"] for l in open(fn)} if fn.exists() else set()
    gen = torch.Generator(device="cuda").manual_seed(args.seed * 1000 + args.shard)
    with open(fn, "a") as out:
        for n, (pid, q, gt) in enumerate(probs):
            if pid in done:
                continue
            p_ids = r.tok(build_prompt(r.tok, args.split, q))["input_ids"]
            f0 = r.forwards
            g_text, _, _ = r.run(p_ids, 0.0, args.k_alt, args.temperature, 1, 0, gen)   # paired greedy
            f1, rd1 = r.forwards, r.reads
            rec = {"pid": pid, "gt": gt, "greedy_correct": is_correct(args.split, cut(g_text, r.fam), gt)}
            if args.method == "guided":
                text, redos, vals = r.guided(q, args.split, p_ids, args.score, args.theta, args.k_alt, args.temperature, args.max_redos, gen)
                rec.update({"redos": redos, "values": vals})
            elif args.method == "smc":
                text, resamples = r.smc(p_ids, args.particles, args.temperature, gen)
                rec.update({"resamples": resamples})
            else:
                text, texts, scores, prunes = r.prism(q, args.split, p_ids, gen, N=args.prism_n, final_k=args.prism_k, temperature=args.temperature)
                rec.update({"svf_scores": scores, "prunes": prunes, "majority_correct": is_correct(args.split, majority(texts, args.split), gt),
                            "k_correct": [is_correct(args.split, t, gt) for t in texts]})
            rec.update({"method_correct": is_correct(args.split, text, gt), "greedy_forwards": f1 - f0,
                        "method_forwards": r.forwards - f1, "method_reads": r.reads - rd1, "text": text})
            out.write(json.dumps(rec) + "\n"); out.flush()
            if (n + 1) % 25 == 0:
                print(f"[baseline {tag}] {args.split} shard {args.shard}: {n + 1}/{len(probs)}", flush=True)


def summary(args):
    out_dir = pathlib.Path(args.out_dir) / "guided"
    rows = [json.loads(l) for f in sorted(glob.glob(str(out_dir / f"{args.tag}_{args.split}_shard*.jsonl"))) for l in open(f)]
    g = [r["greedy_correct"] for r in rows]; m = [r["method_correct"] for r in rows]
    fx = sum(1 for a, b in zip(g, m) if b and not a); br = sum(1 for a, b in zip(g, m) if a and not b)
    gf = sum(r["greedy_forwards"] for r in rows)
    res = {"tag": args.tag, "split": args.split, "n": len(rows), "greedy_acc": round(100 * sum(g) / len(g), 1),
           "method_acc": round(100 * sum(m) / len(m), 1), "fixed": fx, "broken": br, "p": round(mcnemar(fx, br), 3),
           "decoding_compute_vs_greedy": round(sum(r["method_forwards"] for r in rows) / gf, 2),
           "compute_incl_reads_vs_greedy": round(sum(r["method_forwards"] + r["method_reads"] for r in rows) / gf, 2)}
    if "redos" in rows[0]:
        res.update({"share_with_redo": round(sum(r["redos"] > 0 for r in rows) / len(rows), 3), "mean_redos": round(sum(r["redos"] for r in rows) / len(rows), 2)})
    if "majority_correct" in rows[0]:
        mj = [r["majority_correct"] for r in rows]
        mfx = sum(1 for a, b in zip(g, mj) if b and not a); mbr = sum(1 for a, b in zip(g, mj) if a and not b)
        res.update({"majority_of_k_acc": round(100 * sum(mj) / len(rows), 1), "majority_fixed": mfx, "majority_broken": mbr,
                    "majority_p": round(mcnemar(mfx, mbr), 3),
                    "majority_compute_incl_reads_vs_greedy": round(sum(r["method_forwards"] + r["method_reads"] - len(r.get("svf_scores", []))
                                                                       for r in rows) / gf, 2),
                    "pass_at_k": round(100 * sum(any(r["k_correct"]) for r in rows) / len(rows), 1)})
    if "resamples" in rows[0]:
        res["mean_resamples"] = round(sum(r["resamples"] for r in rows) / len(rows), 2)
    (out_dir / f"summary_{args.tag}_{args.split}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def select(args):
    """Best-of-k by a training-free signal on the existing candidate pools."""
    from divr.train_verifier import load_samples
    r = Baselines(args.out_dir, args.name, args.epoch, args.gen_length, args.generator)
    by = load_samples(args.out_dir, args.split, args.sample_tag, args.gen_length)
    qs = {pid: q for pid, q, _ in problems(args.split, 0, data_dir=args.data_dir)}
    sdir = pathlib.Path(args.out_dir) / "scores"
    sdir.mkdir(parents=True, exist_ok=True)
    fn = sdir / f"bl-{args.signal}{args.sample_tag}{file_tag(args.gen_length)}_{args.name}_e{args.epoch}_{args.split}.jsonl"
    done = {json.loads(l)["pid"] for l in open(fn)} if fn.exists() else set()
    with open(fn, "a") as out:
        for n, (pid, S) in enumerate(sorted(by.items())):
            if pid in done or pid not in qs:
                continue
            if args.limit and n >= args.limit:
                break
            p_ids = r.tok(build_prompt(r.tok, args.split, qs[pid]))["input_ids"]
            if args.signal == "svf":
                sc = r.svf(qs[pid], [s["text"] for s in S], args.split).tolist()
            else:
                xs = torch.tensor([p_ids + s["resp"] for s in S], device="cuda")
                sc = r.loglik(xs, len(p_ids)).tolist()
            out.write(json.dumps({"pid": pid, "scores": sc, "correct": [s["correct"] for s in S]}) + "\n")
            if (n + 1) % 100 == 0:
                print(f"[select {args.signal}] {n + 1}/{len(by)}", flush=True)
    rows = [json.loads(l) for l in open(fn)]
    best = sum(r_["correct"][max(range(len(r_["scores"])), key=lambda i: r_["scores"][i])] for r_ in rows)
    pairs = [(a, b) for r_ in rows for i, a in enumerate(r_["scores"]) for j, b in enumerate(r_["scores"]) if r_["correct"][i] and not r_["correct"][j]]
    res = {"signal": args.signal, "split": args.split, "n": len(rows), "best_of_k": round(100 * best / len(rows), 1),
           "within_question_pair_acc": round(sum(a > b for a, b in pairs) / max(1, len(pairs)), 3), "file": str(fn)}
    (sdir / f"summary_bl-{args.signal}{args.sample_tag}{file_tag(args.gen_length)}_{args.name}_e{args.epoch}_{args.split}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "summary", "select"])
    ap.add_argument("--name", default=""); ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--gen_length", type=int, default=GEN)
    ap.add_argument("--n_problems", type=int, default=0)
    ap.add_argument("--generator", default="same", choices=["same", "base"])
    ap.add_argument("--method", default="guided", choices=["guided", "smc", "prism"])
    ap.add_argument("--score", default="conf", choices=["value", "conf", "svf"])
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--k_alt", type=int, default=4)
    ap.add_argument("--max_redos", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--particles", type=int, default=4)
    ap.add_argument("--prism_n", type=int, default=16); ap.add_argument("--prism_k", type=int, default=4)
    ap.add_argument("--signal", default="svf", choices=["svf", "loglik"])
    ap.add_argument("--sample_tag", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--data_dir", default=None)
    args = ap.parse_args()
    {"run": run, "summary": summary, "select": select}[args.mode](args)


if __name__ == "__main__":
    main()
