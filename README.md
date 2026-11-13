# DiVR: Learning When Diffusion Language Models Should Regenerate

Code for the paper [*Learning When Diffusion Language Models Should Regenerate*](https://arxiv.org/abs/2611.17320) (arXiv 2611.17320, 2026).

DiVR (Diffusion Verifier for Regeneration) is an outcome-supervised verifier for masked diffusion language
models. It scores the partially masked canvas at every block boundary and decides whether the block just
decoded should be regenerated, and which of the regenerated alternatives to keep. The verifier shares the
generator's frozen backbone and adds only a LoRA adapter and a two-layer head (about 1% of the backbone),
trained on the generator's own decodes labelled by their final-answer correctness.

This repository contains the verifier training and inference code, the GSM8K SFT recipe of the LLaDA-8B-Base
generator, and the SR-SMC / Prism baselines, all in one evaluation harness.

## Installation

```bash
pip install -r requirements.txt
```

`transformers==4.51.3` is pinned: the LLaDA remote code does not load under 5.x. Everything is run from the
repository root as `python -m <module>`. One 48 GB GPU is enough for every step; the sampling and decoding
scripts shard a benchmark over four GPUs.

## Models

| Generator | Adapter | Note |
| --- | --- | --- |
| `GSAI-ML/LLaDA-8B-Base` | `scripts/01_sft_llada.sh` | the paper's main generator (GSM8K SFT, LoRA r 128); the untuned model is also evaluated |
| `GSAI-ML/LLaDA-8B-Instruct` | none | released checkpoint |
| `Dream-org/Dream-v0-Instruct-7B` | none | released checkpoint |

Each generator gets two verifiers: one trained on its GSM8K decodes (used for GSM8K, MATH-500 and Countdown)
and one trained on its MBPP decodes (used for HumanEval and MBPP).

## Pipeline

The scripts in `scripts/` run the paper's settings end to end. With `MODEL`, `ADAPTER` (or `none`), a `TAG`
that names the generator in file names (e.g. `_sft`, `_inst`, `_dream`) and a verifier `NAME`:

```bash
scripts/01_sft_llada.sh                                   # GSM8K SFT of LLaDA-8B-Base (LLaDA-Base only)
scripts/03_train_verifier.sh $MODEL $ADAPTER $TAG $NAME   # samples + math and code verifiers
scripts/04_guided.sh $NAME                                # DiVR-guided decoding on the five benchmarks
scripts/05_best_of_4.sh $MODEL $ADAPTER $TAG $NAME        # best-of-4 selection by the verifier
scripts/06_baselines.sh ${NAME}_math 1 test               # SR-SMC and Prism on one benchmark
```

The individual steps:

1. **Sample** the generator's own decodes (`divr.sample`): four responses per question at temperature 0.7,
   d1's block decoder (256-token canvas, 32-token blocks, 2 tokens per step; MATH-500 uses 512 tokens), each
   graded by its final answer. 2,000 GSM8K training questions for the math verifier, the 464 MBPP
   training + validation problems for the code verifier.
2. **Train the verifier** (`divr.train_verifier`): every block boundary of every decode is a training state,
   labelled with the decode's outcome. LoRA r 32 / alpha 64 on the attention and feed-forward projections plus a
   4096-1024-1 SiLU head on the mean final-layer state over the response positions (masks included); class-balanced
   BCE plus a within-question pairwise ranking loss; two epochs, four questions per optimizer step. The epoch is
   chosen by held-out within-question pair accuracy (`divr.select_epoch`).
3. **Guided decoding** (`divr.guided_decode`): after each decoded block the verifier scores the canvas; below
   `theta = 0.5` the block is remasked and `K = 4` alternatives are sampled at temperature 0.7; the original and
   the alternatives are scored and the best is kept; at most three regeneration events per problem. Every run
   also records the paired greedy decode, which is the reference for accuracy changes and for cost (forward
   passes weighted by batch rows, verifier reads included in the summary).
4. **Best-of-4** (`divr.score`): the verifier's score of the finished response reranks four complete samples;
   the summary also reports majority and score-weighted voting, pass@4 and pair accuracy per block.
5. **Baselines** (`divr.baselines`): SR-SMC (Luo et al., 2026; four particles at temperature 1.0) and Prism
   (Bai et al., 2026; 16 trajectories pruned to 4 by prompted self-verification, answer by majority vote), both
   reimplemented on the same decoder, prompts and grader; the guided loop with the generator's confidence or
   prompted self-verification as the score; best-of-4 by sequence likelihood or self-verification.

Outputs go under `outputs/` (`samples/`, `verifiers/<name>/`, `guided/`, `scores/`); every run is resumable
and sharded with `--shard i --n_shards n`. Benchmarks are loaded from the Hugging Face Hub; `--data_dir` reads
exported `<split>.jsonl` files of `[pid, question, gt]` rows instead.

## RL post-training comparison

The diffu-GRPO generators of the paper were trained with the public [d1](https://github.com/dllm-reasoning/d1)
code and its GSM8K recipe (3,800 steps on four GPUs). The LLaDA-8B-Instruct run uses d1 unchanged. The untuned
LLaDA-8B-Base run makes two changes: the prompt ends with the assistant prefix `<reasoning>` (passed to d1's
trainer as an opening assistant turn, so that the rendered prompt equals the one used here), and the correctness
reward is this paper's numeric grader instead of d1's exact string match; that reward is
`scripts/d1_lenient_reward.py`, a self-contained drop-in for d1's `correctness_reward_func`. The resulting
adapters are then treated like any other generator: `divr.sample` with `--adapter <checkpoint>`, a verifier
trained on their decodes, and the steps above.

## Paper settings

| | |
| --- | --- |
| Canvas / block / steps | 256 / 32 / 128 (MATH-500: 512 / 32 / 256) |
| Sampling | k 4, temperature 0.7 |
| Verifier | LoRA r 32 alpha 64, head 4096-1024-1 SiLU, lr 1e-4 (LoRA) and 1e-3 (head), 2 epochs, accumulation 4, pair loss weight 1 |
| Guided decoding | theta 0.5 (chosen on the GSM8K dev split), K 4, temperature 0.7, at most 3 regenerations, one block per event |
| GSM8K SFT (LLaDA-Base) | LoRA r 128 alpha 256 dropout 0.05, lr 1e-5, batch 1 x 8, 2 epochs, EOS-filled targets, split seed 42 |
| Benchmarks | GSM8K test (1,319), MATH-500 (500), Countdown three-number tasks (500), HumanEval (164), MBPP test (500) |

## Acknowledgements

The block decoder, the prompts for GSM8K / MATH-500 / Countdown and the MATH answer-equivalence check
(`divr/math_equiv.py`) follow [d1](https://github.com/dllm-reasoning/d1) (Zhao et al., 2025, Apache-2.0;
see `third_party/d1-LICENSE`). The SFT trainer adapts d1's masked-diffusion SFT loss. The Prism baseline is a
port of the released `hts_sampler.py` of Bai et al. (2026).

## Citation

```bibtex
@article{oh2026divr,
  title={Learning When Diffusion Language Models Should Regenerate},
  author={Oh, Minhae and Kim, Jeonghye and Koo, Jinwoo and Kim, Yeongoon and Seo, Jimin and Seol, Minsik and Lee, Jungwoo},
  journal={arXiv preprint arXiv:2611.17320},
  year={2026},
  url={https://arxiv.org/abs/2611.17320}
}
```

## License

MIT (see `LICENSE`).
