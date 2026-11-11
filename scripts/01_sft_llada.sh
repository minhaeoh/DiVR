#!/usr/bin/env bash
# GSM8K SFT of LLaDA-8B-Base (the SFT generator of Table 1). One GPU, about 6 hours on an A6000.
set -e
python -m sft.llada_sft --output_dir checkpoints/llada-base-gsm8k-sft
