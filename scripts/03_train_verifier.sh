#!/usr/bin/env bash
# The math verifier (2,000 GSM8K training questions x 4 samples) and the code verifier (464 MBPP problems x 4) of one generator.
# Usage: scripts/03_train_verifier.sh <model> <adapter|none> <tag> <name>
set -e
MODEL=$1; ADAPTER=$2; TAG=$3; NAME=$4
scripts/02_sample.sh "$MODEL" "$ADAPTER" "$TAG" train --n_problems 2000
scripts/02_sample.sh "$MODEL" "$ADAPTER" "$TAG" mbpp_train --batch_problems 1
CUDA_VISIBLE_DEVICES=0 python -m divr.train_verifier --mode lora --name "${NAME}_math" --model "$MODEL" --adapter "$ADAPTER" --sample_tag "$TAG" \
    --split train --n_problems 2000 --holdout 100 --epochs 2 --blocks_per_step 1 --accum 4 --out_dir outputs
CUDA_VISIBLE_DEVICES=1 python -m divr.train_verifier --mode lora --name "${NAME}_code" --model "$MODEL" --adapter "$ADAPTER" --sample_tag "$TAG" \
    --split mbpp_train --n_problems 0 --holdout 64 --epochs 2 --blocks_per_step 1 --accum 4 --out_dir outputs
python -m divr.select_epoch --name "${NAME}_math" --out_dir outputs
python -m divr.select_epoch --name "${NAME}_code" --out_dir outputs
