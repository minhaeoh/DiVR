#!/usr/bin/env bash
# Decodes that train the verifiers and the best-of-4 candidate pools, four shards on four GPUs.
# Usage: scripts/02_sample.sh <model> <adapter|none> <tag> <split> [extra args]
set -e
MODEL=$1; ADAPTER=$2; TAG=$3; SPLIT=$4; shift 4
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python -m divr.sample --model "$MODEL" --adapter "$ADAPTER" --tag "$TAG" --split "$SPLIT" \
      --k 4 --temperature 0.7 --shard $i --n_shards 4 --out_dir outputs "$@" &
done
wait
