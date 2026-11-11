#!/usr/bin/env bash
# Best-of-4 selection: four samples per problem at temperature 0.7, reranked by the verifier's score of the finished response.
# Usage: scripts/05_best_of_4.sh <model> <adapter|none> <tag> <name>
set -e
MODEL=$1; ADAPTER=$2; TAG=$3; NAME=$4
EM=$(python -m divr.select_epoch --name "${NAME}_math" --out_dir outputs 2>/dev/null)
EC=$(python -m divr.select_epoch --name "${NAME}_code" --out_dir outputs 2>/dev/null)
scripts/02_sample.sh "$MODEL" "$ADAPTER" "$TAG" test
scripts/02_sample.sh "$MODEL" "$ADAPTER" "$TAG" math500 --gen_length 512
scripts/02_sample.sh "$MODEL" "$ADAPTER" "$TAG" countdown --n_problems 500
scripts/02_sample.sh "$MODEL" "$ADAPTER" "$TAG" humaneval --batch_problems 1
scripts/02_sample.sh "$MODEL" "$ADAPTER" "$TAG" mbpp --batch_problems 1
for S in test countdown; do
  CUDA_VISIBLE_DEVICES=0 python -m divr.score score --name "${NAME}_math" --epoch "$EM" --split $S --sample_tag "$TAG" --out_dir outputs
  python -m divr.score summary --name "${NAME}_math" --split $S --sample_tag "$TAG" --out_dir outputs
done
CUDA_VISIBLE_DEVICES=0 python -m divr.score score --name "${NAME}_math" --epoch "$EM" --split math500 --gen_length 512 --sample_tag "$TAG" --chunk 4 --out_dir outputs
python -m divr.score summary --name "${NAME}_math" --split math500 --gen_length 512 --sample_tag "$TAG" --out_dir outputs
for S in humaneval mbpp; do
  CUDA_VISIBLE_DEVICES=1 python -m divr.score score --name "${NAME}_code" --epoch "$EC" --split $S --sample_tag "$TAG" --out_dir outputs
  python -m divr.score summary --name "${NAME}_code" --split $S --sample_tag "$TAG" --out_dir outputs
done
