#!/usr/bin/env bash
# SR-SMC (4 particles, T 1.0) and Prism (16 -> 4, T 0.7) on one benchmark, four shards.
# Usage: scripts/06_baselines.sh <verifier name> <epoch> <split> [extra args, e.g. --gen_length 512]
set -e
V=$1; E=$2; S=$3; shift 3
for M in "smc --particles 4 --temperature 1.0" "prism --prism_n 16 --prism_k 4 --temperature 0.7"; do
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -m divr.baselines run --name "$V" --epoch "$E" --split "$S" --method $M --shard $i --n_shards 4 --out_dir outputs "$@" &
  done
  wait
done
G=$( [ "$S" = math500 ] && echo _g512 )
python -m divr.baselines summary --tag "bl-smc_n4_T1${G}_${V}_e${E}" --split "$S" --out_dir outputs
python -m divr.baselines summary --tag "bl-prism_n16_k4_T0.7${G}_${V}_e${E}" --split "$S" --out_dir outputs
