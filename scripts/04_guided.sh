#!/usr/bin/env bash
# DiVR-guided decoding on the five benchmarks (theta 0.5, K 4, at most three regenerations), four shards each.
# Usage: scripts/04_guided.sh <name>   (verifiers <name>_math and <name>_code from scripts/03)
set -e
NAME=$1
EM=$(python -m divr.select_epoch --name "${NAME}_math" --out_dir outputs 2>/dev/null)
EC=$(python -m divr.select_epoch --name "${NAME}_code" --out_dir outputs 2>/dev/null)
run() {  # <verifier> <epoch> <split> [extra]
  local V=$1 E=$2 S=$3; shift 3
  for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -m divr.guided_decode run --name "$V" --epoch "$E" --split "$S" --theta 0.5 --k_alt 4 \
        --max_redos 3 --shard $i --n_shards 4 --out_dir outputs "$@" &
  done
  wait
  python -m divr.guided_decode summary --name "$V" --tag "${V}_e${E}_th0.5_k4_rb1$( [ "$S" = math500 ] && echo _g512 )" --split "$S" --out_dir outputs
}
run "${NAME}_math" "$EM" test
run "${NAME}_math" "$EM" math500 --gen_length 512
run "${NAME}_math" "$EM" countdown --n_problems 500
run "${NAME}_code" "$EC" humaneval
run "${NAME}_code" "$EC" mbpp
