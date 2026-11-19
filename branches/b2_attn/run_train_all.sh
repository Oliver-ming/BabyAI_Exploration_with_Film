#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

PY="env_embodied/bin/python"

run_batch() {
  local pids=()
  for cmd in "$@"; do
    eval "$cmd" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

echo "=== Batch 1 (8 runs) ==="
run_batch \
  "$PY branches/b2_attn/train_attn.py --variant spatial_attn_gate --seed 0 > logs/b2_attn_sg_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant spatial_attn_gate --seed 1 > logs/b2_attn_sg_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant spatial_attn_gate --seed 2 > logs/b2_attn_sg_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant spatial_softmax_attn --seed 0 > logs/b2_attn_ss_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant spatial_softmax_attn --seed 1 > logs/b2_attn_ss_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant spatial_softmax_attn --seed 2 > logs/b2_attn_ss_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant cross_attn_mha --seed 0 > logs/b2_attn_mha_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant cross_attn_mha --seed 1 > logs/b2_attn_mha_s1.log 2>&1"

echo "=== Batch 2 (8 runs) ==="
run_batch \
  "$PY branches/b2_attn/train_attn.py --variant cross_attn_mha --seed 2 > logs/b2_attn_mha_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant film_spatial_attn --seed 0 > logs/b2_attn_fs_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant film_spatial_attn --seed 1 > logs/b2_attn_fs_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant film_spatial_attn --seed 2 > logs/b2_attn_fs_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant film_cross_mha --seed 0 > logs/b2_attn_fm_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant film_cross_mha --seed 1 > logs/b2_attn_fm_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant film_cross_mha --seed 2 > logs/b2_attn_fm_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant spatial_attn_gate --frac 0.2 --match-steps --seed 0 > logs/b2_attn_sg_f020_s0.log 2>&1"

echo "=== Batch 3 (8 runs) ==="
run_batch \
  "$PY branches/b2_attn/train_attn.py --variant spatial_attn_gate --frac 0.2 --match-steps --seed 1 > logs/b2_attn_sg_f020_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant spatial_attn_gate --frac 0.2 --match-steps --seed 2 > logs/b2_attn_sg_f020_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant cross_attn_mha --frac 0.2 --match-steps --seed 0 > logs/b2_attn_mha_f020_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant cross_attn_mha --frac 0.2 --match-steps --seed 1 > logs/b2_attn_mha_f020_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant cross_attn_mha --frac 0.2 --match-steps --seed 2 > logs/b2_attn_mha_f020_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant film_spatial_attn --frac 0.2 --match-steps --seed 0 > logs/b2_attn_fs_f020_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant film_spatial_attn --frac 0.2 --match-steps --seed 1 > logs/b2_attn_fs_f020_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant film_spatial_attn --frac 0.2 --match-steps --seed 2 > logs/b2_attn_fs_f020_s2.log 2>&1"

echo "ALL 24 RUNS COMPLETED SUCCESSFULLY"
