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

echo "=== Batch 1 (12 full-data runs) ==="
run_batch \
  "$PY branches/b2_attn/train_attn_v2.py --variant gated_attn_chaplot --seed 0 > logs/b2_v2_chaplot_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant gated_attn_chaplot --seed 1 > logs/b2_v2_chaplot_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant gated_attn_chaplot --seed 2 > logs/b2_v2_chaplot_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant cross_mha_nopos --seed 0 > logs/b2_v2_nopos_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant cross_mha_nopos --seed 1 > logs/b2_v2_nopos_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant cross_mha_nopos --seed 2 > logs/b2_v2_nopos_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant res_spatial_softmax --seed 0 > logs/b2_v2_resss_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant res_spatial_softmax --seed 1 > logs/b2_v2_resss_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant res_spatial_softmax --seed 2 > logs/b2_v2_resss_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_res_spatial_softmax --seed 0 > logs/b2_v2_mhresss_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_res_spatial_softmax --seed 1 > logs/b2_v2_mhresss_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_res_spatial_softmax --seed 2 > logs/b2_v2_mhresss_s2.log 2>&1"

echo "=== Batch 2 (9 full-data runs + 3 v1 f020 runs = 12 runs) ==="
run_batch \
  "$PY branches/b2_attn/train_attn_v2.py --variant attn_map_inject --seed 0 > logs/b2_v2_mapinj_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant attn_map_inject --seed 1 > logs/b2_v2_mapinj_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant attn_map_inject --seed 2 > logs/b2_v2_mapinj_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant token_coattn_spatial --seed 0 > logs/b2_v2_coattn_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant token_coattn_spatial --seed 1 > logs/b2_v2_coattn_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant token_coattn_spatial --seed 2 > logs/b2_v2_coattn_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_spatial_relational --seed 0 > logs/b2_v2_rel_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_spatial_relational --seed 1 > logs/b2_v2_rel_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_spatial_relational --seed 2 > logs/b2_v2_rel_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant spatial_softmax_attn --frac 0.2 --match-steps --seed 0 > logs/b2_attn_ss_f020_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant spatial_softmax_attn --frac 0.2 --match-steps --seed 1 > logs/b2_attn_ss_f020_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn.py --variant spatial_softmax_attn --frac 0.2 --match-steps --seed 2 > logs/b2_attn_ss_f020_s2.log 2>&1"

echo "=== Batch 3 (12 v2 f020 runs) ==="
run_batch \
  "$PY branches/b2_attn/train_attn_v2.py --variant res_spatial_softmax --frac 0.2 --match-steps --seed 0 > logs/b2_v2_resss_f020_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant res_spatial_softmax --frac 0.2 --match-steps --seed 1 > logs/b2_v2_resss_f020_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant res_spatial_softmax --frac 0.2 --match-steps --seed 2 > logs/b2_v2_resss_f020_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_res_spatial_softmax --frac 0.2 --match-steps --seed 0 > logs/b2_v2_mhresss_f020_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_res_spatial_softmax --frac 0.2 --match-steps --seed 1 > logs/b2_v2_mhresss_f020_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_res_spatial_softmax --frac 0.2 --match-steps --seed 2 > logs/b2_v2_mhresss_f020_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant attn_map_inject --frac 0.2 --match-steps --seed 0 > logs/b2_v2_mapinj_f020_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant attn_map_inject --frac 0.2 --match-steps --seed 1 > logs/b2_v2_mapinj_f020_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant attn_map_inject --frac 0.2 --match-steps --seed 2 > logs/b2_v2_mapinj_f020_s2.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant token_coattn_spatial --frac 0.2 --match-steps --seed 0 > logs/b2_v2_coattn_f020_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant token_coattn_spatial --frac 0.2 --match-steps --seed 1 > logs/b2_v2_coattn_f020_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant token_coattn_spatial --frac 0.2 --match-steps --seed 2 > logs/b2_v2_coattn_f020_s2.log 2>&1"

echo "=== Batch 4 (3 remaining v2 f020 runs) ==="
run_batch \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_spatial_relational --frac 0.2 --match-steps --seed 0 > logs/b2_v2_rel_f020_s0.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_spatial_relational --frac 0.2 --match-steps --seed 1 > logs/b2_v2_rel_f020_s1.log 2>&1" \
  "$PY branches/b2_attn/train_attn_v2.py --variant mh_spatial_relational --frac 0.2 --match-steps --seed 2 > logs/b2_v2_rel_f020_s2.log 2>&1"

echo "ALL 39 TRAINING RUNS COMPLETED SUCCESSFULLY"
