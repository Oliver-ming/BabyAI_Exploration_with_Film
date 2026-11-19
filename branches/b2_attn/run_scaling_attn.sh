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

pids=()
for s in 0 1 2; do
  $PY branches/b2_attn/train_attn.py --variant spatial_softmax_attn --frac 0.1 --match-steps --seed $s > logs/b2_attn_ss_f010_s${s}.log 2>&1 &
  pids+=($!)
  $PY branches/b2_attn/train_attn.py --variant spatial_softmax_attn --frac 0.5 --match-steps --seed $s > logs/b2_attn_ss_f050_s${s}.log 2>&1 &
  pids+=($!)
  $PY branches/b2_attn/train_attn_v2.py --variant mh_spatial_relational --frac 0.1 --match-steps --seed $s > logs/b2_v2_rel_f010_s${s}.log 2>&1 &
  pids+=($!)
  $PY branches/b2_attn/train_attn_v2.py --variant mh_spatial_relational --frac 0.5 --match-steps --seed $s > logs/b2_v2_rel_f050_s${s}.log 2>&1 &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
echo "ALL 12 SCALING RUNS COMPLETED"
