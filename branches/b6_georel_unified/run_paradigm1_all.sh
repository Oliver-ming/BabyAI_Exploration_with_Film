#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

PY="env_embodied/bin/python"
SCRIPT="branches/b6_georel_unified/train_georel.py"
MAX_JOBS="${MAX_JOBS:-8}"

run_job() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 1
  done
  "$@" &
}

echo "=== [1/2] Training Paradigm 1 Full-Data (100%) Composable Ladder (18 runs) ==="
for var in s2_map1h s3_map1h_relbias s3_res1h_relbias bc30_gru s4_georel_gru s4_res1h_gru; do
  for seed in 0 1 2; do
    run_job "$PY" "$SCRIPT" --variant "$var" --seed "$seed"
  done
done

echo "=== [2/2] Training Paradigm 1 Step-Matched 20% Low-Data Scaling (12 runs) ==="
for var in s2_map1h s3_map1h_relbias s3_res1h_relbias s4_georel_gru; do
  for seed in 0 1 2; do
    run_job "$PY" "$SCRIPT" --variant "$var" --frac 0.2 --match-steps --seed "$seed"
  done
done

wait
echo "=== All 30 Paradigm 1 training runs completed ==="
