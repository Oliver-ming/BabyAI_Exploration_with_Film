#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
PY="${PYTHON:-env_embodied/bin/python}"
MAX_JOBS="${MAX_JOBS:-4}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then echo "MAX_JOBS must be positive" >&2; exit 2; fi
mkdir -p logs/zez-attention-method
pids=()
wait_batch() {
  local status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  pids=()
  return "$status"
}
for frac in 1.0 0.2; do
  extra=()
  if [[ "$frac" == 0.2 ]]; then extra=(--match-steps); fi
  for variant in s2_map1h s3_map1h_relbias; do
    for seed in 0 1 2; do
      "$PY" branches/b6_georel_unified/train_georel.py --variant "$variant" --frac "$frac" --seed "$seed" "${extra[@]}" >"logs/zez-attention-method/${variant}_${frac}_seed${seed}.log" 2>&1 &
      pids+=("$!")
      if (( ${#pids[@]} >= MAX_JOBS )); then wait_batch; fi
    done
  done
done
wait_batch
"$PY" branches/b6_georel_unified/eval_georel.py --workers "$MAX_JOBS"
echo "Completed AB/ABC: 12 runs and evaluation."
