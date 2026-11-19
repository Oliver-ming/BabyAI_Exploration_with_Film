#!/usr/bin/env bash
# 支线 3 (v2) · 系统改进完整实验流水线（单进程严格单线程确定性，多进程并行调度）
set -euo pipefail

cd "$(dirname "$0")/../.."
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

FORCE=${FORCE:-0}
need() { [ "$FORCE" = "1" ] || [ ! -e "$1" ]; }

echo "=== [0] 冻结区完整性检查 & 生成验证集只读 on-policy 探针 ==="
python -c "import sys; sys.path.insert(0, 'branches/_shared'); from branch_utils import frozen_diff_guard; print(frozen_diff_guard())"
if need data/b3_v2_val_probe.pkl; then
  python branches/b3_dagger/collect_interactive.py     --run runs/b3_bc30_seed0 --seed-pool val_probe --allow-val-probe     --episodes 500 --exec-mode unstick --query-gate all     --out data/b3_v2_val_probe.pkl
fi

echo "=== [1] 严格 4500 种子等预算切分实验 (Split-DAgger: 3000 专家 + 1500 交互) ==="
pids=()
for s in 0 1 2; do
  if need "runs/b3_v2_bc3000_seed$s/best_model.pt"; then
    python branches/b3_dagger/train_iwr.py --seed $s --tag v2_bc3000 --arch bc       --expert-limit 3000 --out runs/b3_v2_bc3000_seed$s & pids+=($!)
  fi
  if need "runs/b3_v2_film3000_seed$s/best_model.pt"; then
    python branches/b3_dagger/train_iwr.py --seed $s --tag v2_film3000 --arch film       --expert-limit 3000 --out runs/b3_v2_film3000_seed$s & pids+=($!)
  fi
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

pids=()
for s in 0 1 2; do
  if need "data/b3_v2_split_unstick_s$s.pkl"; then
    python branches/b3_dagger/collect_interactive.py       --run runs/b3_v2_bc3000_seed$s --seed-pool train_tail1500 --episodes 1500       --exec-mode unstick --query-gate all --out data/b3_v2_split_unstick_s$s.pkl & pids+=($!)
  fi
  if need "data/b3_v2_split_ens_s$s.pkl"; then
    python branches/b3_dagger/collect_interactive.py       --run runs/b3_v2_bc3000_seed$s       --ensemble-runs runs/b3_v2_bc3000_seed0 runs/b3_v2_bc3000_seed1 runs/b3_v2_bc3000_seed2       --seed-pool train_tail1500 --episodes 1500       --exec-mode unstick --query-gate ensemble --margin-thresh 0.35       --out data/b3_v2_split_ens_s$s.pkl & pids+=($!)
  fi
  if need "data/b3_v2_split_film_s$s.pkl"; then
    python branches/b3_dagger/collect_interactive.py       --run runs/b3_v2_film3000_seed$s --seed-pool train_tail1500 --episodes 1500       --exec-mode unstick --query-gate all --out data/b3_v2_split_film_s$s.pkl & pids+=($!)
  fi
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

pids=()
for s in 0 1 2; do
  if need "runs/b3_v2_split_dag_seed$s/best_model.pt"; then
    python branches/b3_dagger/train_iwr.py --seed $s --tag v2_split_dag --arch bc       --expert-limit 3000 --extra-data data/b3_v2_split_unstick_s$s.pkl       --intv-weight 2.0 --out runs/b3_v2_split_dag_seed$s & pids+=($!)
  fi
  if need "runs/b3_v2_split_ens_seed$s/best_model.pt"; then
    python branches/b3_dagger/train_iwr.py --seed $s --tag v2_split_ens --arch bc       --expert-limit 3000 --extra-data data/b3_v2_split_ens_s$s.pkl       --intv-weight 2.0 --out runs/b3_v2_split_ens_seed$s & pids+=($!)
  fi
  if need "runs/b3_v2_film_split_seed$s/best_model.pt"; then
    python branches/b3_dagger/train_iwr.py --seed $s --tag v2_film_split --arch film       --expert-limit 3000 --extra-data data/b3_v2_split_film_s$s.pkl       --intv-weight 2.0 --out runs/b3_v2_film_split_seed$s & pids+=($!)
  fi
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

echo "=== [2] 增量交互实验采集 (全新训练种子池 fresh_r1 = 5300:6800) ==="
pids=()
if need data/b3_v2_fresh_expert1500.pkl; then
  python branches/b3_dagger/collect_interactive.py     --seed-pool fresh_r1 --episodes 1500 --exec-mode expert     --out data/b3_v2_fresh_expert1500.pkl & pids+=($!)
fi

for s in 0 1 2; do
  python branches/b3_dagger/collect_interactive.py     --run runs/b3_bc30_seed$s --seed-pool fresh_r1 --episodes 1500     --exec-mode greedy --query-gate all --out data/b3_v2_greedy_s$s.pkl & pids+=($!)
  python branches/b3_dagger/collect_interactive.py     --run runs/b3_bc30_seed$s --seed-pool fresh_r1 --episodes 1500     --exec-mode beta --beta 0.25 --query-gate all --out data/b3_v2_beta025_s$s.pkl & pids+=($!)
  python branches/b3_dagger/collect_interactive.py     --run runs/b3_bc30_seed$s --seed-pool fresh_r1 --episodes 1500     --exec-mode unstick --query-gate all --out data/b3_v2_unstick_s$s.pkl & pids+=($!)
  python branches/b3_dagger/collect_interactive.py     --run runs/b3_bc30_seed$s --seed-pool fresh_r1 --episodes 1500     --exec-mode unstick --query-gate disagree --out data/b3_v2_qdis_s$s.pkl & pids+=($!)
  python branches/b3_dagger/collect_interactive.py     --run runs/b3_bc30_seed$s --seed-pool fresh_r1 --episodes 1500     --exec-mode unstick --query-gate margin --margin-thresh 0.45     --out data/b3_v2_qmar_s$s.pkl & pids+=($!)
  python branches/b3_dagger/collect_interactive.py     --run runs/b3_bc30_seed$s     --ensemble-runs runs/b3_bc30_seed0 runs/b3_bc30_seed1 runs/b3_bc30_seed2     --seed-pool fresh_r1 --episodes 1500     --exec-mode unstick --query-gate ensemble --margin-thresh 0.30     --out data/b3_v2_qens_s$s.pkl & pids+=($!)
  python branches/b3_dagger/collect_interactive.py     --run runs/b2_film_nwd_seed$s --seed-pool fresh_r1 --episodes 1500     --exec-mode unstick --query-gate all --out data/b3_v2_film_r1_s$s.pkl & pids+=($!)
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

echo "=== [3] 增量交互组并行训练 (全组 3 seeds, 单进程 1 线程) ==="
pids=()
for s in 0 1 2; do
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_bcfresh --arch bc     --extra-data data/b3_v2_fresh_expert1500.pkl --intv-weight 1.0 --out runs/b3_v2_bcfresh_seed$s & pids+=($!)
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_greedy --arch bc     --extra-data data/b3_v2_greedy_s$s.pkl --intv-weight 1.0 --out runs/b3_v2_greedy_seed$s & pids+=($!)
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_beta025 --arch bc     --extra-data data/b3_v2_beta025_s$s.pkl --intv-weight 1.0 --out runs/b3_v2_beta025_seed$s & pids+=($!)
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_unstick --arch bc     --extra-data data/b3_v2_unstick_s$s.pkl --intv-weight 1.0 --out runs/b3_v2_unstick_seed$s & pids+=($!)
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_iwr2 --arch bc     --extra-data data/b3_v2_unstick_s$s.pkl --intv-weight 2.0 --out runs/b3_v2_iwr2_seed$s & pids+=($!)
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_iwr3 --arch bc     --extra-data data/b3_v2_unstick_s$s.pkl --intv-weight 3.0 --out runs/b3_v2_iwr3_seed$s & pids+=($!)
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_qdis --arch bc     --extra-data data/b3_v2_qdis_s$s.pkl --intv-weight 2.0 --out runs/b3_v2_qdis_seed$s & pids+=($!)
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_qmar --arch bc     --extra-data data/b3_v2_qmar_s$s.pkl --intv-weight 2.0 --out runs/b3_v2_qmar_seed$s & pids+=($!)
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_qens --arch bc     --extra-data data/b3_v2_qens_s$s.pkl --intv-weight 2.0 --out runs/b3_v2_qens_seed$s & pids+=($!)
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_film_dag --arch film     --extra-data data/b3_v2_film_r1_s$s.pkl --intv-weight 2.0 --out runs/b3_v2_film_dag_seed$s & pids+=($!)
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

echo "=== [4] 第二轮非重叠种子采集与训练 (fresh_r2 = 6800:8300) ==="
pids=()
for s in 0 1 2; do
  python branches/b3_dagger/collect_interactive.py     --run runs/b3_v2_iwr2_seed$s --seed-pool fresh_r2 --episodes 1500     --exec-mode unstick --query-gate all --out data/b3_v2_r2_unstick_s$s.pkl & pids+=($!)
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

pids=()
for s in 0 1 2; do
  python branches/b3_dagger/train_iwr.py --seed $s --tag v2_r2 --arch bc     --extra-data data/b3_v2_unstick_s$s.pkl,data/b3_v2_r2_unstick_s$s.pkl     --intv-weight 2.0 --out runs/b3_v2_r2_seed$s & pids+=($!)
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

echo "=== [5] 统一闭环评估 (S6N3 含失败分解 + S7N4 零样本迁移 + 冻结脚本交叉校验) ==="
EVAL_GROUPS="baseline b3_bc30 b3_bcextra b3_dagger b3_novel b3_r2 b3_v2_bc3000 b3_v2_split_dag b3_v2_split_ens b3_v2_bcfresh b3_v2_greedy b3_v2_beta025 b3_v2_unstick b3_v2_iwr2 b3_v2_iwr3 b3_v2_qdis b3_v2_qmar b3_v2_qens b3_v2_r2 b3_v2_r2_w1 b2_film_nwd b3_v2_film3000 b3_v2_film_split b3_v2_film_dag b3_v2_film_dag_w1"
ALL_RUNS=""
for g in $EVAL_GROUPS; do
  for s in 0 1 2; do
    ALL_RUNS="$ALL_RUNS runs/${g}_seed${s}"
  done
done

python branches/b3_dagger/eval_b3.py --runs $ALL_RUNS --ref-group b3_bc30   --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --with-failures --workers 12   --out reports/b3_v2_eval_s6n3.json &
pid_s6=$!

python branches/b3_dagger/eval_b3.py --runs $ALL_RUNS --ref-group b3_bc30   --env-id BabyAI-GoToLocalS7N4-v0 --seeds test_s7 --workers 12   --out reports/b3_v2_eval_s7n4.json &
pid_s7=$!

wait "$pid_s6" "$pid_s7"

python src/eval_closed_loop.py   --runs runs/b3_v2_split_dag_seed0 runs/b3_v2_iwr2_seed0 runs/b3_v2_r2_seed0   --env-id BabyAI-GoToLocalS6N3-v0 --seeds test   --out reports/b3_v2_frozen_eval_check.json

python -c "import sys; sys.path.insert(0, 'branches/_shared'); from branch_utils import frozen_diff_guard; print(frozen_diff_guard())"
echo "=== ALL DONE ==="
