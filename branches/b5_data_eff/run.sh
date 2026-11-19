#!/usr/bin/env bash
# 支线 5 · 数据效率（Data Efficiency）完整实验流程
#
# 用法：  bash branches/b5_data_eff/run.sh                 # 全流程（训练 96 模型 + S6/S7评估 + 探针 + 作图）
#         SKIP_TRAIN=1 bash branches/b5_data_eff/run.sh    # 仅重跑评估、探针、汇总与图表（约 45 秒）
set -euo pipefail

cd "$(dirname "$0")/../.."
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

THREADS=${THREADS:-1}
WORKERS=${WORKERS:-16}
SKIP_TRAIN=${SKIP_TRAIN:-0}

echo "=== ① 冻结区与兄弟支线自检 ==="
git diff --stat -- src/ data/expert_5000.pkl runs/baseline_seed0 runs/baseline_seed1 \
  runs/baseline_seed2 reports/baseline_eval.json reports/baseline_report.md \
  reports/b5_data_efficiency_eval.json reports/data_efficiency.csv \
  branches/b1_langenc branches/b2_film branches/b4_curriculum
echo "（无输出 = 冻结区与前置分支干净）"

if [ "$SKIP_TRAIN" != "1" ]; then
  echo
  echo "=== ② 训练（96 个 B5 运行，30 epoch，按 S6N3 val_acc 事后选优）==="
  for m in bc_naive bc_step film_step; do
    for f in 0.05 0.10 0.20 0.50; do
      ft=$(python3 -c "print(f'f{int(round($f*100)):03d}')")
      for s in 0 1 2; do
        python branches/b5_data_eff/train_data_eff.py --method "$m" --frac "$f" \
          --seed "$s" --epochs 30 --threads "$THREADS" --out "runs/b5_${m}_${ft}_seed${s}"
      done
    done
  done
  for m in bc_reg bc_d4 film_d4; do
    for f in 0.05 0.10 0.20 0.50 1.00; do
      ft=$(python3 -c "print(f'f{int(round($f*100)):03d}')")
      for s in 0 1 2; do
        python branches/b5_data_eff/train_data_eff.py --method "$m" --frac "$f" \
          --seed "$s" --epochs 30 --threads "$THREADS" --out "runs/b5_${m}_${ft}_seed${s}"
      done
    done
  done
  for m in bc_z2 bc_c4 bc_wd bc_bal gru_small_step; do
    for s in 0 1 2; do
      python branches/b5_data_eff/train_data_eff.py --method "$m" --frac 0.20 \
        --seed "$s" --epochs 30 --threads "$THREADS" --out "runs/b5_${m}_f020_seed${s}"
    done
  done
fi

echo
echo "=== ③ S6N3 与 S7N4 闭环扩展曲线 + 20% 13路消融评估 ==="
python branches/b5_data_eff/eval_data_eff.py --env-id BabyAI-GoToLocalS6N3-v0 --seeds test \
  --workers "$WORKERS" --out reports/b5_data_eff_eval.json
python branches/b5_data_eff/eval_data_eff.py --env-id BabyAI-GoToLocalS7N4-v0 --seeds test_s7 \
  --workers "$WORKERS" --out reports/b5_data_eff_s7n4.json

echo
echo "=== ④ 语义接地演化探针与失败构成拆解 ==="
python branches/b5_data_eff/probe_data_eff_grounding.py --n-states 600 --workers "$WORKERS" \
  --out reports/b5_data_eff_sensitivity.json
python branches/b5_data_eff/compare_failures.py --env-id BabyAI-GoToLocalS6N3-v0 --seeds test \
  --workers "$WORKERS" --out reports/b5_data_eff_failures.json

echo
echo "=== ⑤ 汇总报告 + 图表 ==="
python branches/b5_data_eff/make_summary.py
python branches/b5_data_eff/make_figures.py

echo
echo "=== 完成 ✅ ==="
echo "汇总：reports/b5_data_eff_summary.md"
echo "图：  reports/figures/b5_data_eff/"
echo "日志：branches/b5_data_eff/LOG.md"
