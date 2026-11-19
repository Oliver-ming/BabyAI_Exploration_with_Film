#!/usr/bin/env bash
# 支线 1 · 语言编码器完整实验流程（含审计修复后的对照与探针）
#
# 用法：  bash branches/b1_langenc/run.sh
#         VARIANTS="dan pos_mlp" SEEDS="0" bash branches/b1_langenc/run.sh
#         SKIP_TRAIN=1 bash branches/b1_langenc/run.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

VARIANTS=${VARIANTS:-"mean mean_pos mean_wide dan pos_mlp gru_small gru transformer film_gru_small film_gru"}
SEEDS=${SEEDS:-"0 1 2"}
THREADS=${THREADS:-1}
WORKERS=${WORKERS:-12}
SKIP_TRAIN=${SKIP_TRAIN:-0}

echo "=== ① 冻结区自检 ==="
git diff --stat -- src/ data/expert_5000.pkl runs/baseline_seed0 runs/baseline_seed1 \
  runs/baseline_seed2 reports/baseline_eval.json reports/baseline_report.md \
  branches/b2_film branches/b3_dagger
echo "（无输出 = 冻结区与兄弟支线干净）"

if [ "$SKIP_TRAIN" != "1" ]; then
  echo
  echo "=== ② 训练（跑满 30 epoch，事后选 val_acc 最优）==="
  for v in $VARIANTS; do
    for s in $SEEDS; do
      python branches/b1_langenc/train_langenc.py --variant "$v" --seed "$s" \
        --weight-decay 0.0 --out "runs/b1_${v}_seed${s}" --threads "$THREADS"
    done
  done
fi

RUNS=""
for v in $VARIANTS; do
  for s in $SEEDS; do RUNS="$RUNS runs/b1_${v}_seed${s}"; done
done
REFS=""
for s in $SEEDS; do REFS="$REFS runs/b3_bc30_seed$s"; done
for s in $SEEDS; do REFS="$REFS runs/b2_film_nwd_seed$s"; done

echo
echo "=== ③ 闭环评估（S6N3 + S7N4，严格分桶对照 + 3-seed 配对）==="
python branches/b1_langenc/eval_langenc.py --runs $RUNS --baseline-runs $REFS \
  --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --workers "$WORKERS" \
  --out reports/b1_langenc_eval.json
python branches/b1_langenc/eval_langenc.py --runs $RUNS --baseline-runs $REFS \
  --env-id BabyAI-GoToLocalS7N4-v0 --seeds test_s7 --workers "$WORKERS" \
  --out reports/b1_langenc_s7n4.json

echo
echo "=== ④ 词序敏感度探针 + 闭环打乱测试 ==="
python branches/b1_langenc/probe_word_order.py --runs $REFS $RUNS \
  --n-states 300 --workers "$WORKERS" \
  --out reports/b1_word_order_sensitivity.json \
  --eval-scrambled --scrambled-out reports/b1_scrambled_eval.json

echo
echo "=== ⑤ 语义接地反事实探针（颜色/类型替换）==="
python branches/b1_langenc/probe_grounding.py --runs $REFS $RUNS \
  --n-states 600 --workers "$WORKERS" --out reports/b1_semantic_sensitivity.json

echo
echo "=== ⑥ 失败构成对比（表征混淆 vs 状态覆盖）==="
python branches/b1_langenc/compare_failures.py --runs $REFS $RUNS \
  --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --workers "$WORKERS" \
  --out reports/b1_failure_compare.json

echo
echo "=== ⑦ 汇总报告 + 图表 ==="
python branches/b1_langenc/make_summary.py
python branches/b1_langenc/make_figures.py

echo
echo "=== 完成 ✅ ==="
echo "汇总：reports/b1_langenc_summary.md"
echo "图：  reports/figures/b1_langenc/"
echo "过程：branches/b1_langenc/LOG.md"
