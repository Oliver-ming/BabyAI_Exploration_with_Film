#!/usr/bin/env bash
# 支线 4 · 课程学习（Curriculum Learning）完整实验流程
#
# 用法：  bash branches/b4_curriculum/run.sh                  # 全流程（生成数据+训练39模型+评估+探针+作图）
#         SKIP_DATA=1 SKIP_TRAIN=1 bash branches/b4_curriculum/run.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

VARIANTS=${VARIANTS:-"pace_easy pace_hard pace_rand env_easy_stage env_easy_cumul env_easy_mix env_s6n4_prog env_s6n4_mix env_multiscale_prog env_multiscale_mix film_easy_stage film_multiscale_prog film_multiscale_mix"}
SEEDS=${SEEDS:-"0 1 2"}
THREADS=${THREADS:-1}
WORKERS=${WORKERS:-15}
SKIP_DATA=${SKIP_DATA:-0}
SKIP_TRAIN=${SKIP_TRAIN:-0}

echo "=== ① 冻结区自检 ==="
git diff --stat -- src/ data/expert_5000.pkl runs/baseline_seed0 runs/baseline_seed1 \
  runs/baseline_seed2 reports/baseline_eval.json reports/baseline_report.md \
  branches/b2_film
echo "（无输出 = 冻结区与 b2_film 干净）"

if [ "$SKIP_DATA" != "1" ]; then
  echo
  echo "=== ② 生成多难度辅助专家轨迹池（严格隔离 TEST_SEEDS）==="
  python branches/b4_curriculum/gen_curriculum_data.py --workers 4
fi

if [ "$SKIP_TRAIN" != "1" ]; then
  echo
  echo "=== ③ 训练（39 个运行，30 epoch 等步数采样，仅看 S6N3 val_acc 选优）==="
  for v in $VARIANTS; do
    for s in $SEEDS; do
      python branches/b4_curriculum/train_curriculum.py --variant "$v" --seed "$s" \
        --epochs 30 --weight-decay 0.0 --film-l2 0.0 \
        --threads "$THREADS" --out "runs/b4_${v}_seed${s}"
    done
  done
fi

RUNS=""
for v in $VARIANTS; do
  for s in $SEEDS; do RUNS="$RUNS runs/b4_${v}_seed${s}"; done
done
REFS=""
for s in $SEEDS; do REFS="$REFS runs/b3_bc30_seed${s}"; done
for s in $SEEDS; do REFS="$REFS runs/b2_film_nwd_seed${s}"; done

echo
echo "=== ④ 闭环评估（S6N3 + S7N4 + 5级难度谱 S5N2→S8N6）==="
python branches/b4_curriculum/eval_curriculum.py --runs $RUNS --baseline-runs $REFS \
  --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --workers "$WORKERS" \
  --out reports/b4_curriculum_eval.json
python branches/b4_curriculum/eval_curriculum.py --runs $RUNS --baseline-runs $REFS \
  --env-id BabyAI-GoToLocalS7N4-v0 --seeds test_s7 --workers "$WORKERS" \
  --out reports/b4_curriculum_s7n4.json
python branches/b4_curriculum/eval_curriculum.py --runs $RUNS --baseline-runs $REFS \
  --spectrum --workers "$WORKERS" --spectrum-out reports/b4_curriculum_spectrum.json

echo
echo "=== ⑤ 语义接地探针 + S6N3/S7N4 失败构成拆解 ==="
python branches/b4_curriculum/probe_curriculum_grounding.py --runs $REFS $RUNS \
  --n-states 600 --workers "$WORKERS" --out reports/b4_curriculum_sensitivity.json
python branches/b4_curriculum/compare_failures.py --runs $REFS $RUNS \
  --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --workers "$WORKERS" \
  --out reports/b4_curriculum_failures.json
python branches/b4_curriculum/compare_failures.py --runs $REFS $RUNS \
  --env-id BabyAI-GoToLocalS7N4-v0 --seeds test_s7 --workers "$WORKERS" \
  --out reports/b4_curriculum_failures_s7n4.json

echo
echo "=== ⑥ 汇总报告 + 图表 ==="
python branches/b4_curriculum/make_summary.py
python branches/b4_curriculum/make_figures.py

echo
echo "=== 完成 ✅ ==="
echo "汇总：reports/b4_curriculum_summary.md"
echo "图：  reports/figures/b4_curriculum/"
echo "日志：branches/b4_curriculum/LOG.md"
