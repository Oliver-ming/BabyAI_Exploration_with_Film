#!/usr/bin/env bash
# 支线 2 · FiLM 完整实验流程
#
# 用法：  bash branches/b2_film/run.sh                 # 全流程（约 12 分钟单线程）
#         VARIANTS="film" SEEDS="0" bash branches/b2_film/run.sh
#
# 注意：`concat_nwd` / `film_nwd` 是 §5.1 那个反直觉结论（wd 有害）的关键证据，
#       不能只跑默认 wd 组，否则会误以为 FiLM 的收益里混着正则化差异。
set -euo pipefail

cd "$(dirname "$0")/../.."
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

VARIANTS=${VARIANTS:-"concat film film_only film_res concat_wide"}
SEEDS=${SEEDS:-"0 1 2"}
THREADS=${THREADS:-1}
RUN_NWD=${RUN_NWD:-1}

echo "=== ① 冻结区自检 ==="
git diff --stat -- src/ data/expert_5000.pkl runs/baseline_seed0 runs/baseline_seed1 \
  runs/baseline_seed2 reports/baseline_eval.json reports/baseline_report.md
echo "（无输出 = 冻结区干净）"

echo
echo "=== ② 训练（跑满 30 epoch，事后选 val_acc 最优）==="
for v in $VARIANTS; do
  for s in $SEEDS; do
    python branches/b2_film/train_film.py --variant "$v" --seed "$s" \
      --out "runs/b2_${v}_seed${s}" --threads "$THREADS"
  done
done

if [ "$RUN_NWD" = "1" ]; then
  echo
  echo "=== ②b weight-decay 对照（§5.1 的关键证据）==="
  for s in $SEEDS; do
    python branches/b2_film/train_film.py --variant concat --seed "$s" \
      --weight-decay 0 --film-l2 0 --out "runs/b2_concat_nwd_seed${s}" --threads "$THREADS"
    python branches/b2_film/train_film.py --variant film --seed "$s" \
      --weight-decay 0 --film-l2 0 --out "runs/b2_film_nwd_seed${s}" --threads "$THREADS"
  done
fi

echo
echo "=== ③ 闭环评估（S6N3 + S7N4，复用冻结的评估口径）==="
RUNS=""
for v in $VARIANTS; do
  for s in $SEEDS; do RUNS="$RUNS runs/b2_${v}_seed${s}"; done
done
if [ "$RUN_NWD" = "1" ]; then
  for v in concat film; do
    for s in $SEEDS; do RUNS="$RUNS runs/b2_${v}_nwd_seed${s}"; done
  done
fi
BC30=""
for s in $SEEDS; do BC30="$BC30 runs/b3_bc30_seed$s"; done

python branches/b2_film/eval_film.py --runs $RUNS --baseline-runs $BC30 \
  --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --out reports/b2_film_eval.json
python branches/b2_film/eval_film.py --runs $RUNS --baseline-runs $BC30 \
  --env-id BabyAI-GoToLocalS7N4-v0 --seeds test_s7 --out reports/b2_film_s7n4.json

echo
echo "=== ④ 语言敏感度探针（机制判据）==="
python branches/b2_film/probe_language.py --runs $BC30 $RUNS --n-states 600 \
  --out reports/b2_language_sensitivity.json

echo
echo "=== ⑤ 失败构成对比（次判据：attribute_confusion）==="
FILM_RUNS="${RUNS//runs\/b2_concat_seed[0-9]/}"
python branches/b2_film/compare_failures.py --baseline-runs $BC30 \
  --film-runs runs/b2_film_seed0 runs/b2_film_seed1 runs/b2_film_seed2 \
               runs/b2_film_only_seed0 runs/b2_film_only_seed1 runs/b2_film_only_seed2 \
               runs/b2_film_res_seed0 runs/b2_film_res_seed1 runs/b2_film_res_seed2 \
  --seeds test --out reports/b2_failure_compare.json

echo
echo "=== ⑥ 汇总 + 作图 ==="
python branches/b2_film/make_summary.py
python branches/b2_film/make_figures.py

echo
echo "=== 完成 ✅ ==="
echo "汇总：reports/b2_film_summary.md"
echo "图：  reports/figures/b2_film/"
echo "过程：branches/b2_film/LOG.md"
