#!/usr/bin/env bash
# 基准复现（P0 + P1 + P2）一键跑通。
# 用法：  bash run_baseline.sh
# 预计耗时：专家数据 ~10s，训练 ~35s，评估 ~40s，作图/报告 ~15s
set -euo pipefail

cd "$(dirname "$0")"
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

THREADS=${THREADS:-6}

echo "=== ① 环境验收 ==="
python src/check_env.py

echo
echo "=== ② 生成专家数据（5000 条 = 训练 4500 + 验证 500）==="
python src/gen_expert.py --seed-set all

echo
echo "=== ③ §4.3 验收：数据规模与动作分布 ==="
python - <<'PY'
import pickle, collections
eps = pickle.load(open("data/expert_5000.pkl", "rb"))
tr = sum(e["episode_len"] for e in eps)
c = collections.Counter(a for e in eps for a in e["actions"])
print(f"episodes={len(eps)} transitions={tr} avg_len={tr/len(eps):.2f}")
print("action dist:", {k: f"{v/tr:.1%}" for k, v in c.most_common()})
assert len(eps) == 5000, len(eps)
assert not (set(c) & {3, 4, 5, 6}), f"expert used forbidden actions: {set(c) & {3,4,5,6}}"
print("PASS")
PY

echo
echo "=== ④ 数据审计 ==="
python src/audit_data.py

echo
echo "=== ⑤ 训练 BC 基线（3 seed + 2 消融）==="
pids=()
for s in 0 1 2; do
  python src/train_bc.py --seed "$s" --epochs 30 --out "runs/baseline_seed$s" --threads "$THREADS" &
  pids+=($!)
done
python src/train_bc.py --seed 0 --epochs 30 --img-mode div255 --out runs/ablation_div255_seed0 --threads "$THREADS" &
pids+=($!)
python src/train_bc.py --seed 0 --epochs 30 --no-language --out runs/ablation_nolang_seed0 --threads "$THREADS" &
pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done

echo
echo "=== ⑥ §6.4 验收：三个 seed 的最佳验证准确率 ==="
python - <<'PY'
import glob, json
for d in sorted(glob.glob("runs/baseline_seed*")):
    h = json.load(open(f"{d}/history.json"))
    best = max(h, key=lambda e: e["val_acc"])
    print(f"{d}: best val_acc={best['val_acc']:.3f} @ epoch {best['epoch']}")
PY

echo
echo "=== ⑦ 闭环评估（主指标 + 平凡基线 + 跨房间迁移 + 消融）==="
python src/eval_closed_loop.py --runs runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
  --policies always_forward always_left random \
  --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --out reports/baseline_eval.json
python src/eval_closed_loop.py --runs runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
  --env-id BabyAI-GoToLocalS7N4-v0 --seeds test_s7 --out reports/s7n4_transfer.json
python src/eval_closed_loop.py --runs runs/ablation_div255_seed0 runs/ablation_nolang_seed0 \
  --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --out reports/ablation_eval.json

echo
echo "=== ⑧ mission 分解 / 失败归因 / 轨迹可视化 ==="
python src/mission_breakdown.py
python src/analyze_failures.py --run runs/baseline_seed0 --seeds test
rm -f videos/*.gif videos/index.json reports/failures/* 2>/dev/null || true
python src/render_episodes.py --run runs/baseline_seed0 --n-success 2 --n-failure 4 \
  --require-types orientation_lock,attribute_confusion,distractor_priority,turn_oscillation --scan 300

echo
echo "=== ⑨ 作图 + 汇总报告 ==="
python src/make_figures.py
python src/make_report.py

echo
echo "=== 全部完成 ✅ ==="
echo "总报告：reports/baseline_report.md"
echo "图：    reports/figures/"
echo "gif：   videos/"
