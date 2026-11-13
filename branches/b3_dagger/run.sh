#!/usr/bin/env bash
# 支线 3 · DAgger-lite 一键复现（约 12 分钟）
# 用法：bash branches/b3_dagger/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

EP=1500

echo "=== ① 采集 DAgger（每个策略用自己的 seed）==="
for s in 0 1 2; do
  python branches/b3_dagger/collect_dagger.py --run runs/baseline_seed$s \
    --episodes $EP --out data/b3_dagger1500_pol$s.pkl
done

echo "=== ② 等量对照：同样条数的额外专家数据 ==="
python branches/b3_dagger/collect_dagger.py --episodes $EP --mode expert \
  --out data/b3_expertextra_1500.pkl

echo "=== ③ 训练：bc30（无早停基线）/ bc_extra / dagger / nodedup ==="
pids=()
for s in 0 1 2; do
  python branches/b3_dagger/train_dagger.py --seed $s --variant bc \
    --out runs/b3_bc30_seed$s > /tmp/b3_bc30_$s.log 2>&1 & pids+=($!)
  python branches/b3_dagger/train_dagger.py --seed $s --variant bc_extra \
    --dagger-data data/b3_expertextra_1500.pkl --dedup episode \
    --out runs/b3_bcextra_seed$s > /tmp/b3_extra_$s.log 2>&1 & pids+=($!)
  python branches/b3_dagger/train_dagger.py --seed $s --variant dagger \
    --dagger-data data/b3_dagger1500_pol$s.pkl --dedup episode \
    --out runs/b3_dagger_seed$s > /tmp/b3_dag_$s.log 2>&1 & pids+=($!)
  python branches/b3_dagger/train_dagger.py --seed $s --variant dagger \
    --dagger-data data/b3_dagger1500_pol$s.pkl --dedup none \
    --out runs/b3_nodedup_seed$s > /tmp/b3_nd_$s.log 2>&1 & pids+=($!)
  python branches/b3_dagger/train_dagger.py --seed $s --variant dagger \
    --dagger-data data/b3_dagger1500_pol$s.pkl --dedup global \
    --out runs/b3_novel_seed$s > /tmp/b3_nv_$s.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "=== ③b 第二轮 DAgger（迭代）==="
for s in 0 1 2; do
  python branches/b3_dagger/collect_round2.py --policy runs/b3_dagger_seed$s \
    --episodes $EP --out data/b3_dagger_r2_pol$s.pkl
done
for s in 0 1 2; do
  python branches/b3_dagger/train_dagger.py --seed $s --variant dagger \
    --dagger-data data/b3_dagger1500_pol$s.pkl,data/b3_dagger_r2_pol$s.pkl \
    --dedup episode --out runs/b3_r2_seed$s > /tmp/b3_r2_$s.log 2>&1 &
done
wait

echo "=== ④ 闭环评估（复用冻结脚本）==="
ALL=""
for s in 0 1 2; do
  ALL="$ALL runs/b3_bc30_seed$s runs/b3_bcextra_seed$s runs/b3_dagger_seed$s runs/b3_nodedup_seed$s runs/b3_novel_seed$s runs/b3_r2_seed$s"
done
python src/eval_closed_loop.py --runs $ALL --env-id BabyAI-GoToLocalS6N3-v0 --seeds test \
  --out reports/b3_dagger_eval.json
python src/eval_closed_loop.py --runs runs/b3_dagger_seed0 runs/b3_dagger_seed1 runs/b3_dagger_seed2 \
  runs/b3_r2_seed0 runs/b3_r2_seed1 runs/b3_r2_seed2 \
  --env-id BabyAI-GoToLocalS7N4-v0 --seeds test_s7 --out reports/b3_dagger_s7n4.json

echo "=== ⑤ 失败构成对比 + 机制验证 ==="
python branches/b3_dagger/compare_failures.py \
  --runs runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
         runs/b3_bc30_seed0 runs/b3_bc30_seed1 runs/b3_bc30_seed2 \
         runs/b3_bcextra_seed0 runs/b3_bcextra_seed1 runs/b3_bcextra_seed2 \
         runs/b3_dagger_seed0 runs/b3_dagger_seed1 runs/b3_dagger_seed2 \
         runs/b3_nodedup_seed0 runs/b3_nodedup_seed1 runs/b3_nodedup_seed2 \
  --seeds test --out reports/b3_dagger_failure_compare.json
python branches/b3_dagger/verify_attribute_mechanism.py

echo "=== ⑥ 诊断 + 汇总 ==="
python branches/_shared/check_label_conflicts.py --data data/expert_5000.pkl --split train \
  --out reports/b3_conflict_expert.json
python branches/_shared/check_label_conflicts.py --data data/b3_dagger1500_pol0.pkl \
  --out reports/b3_conflict_dagger.json
python branches/b3_dagger/diagnose_labels.py
python branches/b3_dagger/make_summary.py

echo
echo "=== 完成 ✅ ==="
echo "汇总：reports/b3_dagger_summary.md"
echo "日志：branches/b3_dagger/LOG.md"
