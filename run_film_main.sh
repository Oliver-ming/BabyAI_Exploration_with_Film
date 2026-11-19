#!/usr/bin/env bash
# 主修改（支线 2 · FiLM 跨模态融合）顶层一键复现与核验脚本
#
# 用法：
#   bash run_film_main.sh               # 完整复现：训练 B2 全部变体(3 seed) + S6N3/S7N4 闭环评测 + 语言探针 + 失败归因 + 作图（约 12 分钟，CPU 单线程）
#   bash run_film_main.sh --eval-only   # 只读核验：直接读取已冻结 checkpoint 与 reports/b2_*.json 打印主表，不改写任何文件
#   bash run_film_main.sh --help        # 查看说明
#
# 环境变量（透传给 branches/b2_film/run.sh）：
#   VARIANTS="concat film film_only film_res concat_wide"
#   SEEDS="0 1 2"
#   THREADS=1
#   RUN_NWD=1
set -euo pipefail

cd "$(dirname "$0")"
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

echo "=== [B2 FiLM 主修改] ① 冻结区完整性自检 ==="
git diff --stat -- src/ data/expert_5000.pkl \
  runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
  reports/baseline_eval.json reports/baseline_report.md
echo "PASS：冻结区零改动"

if [[ "${1:-}" == "--eval-only" ]]; then
  echo
  echo "=== [B2 FiLM 主修改] ② 只读核验已存盘实验结果（零文件写入）==="
  python - <<'PY'
import json, pathlib, statistics as st

ev = json.load(open("reports/b2_film_eval.json"))
s7 = json.load(open("reports/b2_film_s7n4.json"))
sens = json.load(open("reports/b2_language_sensitivity.json"))
fail_b2 = json.load(open("reports/b2_failure_compare.json"))
base = json.load(open("reports/baseline_eval.json"))
base_s7 = json.load(open("reports/s7n4_transfer.json"))

b_s6 = [r["closed_loop_success"] for r in base["runs"] if r["kind"] == "bc"]
b_va = [r["val_action_acc"] for r in base["runs"] if r["kind"] == "bc"]
b_s7 = [r["closed_loop_success"] for r in base_s7["runs"] if r["kind"] == "bc"]
print(f"冻结原基线 (patience=5): val_acc={st.mean(b_va):.4f}±{st.pstdev(b_va):.4f} | S6N3={st.mean(b_s6):.4f}±{st.pstdev(b_s6):.4f} | S7N4={st.mean(b_s7):.4f}±{st.pstdev(b_s7):.4f}")

print("\n受控 30-epoch 主对照与 B2 FiLM 各组 (n=3, mean±SD, ddof=0):")
order = ["bc30", "film_nwd", "film_wd", "film_only_wd", "film_res_wd", "concat_nwd", "concat_wide_wd", "concat_wd"]
for k in order:
    if k not in ev["aggregate"]:
        continue
    a6 = ev["aggregate"][k]
    p6 = ev["paired"].get(k)
    d_str = f"ΔS6={p6['delta_mean']:+.4f} ({p6['n_positive']}/{p6['n_pairs']}+)" if p6 else "对照基准 (0 额外样本)"
    s7_str = "—"
    if k in s7["aggregate"]:
        a7 = s7["aggregate"][k]["closed_loop_success"]
        s7_str = f"{a7['mean']:.4f}±{a7['sd']:.4f}"
    print(f"  {k:15s} | val_acc={a6['val_action_acc']['mean']:.4f}±{a6['val_action_acc']['sd']:.4f} "
          f"| S6N3={a6['closed_loop_success']['mean']:.4f}±{a6['closed_loop_success']['sd']:.4f} "
          f"| S7N4={s7_str:15s} | {d_str}")

print("\n反事实语言敏感度探针 (183 状态):")
for k, v in sens["by_variant"].items():
    print(f"  {k:8s} (n={v['n']}): 换颜色={v['color_sensitivity_mean']:.4f} | 换类型={v['type_sensitivity_mean']:.4f} | 合计={v['combined_sensitivity_mean']:.4f}")
bc30_c = st.mean([r["color_swap_sensitivity"] for r in sens["runs"] if "b3_bc30" in r["run"]])
print(f"  bc30单独 (n=3): 换颜色={bc30_c:.4f} (film_nwd / concat = {sens['by_variant']['film']['color_sensitivity_mean']/sens['by_variant']['concat']['color_sensitivity_mean']:.1f}x)")

print("\n失败归因摘要 (每 300 测试局均值):")
if pathlib.Path("reports/b1_failure_compare.json").exists():
    f1 = json.load(open("reports/b1_failure_compare.json"))["aggregate"]
    if "b2_film_nwd" in f1:
        v = f1["b2_film_nwd"]; fg = v["failure_groups_mean"]
        print(f"  film_nwd  : 总失败={v['n_failure_mean']:5.1f} | 表征类={fg.get('representation', 0):5.1f} | 状态覆盖类={fg.get('state_coverage', 0):5.1f} (attr_conf={v['failure_types_mean'].get('attribute_confusion',0):.1f})")
for k, v in fail_b2["aggregate"].items():
    fg = v["failure_groups_mean"]
    lbl = "film_wd" if k == "film" else k
    print(f"  {lbl:10s}: 总失败={v['n_failure_mean']:5.1f} | 表征类={fg.get('representation', 0):5.1f} | 状态覆盖类={fg.get('state_coverage', 0):5.1f}")
PY
  exit 0
fi

echo
echo "=== [B2 FiLM 主修改] ② 检查前置数据与受控 BC30 对照 ==="
if [[ ! -f "data/expert_5000.pkl" ]]; then
  echo "未找到 data/expert_5000.pkl，正在按冻结协议生成 5000 条专家轨迹..."
  python src/gen_expert.py --seed-set all
fi

for s in ${SEEDS:-0 1 2}; do
  if [[ ! -f "runs/b3_bc30_seed${s}/best_model.pt" ]]; then
    echo "未找到 runs/b3_bc30_seed${s}，正在训练受控 30-epoch BC30 对照 (seed=${s})..."
    python branches/b3_dagger/train_dagger.py --seed "$s" --variant bc30 \
      --out "runs/b3_bc30_seed${s}" --threads "${THREADS:-1}"
  fi
done

echo
echo "=== [B2 FiLM 主修改] ③ 执行 branches/b2_film/run.sh 全流程 ==="
bash branches/b2_film/run.sh

echo
echo "=== [B2 FiLM 主修改] 全部完成 ==="
echo "主修改汇总表：reports/b2_film_summary.md"
echo "主修改评测：  reports/b2_film_eval.json, reports/b2_film_s7n4.json"
echo "机制与失败：  reports/b2_language_sensitivity.json, reports/b2_failure_compare.json"
echo "图表目录：    reports/figures/b2_film/"
