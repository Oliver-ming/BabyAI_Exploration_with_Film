# P3 · 赛道 2 · BabyAI 语言模仿学习（GoToLocal）—— 基准复现

**当前进度：P0 + P1 + P2 全部完成（基准复现阶段）。P3（DAgger-lite）与 P4（数据效率消融）尚未动工。**

核心结论一句话：**验证集动作准确率 0.843，闭环 Success 只有 0.747；最强平凡基线「永远 forward」准确率 56.4% 但 Success 0.060。
所有失败 episode 都跑满 36 步上限——失败不是「走得远」，而是「卡住」。** 完整分析见 `reports/baseline_report.md`。

---

## 1. 安装（uv + env_embodied）

```bash
# uv
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env

# 项目环境（必须叫 env_embodied）
uv venv env_embodied --python 3.12
source env_embodied/bin/activate

# 依赖：照 requirements-lock.txt 精确复现
uv pip install -r requirements-lock.txt
```

**关键版本（锁死，三个 seed 必须同环境）**：`torch==2.15.0+cpu`、`minigrid==3.2.0`、`gymnasium==1.4.0`、`numpy==2.5.4`。

无显示器服务器：`source env_embodied/bin/activate` 时已自动导出 `SDL_VIDEODRIVER=dummy`（写在 activate 末尾）；
用别的 shell 时请手动 `export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy`。

## 2. 五组命令（从零到全部产物）

```bash
source env_embodied/bin/activate
mkdir -p third_party && cd third_party        # 只读参考源码，不要加进 PYTHONPATH
git clone --depth 1 --branch v3.2.0 https://github.com/Farama-Foundation/Minigrid.git minigrid-src
git clone --depth 1 https://github.com/mila-iqia/babyai.git babyai-orig
cd ..

# ① 验收环境（§3.3）
python src/check_env.py

# ② 生成专家数据（5000 条 = 训练 4500 + 验证 500）
python src/gen_expert.py --seed-set all          # -> data/expert_5000.pkl

# ③ 数据审计（§5 的六项）
python src/audit_data.py                         # -> reports/data_audit.md

# ④ 训练基线（3 个 seed，约 30 秒）
for s in 0 1 2; do
  python src/train_bc.py --seed $s --epochs 30 --out runs/baseline_seed$s --threads 6
done

# ⑤ 闭环评估（主指标 + 平凡基线 + 跨房间迁移）
python src/eval_closed_loop.py --runs runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
  --policies always_forward always_left random \
  --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --out reports/baseline_eval.json
python src/eval_closed_loop.py --runs runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
  --env-id BabyAI-GoToLocalS7N4-v0 --seeds test_s7 --out reports/s7n4_transfer.json

# ⑥ 分析：mission 分解 / 失败归因 / 轨迹 gif
python src/mission_breakdown.py                  # -> reports/mission_breakdown.{csv,md}
python src/analyze_failures.py --run runs/baseline_seed0 --seeds test   # -> reports/failure_analysis.md
python src/render_episodes.py --run runs/baseline_seed0 --n-success 2 --n-failure 4 \
  --require-types orientation_lock,attribute_confusion,distractor_priority,turn_oscillation

# ⑦ 作图 + 汇总报告
python src/train_bc.py --seed 0 --epochs 30 --img-mode div255 --out runs/ablation_div255_seed0
python src/train_bc.py --seed 0 --epochs 30 --no-language --out runs/ablation_nolang_seed0
python src/eval_closed_loop.py --runs runs/ablation_div255_seed0 runs/ablation_nolang_seed0 \
  --seeds test --out reports/ablation_eval.json
python src/make_figures.py                       # -> reports/figures/
python src/make_report.py                        # -> reports/baseline_report.md
```

一键跑完全程：`bash run_baseline.sh`

## 3. 脚本清单

| 脚本 | 作用 | 主要产物 |
|---|---|---|
| `src/common.py` | **seed 划分 + 模型 + 编码的唯一定义处** | — |
| `src/check_env.py` | §3.3 验收（obs shape / max_steps / BabyAIBot） | 终端 PASS |
| `src/gen_expert.py` | BabyAIBot 生成成功专家 episode | `data/expert_5000.pkl` |
| `src/audit_data.py` | §5 六项数据审计 | `reports/data_audit.md` |
| `src/train_bc.py` | BC 基线（§6 架构），支持 `--data-frac` / `--img-mode` / `--no-language` | `runs/*/` |
| `src/eval_closed_loop.py` | 闭环评估（主指标），支持平凡策略对照 | `reports/*.json` |
| `src/mission_breakdown.py` | §9.1 mission 类型分解 | `reports/mission_breakdown.{csv,md}` |
| `src/analyze_failures.py` | §9.3 全量失败归因统计 | `reports/failure_analysis.{md,json}` |
| `src/render_episodes.py` | §8 回放 gif + 失败案例 gif/json/png | `videos/`, `reports/failures/` |
| `src/make_figures.py` | §11.1 图 1/2/4/6/7 + 汇总表 | `reports/figures/` |
| `src/make_report.py` | 把全部产物拼成完整报告 | `reports/baseline_report.md` |

## 4. ⚠️ 对指导文档 seed 配置的两处必要修正

指导文档 §1.3 写 `TRAIN_SEEDS = range(0, 4000)` 生成 **5000 条**专家演示。这做不到，原因有二：

1. **一个 env seed 只能产出一条确定性 episode**，4000 个 seed 最多 4000 条**不重复**样本。
   循环抽样凑 5000 条会往训练集注入 1000 条完全重复的 episode，直接污染 §10.2 的数据效率消融。
2. **验证集也需要专家动作**。§1.2 ① 强制用验证集动作准确率选 checkpoint，所以 VAL seed 上也要跑专家（额外 500 条）。
   若训练集仍占满 5000 条，专家总数变 5500，**违反手册「最多 5000 条成功专家 episode」的硬上限**。

本实现的划分（`src/common.py` 是唯一定义处）：

| 用途 | seed 段 | 条数 |
|---|---|---|
| 训练专家（主段） | `0..4000` | 4000 |
| 训练专家（补足段） | `4800..5300` | 500 |
| 验证专家 | `4000..4500` | 500 |
| 测试 | `4500..4800` | 300 |
| 迁移测试 S7N4 | `10000..10300` | 300 |

- **专家 episode 总数 = 4500 + 500 = 5000，正好等于上限、未超。**
- 补足段 `4800..5300` 与 VAL / TEST / TEST_S7 全不相交，也从未参与评估，
  所以「测试 seed 绝不进训练」仍然成立。

## 5. 主要结果（`reports/baseline_report.md` 摘要）

| run | val 动作准确率 | 闭环 Success | 平均回报 | 成功长度中位数 | 超时 |
|---|---|---|---|---|---|
| baseline_seed0 | 0.890 | **0.877** | 0.773 | 4.0 | 37/300 |
| baseline_seed1 | 0.817 | **0.663** | 0.584 | 4.0 | 101/300 |
| baseline_seed2 | 0.823 | **0.700** | 0.612 | 4.0 | 90/300 |
| **mean ± SD** | 0.843 ± 0.033 | **0.747 ± 0.093** | 0.656 | 4.0 | — |

| 平凡策略 | teacher-forcing 准确率 | 闭环 Success |
|---|---|---|
| always `forward` | 56.4% | 0.060 |
| always `left` | 22.0% | 0.000 |
| uniform random（转向/前进） | ≈33% | 0.543 |

- **S7N4 零样本迁移**：Success 从 0.747 降到 0.674（Δ = −0.072），零适配直接加载。
- **mission 分解**：`a` 冠词 0.937 > `the` 冠词 0.724（多目标场景容错更高）。
- **失败归因**（seed0, 300 ep）：`attribute_confusion` 19、`distractor_priority` 12、
  `turn_oscillation` 4、`approach_then_lost` 1、`orientation_lock` 1；**全部 37 条失败都跑满 36 步**。
- **失败案例**：`reports/failures/` 5 个（覆盖 §8.3 要求的 `attribute_confusion` /
  `distractor_priority` / `orientation_lock` 三类 + 两个补充类），各含 gif、逐步动作日志、明确归因。

## 6. 目录

```
p3-babyai/
├── env_embodied/              # 虚拟环境（不提交）
├── requirements-lock.txt      # uv pip freeze，复现凭证
├── pyproject.toml
├── run_baseline.sh            # 一键跑完 P0→P2
├── src/                       # 全部脚本（见 §3）
├── data/expert_5000.pkl       # 5000 条专家 episode（含 split 字段）
├── runs/*/                    # config.json / vocab.json / best_model.pt / history.json / predictions_val.npz
├── reports/
│   ├── data_audit.md
│   ├── baseline_report.md     # ★ 总报告
│   ├── baseline_eval.json
│   ├── s7n4_transfer.json
│   ├── ablation_eval.json
│   ├── mission_breakdown.{csv,md}
│   ├── failure_analysis.{md,json}
│   ├── failures/              # 失败案例 gif/png/json（4 类各 1 个）
│   └── figures/               # 图 1/2/4/6/7 + summary_table.md
├── videos/                    # 成功/失败轨迹 gif
├── docs/                      # 指导文档
└── third_party/               # 只读参考源码（不提交）
```
