> **历史版本说明（zez-attention-method）**：下文保留旧实验/交付的记录。本分支唯一主比较为 BC30 vs ABC，当前范围与叙事见 `../docs/ATTENTION_STORY.md`。旧文中的主方法、全家族结果或合规结论不能直接用于本分支提交。

# 支线隔离契约与多支线总览（五条候选改进并行开发与归档约定）

本项目的**基线复现**已完成并严格**冻结**。围绕课程手册 §3.2（赛道 2：BabyAI 接地语言模仿学习）列出的全部五项候选改进方向，我们在 `branches/b1_*` .. `branches/b5_*` 下独立完成了系统性受控实验，并依据手册「**指定一项主要比较（One Primary Modification），其余作为消融/辅助实验**」的规则，确立 **`b2_film`（支线 2：FiLM 跨模态融合）** 为本项目的**唯一主修改（Primary Modification）**，其余四条支线（B1 语言编码器、B3 DAgger-lite、B4 课程学习、B5 数据效率）作为已完成的正交验证、辅助对照与次指标（数据效率）支撑研究。

本文件定义了全部五条支线共同遵守的隔离边界、方法论规范与最终完成状态。

---

## 1. 冻结区（任何支线与文档整理均不得修改）

| 路径 | 内容 | 为什么冻结 |
|---|---|---|
| `src/common.py` | seed 分段、14 词词表、`BCNet`、编码工具、`assert_no_leak` | **实验协议的唯一出处**。任一支线改它，数据划分与基线模型定义立刻漂移，跨支线结果不可比 |
| `src/train_bc.py` | BC 基线训练循环（`patience=5`，上限 30 epoch） | 冻结基线数字（`val_acc 0.843` / S6N3 `0.747`）由它产出，改动会让历史结果无法复现 |
| `src/eval_closed_loop.py` | 闭环评估函数 `evaluate()` | **主指标唯一口径**。全部五条支线均直接调用同一套 `evaluate()` |
| `src/gen_expert.py`、`src/audit_data.py` | 专家数据生成与审计 | 5,000 条成功专家 episode（4,500 训练 + 500 验证）是固定预算，禁止重新生成覆盖 |
| `src/mission_breakdown.py`、`src/analyze_failures.py`、`src/render_episodes.py`、`src/make_figures.py`、`src/make_report.py`、`src/check_env.py` | 分析 / 失败分类器 `classify_failure` / 可视化 / 基线报告 | 只读复用，确保失败归因口径跨支线完全一致 |
| `data/expert_5000.pkl` | 5,000 条专家数据 | 支线如需额外或跨环境数据时**必须在 `data/bN_*` 下新建文件**，严禁改写它 |
| `runs/baseline_seed{0,1,2}`、`runs/ablation_*` | 冻结基线与基线消融 checkpoint | 全项目历史对照锚点 |
| `reports/baseline_*`、`reports/data_audit.md`、`reports/mission_breakdown.*`、`reports/s7n4_transfer.json`、`reports/failure_analysis.*`、`reports/failures/`、`reports/figures/*.png`、`videos/**` | 基线全部产物与演示视频 | 同上 |

**允许的唯一复用方式**：`import` 冻结模块并只读调用（如 `from eval_closed_loop import evaluate`、`from render_episodes import classify_failure`）。**不允许**编辑、monkey-patch 或通过环境变量篡改冻结模块语义。

自检命令（期望无任何输出）：

```bash
git diff --stat -- src/ data/expert_5000.pkl \
  runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
  reports/baseline_eval.json reports/baseline_report.md
```

---

## 2. 支线的专属区与命名规范

```text
branches/
├── README.md                 ← 本文件（隔离契约与五支线完成总览）
├── _shared/                  ← 支线共用只读辅助工具 + 归档的语言接地失败案例
├── _template/                ← 支线实验记录骨架（LOG.md / run.sh）
├── b2_film/                  ← ★ 主修改（Primary Modification）：支线 2 · FiLM 跨模态融合
├── b1_langenc/               ← 辅助支线 1：语言编码器消融（GRU / Transformer / DAN / PosMLP）
├── b3_dagger/                ← 辅助支线 3：DAgger-lite 交互式模仿与正交叠加（FiLM+DAgger）
├── b4_curriculum/            ← 辅助支线 4：课程学习与多尺度泛化谱（S5N2..S8N6）
└── b5_data_eff/              ← 次指标与辅助支线 5：数据效率扩展规律与群等变增广
```

每条支线**只写带自身前缀的产物**，互不覆盖：

| 类型 | 位置 | 命名规范 |
|---|---|---|
| 代码与一键脚本 | `branches/bN_xxx/*.py`、`run.sh` | 支线子目录内独立维护 |
| 训练产物 | `runs/bN_<variant>_seed<k>/` | **必须以 `bN_` 开头**（含 `config.json`、`history.json`、`best_model.pt`） |
| 评估与诊断 JSON | `reports/bN_*.json` | 带 `bN_` 前缀，全部可溯源 |
| 汇总文档 | `reports/bN_*_summary.md` | 由各支线 `make_summary.py` 从原始 JSON 自动生成 |
| 图表 | `reports/figures/bN_xxx/*.png` | 子目录隔离 |
| 支线新增数据 | `data/bN_*.pkl` | 仅 B3（交互 rollout）与 B4（多环境课程）新建，B2/B1/B5 零新增环境采样 |
| 研发过程日志 | `branches/bN_xxx/LOG.md` | 记录文献调研、设计约束、受控结果与跨支线关系 |

---

## 3. 每条支线统一执行的四步闭环与双基线口径

全部五条支线均严格走完：**步骤 A（前置文献调研与设计约束）→ 步骤 B（复用冻结评估与分类器实现）→ 步骤 C（3 seed 受控实验 + 机制探针）→ 步骤 D（基于实测现象补充文献与交叉归因）**。

### 关键方法论约定：消除早停混淆的双基线锚点

在支线 3 的早期诊断中发现：原冻结基线 `runs/baseline_seed{0,1,2}` 因采用 `patience=5` 早停，在 `seed1`（epoch 13 停，最佳 epoch 8）和 `seed2`（epoch 14 停，最佳 epoch 9）上过早终止，使其闭环成功率停留在 **`0.747 ± 0.093`**（`val_acc = 0.843 ± 0.033`）。在**不改任何模型参数、数据或超参**的前提下，仅取消 `patience=5` 跑满 30 epoch 并严格按 `val_action_acc` 事后选模（即受控基线 **`BC30`**，`runs/b3_bc30_seed{0,1,2}`），S6N3 成功率即升至 **`0.814 ± 0.045`**（`val_acc = 0.873 ± 0.013`），S7N4 零样本迁移为 **`0.697 ± 0.056`**。

因此，**全项目所有支线（B1–B5）统一以受控 30-epoch `BC30`（`0.814 ± 0.045`）作为公平比较分母**，同时在总表中并列展示原冻结基线（`0.747 ± 0.093`），彻底剥离训练时长/早停协议混淆。全部均值与标准差统一采用 3 个初始化种子（`seed 0/1/2`）的总体标准差（`ddof=0`，即 `statistics.pstdev`），与全部 `reports/*.json` 严格一致。

---

## 4. 共享的评估与机制诊断口径（零漂移）

1. **主指标与迁移指标**：统一调用 `src/eval_closed_loop.py:evaluate()`，S6N3 未见测试集固定 `4500:4800`（300 局/模型），S7N4 零样本跨房间迁移固定 `10000:10300`（300 局/模型）。
2. **选模硬约束**：所有支线 checkpoint **仅由验证集动作准确率 `val_action_acc`（`4000:4500`）选出**，绝不使用闭环成功率选模；训练入口由 `common.assert_no_leak` 硬断言零测试种子泄漏。
3. **反事实语言敏感度探针**：在 183 个场景内确有替换候选物体的测试状态上（138 组同类异色、59 组同色异类），固定 `(image, direction)` 仅替换指令中的颜色或物体类型词，统计贪心动作改变比例。
4. **失败机制分类**：统一调用 `src/render_episodes.py:classify_failure()`，将超时失败归并为**表征接地类**（`attribute_confusion` + `distractor_priority`）与**状态覆盖类**（`turn_oscillation` + `approach_then_lost` + `orientation_lock` + `forward_stall` + `wander_timeout`）。

---

## 5. 五条支线的逻辑关系（为什么最终确立 B2 FiLM 为主修改）

五条支线共同拼出了一幅自洽的因果图景：

1. **瓶颈定位**：`BC30` 每 300 局平均失败 `55.7` 局，其中 **`38.3` 局（68.8%）是表征接地类失败**；反事实探针进一步揭示 Concat 基线的颜色敏感度仅 **`0.065`**（`BC30` 单独为 `0.051`），超过 93% 的情况下更换目标颜色词根本不改变动作（典型的视觉捷径 / Information Collapse）。
2. **为什么选 B2（FiLM）为主修改而非 B1 或 B3**：
   - **对比 B1（语言编码器）**：GoToLocal 指令是长度固定为 5 的无歧义语法模板（`go to the/a <color> <type>`），词序本身不携带额外语义。纯换编码器最强的 `gru_small` 仅将 S6N3 提至 `0.868`（颜色敏感度仍仅 `0.058`）；而保持 448 参数平均词嵌入不变、仅在 `conv1` 后引入单层线性恒等初始化 **FiLM 乘性调制（`b2_film`）**，在**零额外专家样本、仅增 2,112 参数（+0.95%）**下即将 S6N3 推至 **`0.959 ± 0.006`**（`+0.144`）、S7N4 推至 **`0.900 ± 0.021`**（`+0.203`），颜色敏感度跃升 **4.1×（`0.268`）**，表征失败砍掉 **83.5%（`38.3 → 6.3`）**。
   - **对比 B3（DAgger-lite）**：DAgger 需要在线交互与专家重标注，主要修复状态覆盖类失败（`17.3 → 8.3`），对属性混淆无能为力；但当把 B3 叠加在 B2 之上时（`FiLM + DAgger`），S6N3 进一步升至 **`0.970 ± 0.007`**（失败降至 `9.0/300`），完美证明了「表征接地（B2）」与「状态分布纠偏（B3）」的正交性。
   - **联动 B5（数据效率）与 B4（课程泛化）**：B5 证明步数匹配下 `film_step` 仅用 **20% 专家数据（900 ep）即达 `0.814`**，追平 100% 数据的 `BC30`（5× 样本压缩，配合 D4 等变增广更达 10× 压缩）；B4 证明在 FiLM 基础上引入多尺度房间训练可将未见 S7N4 进一步推至 **`0.924`**、S8N6 推至 **`0.844`**。

---

## 6. 当前支线状态（全部完成）

| 支线 | 目录 | 角色定位与状态 | 核心定量结论（全部对比受控 `BC30`: S6N3 `0.814 ± 0.045`, S7N4 `0.697 ± 0.056`, 3 seeds） |
|---|---|---|---|
| **2 · FiLM 跨模态融合** | `b2_film/` | **✅ 完成（★ 唯一主修改 Primary Modification）** | **零额外样本**，`film_nwd` S6N3 **`0.959 ± 0.006`**（`+0.144`, 3/3 正）、S7N4 **`0.900 ± 0.021`**（`+0.203`, 3/3 正）；`film_wd` `0.946 ± 0.009`、`film_only_wd`（省 2k 参数）`0.947 ± 0.007`；等参数 `concat_wide` 仅 `0.718` 排除容量解释；颜色敏感度 `0.065 → 0.268`（4.1×），表征失败 `38.3 → 6.3`（`film_wd` 为 `8.3`） |
| **5 · 数据效率** | `b5_data_eff/` | **✅ 完成（手册规定次指标 + 扩展消融）** | 基线协议 20%/50%/100% S6N3 为 `0.732 / 0.747 / 0.814`；步数对齐下 `bc_step` 为 `0.742 / 0.751 / 0.814`，而 **`film_step@20%`（900 ep）即达 `0.814 ± 0.023`**（5× 压缩追平 100% `BC30`，50% 达 `0.929`）；`film_d4_tta@10%`（450 ep）达 `0.828 ± 0.008`（10× 压缩） |
| **1 · 语言编码器** | `b1_langenc/` | **✅ 完成（辅助/消融支线）** | 11 组受控对比：纯 Concat 下最强 `gru_small`（32d）S6N3 `0.868 ± 0.013`（`+0.053`, 3/3 正）、S7N4 `0.803 ± 0.026`（`+0.107`, 3/3 正），但颜色敏感度仍仅 `0.058`；`FiLM+GRU`（`0.949`/`0.944`）不优于 `FiLM+MeanPool`（`0.959`），且词序打乱时暴跌 `-0.151`～`-0.342`（MeanPool 严格 `0.000`） |
| **3 · DAgger-lite** | `b3_dagger/` | **✅ 完成（辅助/正交支线，v1+v2）** | 发现并修正 `patience=5` 早停低估（`0.747 → 0.814`）；4,500 种子严苛等预算下 `Split-DAgger` 达 `0.851 ± 0.002`（`Split-EnsembleDAgger` 少用 16.9% 标签达 `0.858`）；两轮非重叠 `Loop-Unsticking` 达 `0.909 ± 0.019`（S7N4 `0.828`）；与主修改正交叠加 **`FiLM + DAgger` 达 `0.970 ± 0.007`**（S7N4 `0.918` @ w=1.0，总失败压至 `9.0/300`） |
| **4 · 课程学习** | `b4_curriculum/` | **✅ 完成（辅助/泛化支线）** | 14 组 × 5 环境谱：同域排序 `pace_easy`（`0.857` S6 / `0.766` S7）优于随机阶梯与反课程 `pace_hard`（`0.818`），无干扰预热 `env_easy_cumul`（S7 `0.673`）反而损害接地；`FiLM` 结合多尺度训练（不含 7×7）将未见 **S7N4 推至 `0.924 ± 0.004`**、**S8N6 推至 `0.844`** |
| **2b · Attention 与融合消融** | `b2_attn/` | **✅ 完成（FiLM vs Attention 对照消融）** | 同预算 3-seed：`Spatial-Softmax`（+2,112 参数）达 `0.971 ± 0.003`（S7N4 `0.964`, S8N6 `0.887`）并 3/3 修复拥挤竞争 `seed 4502`，但绕障 `seed 4501` 降至 `1/3`（FiLM `2/3`）；串联融合 `FiLM+Spatial`（`0.959`/S7N4 `0.939`）与 `FiLM+MHA`（`0.963`/S7N4 `0.926`）均可提升跨房间迁移但未超越独立 `Spatial-Softmax` |
