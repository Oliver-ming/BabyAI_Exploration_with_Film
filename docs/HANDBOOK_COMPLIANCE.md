# 手册合规性逐条自查表（对照《AIAA4220 期末仿真项目手册》中英文版）

> **目的**：将课程手册 §2.1 / §2.2 / §3.2 / §5 / §6 / §7 的每一项硬性要求逐条映射到本仓库的可验证文件与实测数据，确保**唯一主修改（支线 2：FiLM 跨模态融合，`b2_film`）**、**双基线无混淆对比**、**规定次指标（含数据效率）**及四条辅助/消融支线（B1 / B3 / B4 / B5）100% 合规可溯源。

---

## A. 共同要求（手册 §2.1 / §2.2）

| # | 手册要求 | 状态 | 仓库证据与溯源路径 |
|---|---|---|---|
| A1 | 冻结软件版本 | ✅ | `requirements-lock.txt`（36 个依赖包精确锁定：`torch==2.15.0+cpu`, `minigrid==3.2.0`, `gymnasium==1.4.0`, `numpy==2.5.4`）、`pyproject.toml` |
| A2 | 记录所用硬件 | ✅ | `docs/environment.md` §1：双路 AMD EPYC 7453（112 逻辑核）、503 GiB RAM、8× NVIDIA RTX 4090（全程仅用 CPU 单线程，`torch.cuda.is_available() == False`） |
| A3 | 记录运行配置 | ✅ | `docs/environment.md` §3：`torch.set_num_threads(1)`（保证同 seed 模型逐字节复现）、`SDL_VIDEODRIVER=dummy`、全链路显式随机种子 |
| A4 | 记录代码 commit | ✅ | 每个 `runs/*/config.json` 均记录 `git_commit`：原基线 `018a695f`，主修改 `b2_film` 与 `BC30` 均为 `6d91f9c1`，辅助支线 B1/B3-v2/B4/B5 为 `84dfe118`（`src/` 语义零变更，详见 `docs/environment.md` §5） |
| A5 | 复现未修改的基线 | ✅ | 原冻结基线（`patience=5`）：S6N3 `0.747 ± 0.093`（`val_acc 0.843 ± 0.033`）；受控满 30 epoch 基线 `BC30`：S6N3 `0.814 ± 0.045`（`val_acc 0.873 ± 0.013`），详见 `reports/baseline_report.md` 与 `reports/b2_film_eval.json` |
| A6 | **选择一项主要修改 + 可检验假设** | ✅ | **唯一主修改：支线 2 · FiLM 跨模态融合（`branches/b2_film/`）**。跑前写死假设与判据见 `branches/b2_film/LOG.md` §1：用单层线性恒等初始化 `(1+Δγ)⊙h1+Δβ` 替代纯拼接，提升闭环 Success、反事实颜色敏感度并压降表征接地类失败（B1/B3/B4/B5 按手册 §3.2 规定均定位为辅助消融） |
| A7 | 同预算、同 seed、同评估代码公平比较 | ✅ | 主修改 `b2_film` 与 `BC30` **零额外专家样本**（严格共用 `data/expert_5000.pkl` 的 4,500 训练 + 500 验证）、同跑满 30 epoch 按 `val_acc` 选模、同 seed `0/1/2`、100% 调用冻结 `src/eval_closed_loop.py:evaluate()` 评测固定未见测试集 `4500:4800` 与 `10000:10300` |
| A8 | 报告不确定性 | ✅ | 全部分组均报告 3-seed `mean ± SD`（总体标准差 `ddof=0`，`statistics.pstdev`）及同 seed 配对差值与正号比（如 `film_nwd` 相对 `BC30` 配对 `+0.144 ± 0.044`，3/3 全正） |
| A9 | **至少解释一例成功与一例失败** | ✅ | `docs/CASE_ANALYSIS.md`：详解 1 例成功（`seed 4501`，8 步绕障）+ 5 例基线语言接地失败（`4516, 4565, 4570, 4582, 4560`，均被 `b2_film_nwd_seed0` 在 2～8 步修复）+ 1 例 `b2_film_nwd` 自身在 7×7 视野盲区下的真实振荡失败（`seed 4568`，由 `FiLM+DAgger` 修复） |
| A10 | 逐 episode / 细粒度原始结果文件 | ✅ | 逐 run + 36 种 mission 明细见 `reports/baseline_eval.json`、`reports/b2_film_eval.json`、`reports/b2_film_s7n4.json`；逐 episode 逐步轨迹见 `reports/b3_per_episode_baseline.json`（即 `baseline_seed0`/`bc30_seed0`）、`reports/failures/seed*.json`、`branches/_shared/lang_grounding_failures/seed*.json` |
| A11 | 汇总表 | ✅ | `reports/b2_film_summary.md`（主修改汇总表）、`reports/figures/summary_table.md`（基线汇总表）、`reports/b{1,3,4,5}_*_summary.md` |
| A12 | 作图脚本 | ✅ | 基线：`src/make_figures.py`；主修改 B2：`branches/b2_film/make_figures.py`（产出 `reports/figures/b2_film/b2_fig{1,2,3}_*.png`）；报告出版级图表：`scripts/` |
| A13 | 30–60 秒演示视频 | ✅ | `videos/demo_all.gif`（197 帧 @ 4fps ≈ **49 秒**，符合 30–60 秒硬区间）及各单局 `videos/*.gif` |
| A14 | 简短 README（安装/基线/主修改/评估/作图命令） | ✅ | `README.md` §1–§2c，并配备顶层一键脚本 `bash run_baseline.sh` 与 **`bash run_film_main.sh`** |
| A15 | 团队贡献说明 | ⚠️ 模板就绪 | `docs/SUBMISSION.md` 及书面报告 Appendix A 已备好规范模板，提交前按小组成员实际分工填入姓名/学号 |
| A16 | 生成式 AI 使用披露 | ✅ | `docs/AI_USAGE.md`（按手册 §7 完整披露工具、B1–B5 与主线切换至 B2 的实质帮助范围、人工逐项核验及已修正的 AI 引入错误） |

---

## B. 赛道 2 专属要求（手册 §3.2：BabyAI 接地语言模仿学习）

| # | 手册要求 | 状态 | 仓库证据与实测数值 |
|---|---|---|---|
| B1 | 规定环境 `BabyAI-GoToLocalS6N3-v0` | ✅ | 全程一致（`src/common.py:ENV_ID`） |
| B2 | 专家预算 ≤ 5,000 条成功专家 episode | ✅ | `data/expert_5000.pkl`：**训练 4,500 + 验证 500 = 5,000 条**（100% 成功，零超限；主修改 B2 额外专家数为 **0**；seed 分段合规修正说明见下节 D） |
| B3 | 训练预算 ≤ 30 epoch（含早停） | ✅ | 原基线 `patience=5`（上限 30）；`BC30` 与主修改 `b2_film` 统一跑满 30 epoch 后按验证准确率选模 |
| B4 | 3 个神经网络初始化 seed | ✅ | 全部实验组均固定为 `seed 0, 1, 2` |
| B5 | 每模型评估 300 个未见环境 seed | ✅ | S6N3 固定 `4500:4800`（300 局），S7N4 固定 `10000:10300`（300 局） |
| B6 | 使用 `BabyAIBot` 生成成功专家轨迹 | ✅ | `src/gen_expert.py` 调用 `minigrid.utils.baby_ai_bot.BabyAIBot`，失败轨迹 0 条 |
| B7 | 按环境 seed 划分，测试 seed 绝不进训练 | ✅ | `src/common.py:split_of` 与 `assert_no_leak` 在每次训练入口硬断言拦截 |
| B8 | 小型 CNN 编码 7×7 图像 | ✅ | `nn.Embedding(11, 8)` 逐通道查表 → 24 通道 → 两层 3×3 `Conv2d(32)` + ReLU |
| B9 | 固定词表 + 平均 32 维词嵌入 | ✅ | `VOCAB` 14 词（含 `<pad>`），非 pad 词掩码均值池化为 32 维向量（基线与主修改 B2 均保持此 448 参数语言嵌入不变） |
| B10 | one-hot 编码方向 | ✅ | `F.one_hot(direction, num_classes=4)` |
| B11 | 拼接 + 128 隐层 + 交叉熵预测 7 动作 | ✅ | 基线 `BCNet` 严格 `Linear(1604→128) → ReLU → Linear(128→7)` + `CrossEntropyLoss` |
| B12 | **仅用验证集动作准确率选择 checkpoint** | ✅ | `src/train_bc.py` 与 `branches/b2_film/train_film.py` 全程仅用 `val_acc` 选出 `best_model.pt`，严禁使用闭环 Success 选模 |
| B13 | 闭环任务 Success 为评估，单独准确率不是任务指标 | ✅ | 报告并列对比平凡策略 `always_forward`（验证准确率高达 `56.4%`，闭环 Success 仅 `0.060`），并在双基线与 FiLM 中联合呈现两者 |
| B14 | **主指标：未见 S6N3 闭环 Success** | ✅ | 原基线 `0.747 ± 0.093` → 受控 `BC30` `0.814 ± 0.045` → **主修改 `B2 FiLM` (`film_nwd`) `0.959 ± 0.006`**（`+0.144`, 3/3 正；`film_wd` `0.946 ± 0.009`；`film_only_wd` `0.947 ± 0.007`；等参数 `concat_wide` `0.718 ± 0.023`） |
| B15 | 次指标：平均回报 (`mean_return`) | ✅ | 原基线 `0.656` → `BC30` `0.723 ± 0.037` → **`B2 FiLM` (`film_nwd`) `0.871 ± 0.005`**（`film_wd` `0.858`） |
| B16 | 次指标：成功 episode 长度中位数 | ✅ | 原基线与 `BC30` 均为 **`4.0` 步** → **`B2 FiLM`（全部 FiLM 变体）缩短至 `3.0` 步** |
| B17 | 次指标：留出验证集动作准确率 (`val_action_acc`) | ✅ | 原基线 `0.843 ± 0.033` → `BC30` `0.873 ± 0.013` → **`B2 FiLM` (`film_nwd`) `0.972 ± 0.004`**（`film_wd` `0.971 ± 0.005`） |
| B18 | **次指标：`BabyAI-GoToLocalS7N4-v0` 零样本迁移 Success** | ✅ | 原基线 `0.674 ± 0.082` → `BC30` `0.697 ± 0.056` → **`B2 FiLM` (`film_nwd`) `0.900 ± 0.021`（`+0.203`, 3/3 正，跨尺度增益甚至大于同分布 `+0.144`；`film_wd` `0.862 ± 0.017`）** |
| B19 | **次指标：数据效率（20% / 50% / 100%）** | ✅ | ① 基线 30-epoch 协议（`reports/b5_data_efficiency_eval.json`）：20% `0.732 ± 0.007`、50% `0.747 ± 0.012`、100% `0.814 ± 0.045`；② 步数对齐（`reports/b5_data_eff_summary.md`）：`bc_step` 为 `0.742 / 0.751 / 0.814`，而**主修改 `film_step` 为 `0.814 / 0.929 / 0.959`（仅 20% 样本即追平 100% `BC30`，5× 压缩）** |
| B20 | 规定证据：训练/验证损失曲线 | ✅ | 基线 `reports/figures/fig1_loss_curves.png` 及各 `runs/b2_*/history.json` |
| B21 | 规定证据：三 seed Success 对比 | ✅ | `reports/b2_film_summary.md`、`reports/figures/b2_film/b2_fig1_success.png` |
| B22 | 规定证据：按 mission 类型分解 | ✅ | `reports/mission_breakdown.{csv,md}`、`reports/figures/fig4_mission_heatmap.png` 及 `reports/b2_film_eval.json` 内 `per_mission` 字典 |
| B23 | 规定证据：**至少三个语言接地失败案例** | ✅ | 共归档并逐例剖析 **5 个语言接地失败案例**（`seed 4516, 4565, 4570, 4582, 4560`），并在 `docs/CASE_ANALYSIS.md` 实测给出 `B2 FiLM` 对全部 5 例的精确修复步数与动作序列 |

---

## C. 多候选探索与「唯一主修改」合规界定（手册 §3.2 说明）

手册 §3.2 明确规定：「*若测试多种变体，须指定一项主要比较，其余作为消融实验（If you test several variants, designate one primary comparison and treat the rest as ablations）。*」

本项目严格遵循该规则：
- **唯一指定主要比较（Primary Comparison）**：**`BC30`（受控 30-epoch 未修改基线） vs `B2 FiLM`（`b2_film`，跨模态特征线性调制）**，同用 5,000 条专家预算（零额外样本），辅以参数量匹配对照 `concat_wide`、纯调制对照 `film_only`、正则化对照 `wd=0 vs 1e-4` 及反事实语言敏感度探针。
- **辅助/消融与次指标实验**：`B5`（支撑规定次指标「数据效率」）、`B1`（证明仅换语言编码器无法解决颜色接地）、`B3`（诊断 `patience=5` 早停混淆，并验证 `FiLM + DAgger = 0.970` 的正交互补性）、`B4`（验证课程排序与多尺度泛化谱），以及 `b2_attn`（与三类 Attention 及两种串联 `FiLM+Attention` 融合的同预算受控消融，见 `reports/b2_attn_summary.md`）。

---

## D. 主动向评审说明的一处必要协议修正（手册 §5 要求披露任何偏离）

手册 §3.2 规定「最多生成 5,000 条成功专家 episode」，而早期开发指导示例曾写 `TRAIN_SEEDS = range(0, 4000)` 生成 5,000 条训练轨迹。我们在实现中将专家种子固定划分为 **训练 `0..4000 ∪ 4800..5300`（4,500 条）+ 验证 `4000..4500`（500 条）= 合计恰好 5,000 条唯一专家 episode**：
1. 单个确定性环境种子仅对应唯一一条专家轨迹，`0..4000` 无法在不注入 1,000 条重复样本的前提下凑满 5,000 条；
2. 手册强制要求**仅用验证集动作准确率选 checkpoint**，验证集 `4000..4500` 必须消耗 500 条专家轨迹预算；若训练集占满 5,000 条，总专家轨迹将达 5,500 条而违反 5,000 条上限；
3. 补足段 `4800..5300` 与验证集 `4000..4500`、测试集 `4500..4800` 及跨房间测试集 `10000..10300` **严格互不相交**（`src/common.py:assert_no_leak` 硬断言保护），完全满足零测试泄漏与 ≤5,000 条总预算两项硬约束。
