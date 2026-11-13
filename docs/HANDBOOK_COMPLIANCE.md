# 手册合规性自查（对照《AIAA4220 期末仿真项目手册（中文）》）

> 目的：把"手册要求哪些、我们做到哪、还缺什么"逐条列出来，避免在实践中漏项。
> 手册原文提取自 `AIAA4220_Project3_资料包/.../AIAA4220_期末仿真项目手册_中文.pdf`。

## A. 共同要求（手册 §2.1 / §2.2）

| # | 手册要求 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| A1 | 冻结软件版本 | ✅ | `requirements-lock.txt`（36 包精确 pin）、`pyproject.toml`、`runs/*/config.json` 里逐运行记录 |
| A2 | **记录所用硬件** | ❌ **缺失** | 只在对话里查过，**没写进任何文件**。需要落到 README 或 `docs/environment.md` |
| A3 | **记录配置** | ⚠️ 部分 | `config.json` 有超参 + 数据规模，**但硬件、OS、驱动、`torch.get_num_threads()` 未记录** |
| A4 | **记录代码 commit** | ✅ | `config.json:git_commit` 字段。注意：baseline 记的是 `018a695`，b3 记的是 `6d91f9c`（不同时点，需说明都用同一份源码） |
| A5 | 复现未修改的基线 | ✅ | `reports/baseline_report.md`、`reports/baseline_eval.json` |
| A6 | 一项主修改 + 可检验假设 | ✅ | 支线 3 DAgger：`branches/b3_dagger/LOG.md` §1 明确写了假设与**跑前定死的判据** |
| A7 | 同预算、同 seed、同评估代码下公平比较 | ✅ | 全部复用 `src/eval_closed_loop.py`；测试 seed 固定 `4500:4800`；`common.assert_no_leak` 硬断言 |
| A8 | 报告不确定性 | ✅ | 全部结果给 mean ± SD（3 seed） |
| A9 | **至少解释一例成功与一例失败** | ✅ 失败（8 例）/ ❌ 成功（未单独剖析） | `reports/failures/` 有 8 个失败案例；**成功案例只放了 gif，没有逐步解释** |
| A10 | 逐 episode 原始结果（CSV 或 JSON） | ✅ 补上了 | `reports/b3_per_episode_baseline.json` / `_dagger.json`（用 `--save-trajectories` 生成） |
| A11 | 汇总表 | ✅ | `reports/figures/summary_table.md`、`reports/b3_dagger_summary.md` |
| A12 | 作图脚本 | ✅ | `src/make_figures.py`、`branches/b3_dagger/make_summary.py` |
| A13 | 30–60 秒演示视频 | ✅ | `videos/demo_all.gif`（197 帧 @ 4fps ≈ 49 秒） |
| A14 | 简短 README（安装/基线/修改版/评估/作图命令） | ⚠️ **缺"修改版"命令** | README 只写了 P0–P2 的五组命令，**没有 `branches/b3_dagger/run.sh` 的入口** |
| A15 | **团队贡献说明** | ❌ **缺失** | 需新建 |
| A16 | **生成式 AI 使用披露** | ❌ **缺失** | 需新建（本项目的代码与报告由 Cursor AI 大量生成，必须披露） |

## B. 赛道 2 专属（手册 §3.2）

| # | 手册要求 | 状态 | 证据 / 缺口 |
|---|---|---|---|
| B1 | 环境 `BabyAI-GoToLocalS6N3-v0` | ✅ | 全程一致 |
| B2 | 专家 episode ≤ 5000 | ✅ | 4500 训练 + 500 验证 = 5000（**偏离**：训练段展宽到 `4800..5300`，已写明理由） |
| B3 | 训练 ≤ 30 epoch（含早停） | ✅ | 支线全部跑满 30 epoch |
| B4 | 3 个初始化 seed | ✅ | 所有组都是 seed 0/1/2 |
| B5 | 每模型 300 个未见 seed | ✅ | `4500:4800` |
| B6 | BabyAIBot 生成成功专家轨迹 | ✅ | `src/gen_expert.py` |
| B7 | 按环境 seed 划分，测试 seed 不进训练 | ✅ | `common.py` 的 `assert_no_leak` + `split_of` |
| B8 | 小型 CNN/MLP 编码 7×7 图像 | ✅ | `BCNet.vision`（2 层 conv，32 通道） |
| B9 | 固定词表 + 平均 32 维词嵌入 | ✅ | `VOCAB` 14 词、`lang_dim=32` |
| B10 | one-hot 编码方向 | ✅ | `F.one_hot(direction, 4)` |
| B11 | 拼接 + 128 隐层 + 交叉熵 + 7 动作 | ✅ | 严格一致 |
| B12 | **仅用验证集动作准确率选 checkpoint** | ✅ | `train_bc.py` 与 `train_dagger.py` 的选模准则 |
| B13 | 闭环 Success 为评估，单独准确率不是指标 | ✅ | 报告里两者并列并解释差距 |
| B14 | 主指标：未见 seed 闭环 Success | ✅ | `reports/b3_dagger_eval.json` |
| B15 | 次指标：平均回报 | ✅ | 已报告 |
| B16 | 次指标：成功 episode 长度中位数 | ✅ | 已报告（4.0） |
| B17 | 次指标：留出动作准确率 | ✅ | `val_action_acc` |
| B18 | 次指标：S7N4 上的 Success | ✅ | 0.674（基线）→ 0.747/0.769（DAgger） |
| B19 | **次指标：数据效率** | ✅ **已补** | `reports/b5_data_efficiency_eval.json`（20/50/100%）—— 但它是**另一条支线（支线 5）**的核心内容，此处只作为次指标的最小补充 |
| B20 | 证据：训练/验证损失 | ✅ | `fig1_loss_curves.png` |
| B21 | 证据：三 seed Success 对比 | ✅ | `summary_table.md` |
| B22 | 证据：**按 mission 类型分解** | ✅ | `reports/mission_breakdown.{csv,md}`、`fig4` |
| B23 | 证据：**至少三个语言接地失败案例** | ⚠️ **数量需确认** | 冻结目录里 5 个案例中只有 2 个属"语言接地"（`attribute_confusion`、`distractor_priority`）。我另补了 3 个到 `branches/_shared/lang_grounding_failures/`，合计 **5 个语言接地案例** |

## C. 需要我补的（按优先级）

### C1【最高】生成式 AI 使用披露 + 团队贡献说明（A15/A16）

手册明确写「未披露的生成内容…可能按学术不端处理」。这是**硬性合规项**，不能缺。

### C2【高】硬件与配置记录（A2/A3）

需要把 CPU 型号、核数、内存、GPU、OS、驱动、`torch` 线程设置写进文档。

### C3【高】README 补"修改版"命令（A14）

手册要求 README 含「安装、基线、**修改版**、评估、作图命令」。目前缺修改版入口。

### C4【中】成功案例的逐步解释（A9）

手册要求「至少解释一例成功与一例失败」。失败做了 8 个，成功目前只有 gif。

### C5【中】语言接地失败案例的整理（B23）

把 5 个语言接地案例单独归档成一份说明，而不是散在 `reports/failures/`。

### C6【低】阶段归属的说明

目前的目录结构是：`reports/baseline_*`（冻结的基线）+ `reports/b3_*`（支线 3）+ `reports/b5_*`（数据效率最小补充）。
需要在 README 里说清哪个是**主修改**、哪个是**次指标补充**、哪个属于另一条支线，
否则评审可能误以为我把支线 5 也做了（而实际上只是补了手册要求的"数据效率"次指标）。
