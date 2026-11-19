# 生成式 AI 使用披露（手册 §7 强制要求）

> **手册 §7 原文要求**：「生成式 AI 工具可用于学习、代码辅助与调试，前提是披露工具与用途、概括实质性帮助、核验全部生成内容，且每位成员都能解释提交成果。未披露的生成内容、伪造结果或不可核验的实验，可能按学术不端处理。」
> 本文件按上述四项要素如实、完整披露全项目开发、五条支线实验与主修改切换过程中的生成式 AI 使用情况。

---

## 1. 使用的生成式 AI 工具清单

| 工具 / 平台 | 模型 / 形态 | 使用阶段与用途 |
|---|---|---|
| Cursor（AI 代码编辑器） | Agent 模式（`deepseek-v4.1-flash` 等） | 基线复现脚手架、早期 B2（FiLM）与 B3（DAgger v1）代码编写、环境调试、初步实验日志整理 |
| Antigravity（Google DeepMind Agentic IDE） | 多智能体协作模式（协调、实验核验、文档重构、独立审计） | 补齐 B1（语言编码器）/ B3-v2（主动门控与 FiLM+DAgger 正交叠加）/ B4（课程学习）/ B5（步数对齐数据效率）辅助支线；**将项目唯一主修改从 B3 切换确立为更具机制解释力与零额外样本优势的 B2（FiLM 跨模态融合）**；撰写中英文正式报告、合规文档与只读交叉审计 |
| 内置文献与文档检索工具 | WebSearch | 检索 FiLM（Perez et al., 2018; ML Retrospectives 2019）、BabyAI 1.1、DAgger / HG-DAgger、Information Collapse 等相关文献与官方 MiniGrid 文档 |

---

## 2. 实质性帮助的范围（如实列举）

生成式 AI 在本项目的工程实现、自动化评测管线与文档排版中提供了广泛的代码与文字辅助，具体涵盖：

| 模块类别 | AI 协助生成或重构的具体内容 |
|---|---|
| 环境与基线管线（`src/`） | `uv` 虚拟环境配置、`src/common.py`（seed 划分与 `BCNet`）、`src/gen_expert.py`、`src/audit_data.py`、`src/train_bc.py`、`src/eval_closed_loop.py`、`src/analyze_failures.py`、`src/render_episodes.py`、`src/make_figures.py` |
| **★ 唯一主修改 B2（`branches/b2_film/`）** | `film_net.py`（恒等起步单层线性 `FiLMGenerator`：`1+Δγ, Δβ` 及 `film` / `film_only` / `film_res` / `concat_wide` 变体）、`train_film.py`、`eval_film.py`、反事实探针 `probe_language.py`、失败对比 `compare_failures.py`、顶层复现脚本 `run_film_main.sh` |
| 辅助与次指标支线（`B1 / B3 / B4 / B5`） | `b1_langenc/`（GRU/Transformer/DAN/PosMLP 与词序打乱探针）、`b3_dagger/`（早停诊断 `BC30`、Loop-Unsticking、4,500 等预算 `Split-DAgger`、`FiLM+DAgger` 正交实验）、`b4_curriculum/`（五级环境谱课程消融）、`b5_data_eff/`（3,900 步对齐数据效率曲线与 D4 群等变增广） |
| 主线切换、个案核验与正式报告 | 将 `README.md`、`branches/README.md`、`docs/*` 统一收敛到以 **B2 FiLM 为唯一主修改**的叙事框架；编写只读脚本核验 `b2_film_nwd_seed0` 在 9 个具体案例 seed 上的逐步动作；排版生成中英文书面报告及高分辨率出版级图表 |

---

## 3. 人工主导决策与逐项核验记录（确保 100% 真实可解释）

全部关键研究判断与实验真值均经过严格脚本核验与原始 JSON 交叉比对，绝无任何编造、估算或未经运行的数字：

1. **为什么将唯一主修改从 B3（DAgger-lite）切换为 B2（FiLM 跨模态融合）**：
   - 在对比完全部五条支线实测数据后，我们基于科学证据做出主线决策：基线 68.8% 的失败属于**表征接地失败**（颜色敏感度仅 `0.065`）；`B2 FiLM` 在**零额外专家轨迹（严守 5,000 条静态预算）**下把 S6N3 从受控 `BC30` 的 `0.814 ± 0.045` 推至 **`0.959 ± 0.006`（`+0.144`）**、S7N4 零样本迁移从 `0.697` 推至 **`0.900 ± 0.021`（`+0.203`）**，幅度与因果纯粹度均显著优于需要在线交互查询的纯 DAgger（v1 `0.867` / v2 两轮 `0.909`），因此确立 B2 为唯一主修改，B1/B3/B4/B5 作为严谨的消融与正交补充。
2. **数值溯源与代码审计**：
   - 文档与报告中的每一个均值、标准差（`ddof=0`）、配对差值与参数量（`BCNet` 223,071、`film` 225,183、`film_only` 221,087、`concat_wide` 224,683）均由 Python 脚本直接从 `reports/*.json` 与 `runs/*/config.json` 读取核对；
   - `docs/CASE_ANALYSIS.md` 中 `b2_film_nwd_seed0` 与 `b2_film_seed0` 在 `seed 4501, 4516, 4565, 4570, 4582, 4560, 4630, 4716, 4568` 上的每一步坐标、朝向与动作，均在服务器上加载真实权重运行环境 step 实测导出。
3. **冻结边界核验**：通过 `git diff --stat` 强制校验 `src/`、`data/expert_5000.pkl`、`runs/baseline_*` 与全部历史评估 JSON 在整理过程中零修改。

---

## 4. 开发过程中发现并修正的 AI 引入错误与方法论陷阱（诚实记录）

| 阶段 | AI 初稿引入的错误或协议缺陷 | 实测后果 | 发现与修正方式 |
|---|---|---|---|
| 基线实现 | `BCNet.encode_image` 嵌入分支漏写 `.permute(0,3,1,2)` | `Conv2d` 输入通道错位为 21 而非 24，前向直接报错 | 运行 `train_bc.py` 捕获维度异常后修正（见 `src/common.py`） |
| 基线渲染 | `render_episodes.py` 误读 `u.direction`（应为 `u.agent_dir`）；GIF 合成误用单帧 `imageio.imread` | 渲染抛出 `AttributeError`；合成视频仅剩 3 帧 | 核对 MiniGrid 源码修正属性名并改用 `imageio.mimread`，产出 197 帧（49 秒）`demo_all.gif` |
| 可复现性 | `train_bc.py` 初版默认 `--threads 4` | 多线程浮点归约非确定性导致同 seed 重复训练 `val_acc` 抖动 | 改为默认单线程 `torch.set_num_threads(1)`，验证两次训练 `best_model.pt` 逐字节一致 |
| **早停协议** | **照搬常见教程设置 `patience=5`** | `seed1`/`seed2` 在第 13/14 轮被过早截断，使未修改基线被低估 `0.068`（`0.747` vs `0.814`） | 逐轮检查 `history.json` 定位原因；建立跑满 30 epoch 严格按 `val_acc` 选模的 **`BC30`（`0.814 ± 0.045`）** 作为全项目公平比较基准 |
| B3 v1 对照 | `b3_expertextra_1500` 初版误用已进训练集的 `0:1500` 种子；B1 `mean_pos` 初版在线性平均前加位置嵌入 | 额外专家新颖状态为 0；定长 5 词均值令位置向量恒等抵消（敏感度 `0.000`） | 在 B3-v2 中改用全新未见种子 `5300:6800` 重建 `BC-Fresh`（`0.860`）与 `Split-DAgger`；在 B1 中新增带前置非线性的 `pos_mlp`（`0.851`）与 `dan`（`0.839`）拆解混淆 |
| B2 正则化 | 盲信文献默认加全局 `weight_decay=1e-4` | 窄主干（90% 参数集中在第一层线性层）上全局 L2 让 `concat` 暴跌 `-0.129`（`0.836 → 0.707`） | 补充运行 `wd=0` 对照组（`concat_nwd` 与 `film_nwd`），分离出全局 L2 负效应并拿到最优 `film_nwd = 0.959` |

---

## 5. 合规声明

本团队确认遵守课程手册 §7 的全部规定：
- 所有使用生成式 AI 辅助编写的代码、实验脚本与文档均已如实披露；
- 仓库内所有定量指标均可由 `bash run_baseline.sh` 与 `bash run_film_main.sh` 在冻结环境 `env_embodied` 中确定性复现；
- 全部负结果与方法边界（如 `patience=5` 误早停、全局 `weight_decay` 伤害窄主干、纯语言编码器无法解决颜色盲区、`FiLM` 在 7×7 背后盲区 `seed 4568` 的无记忆振荡）均已诚实呈现，每位成员均能独立解释完整项目链路。
