# 支线 5 · 数据效率（Data Efficiency）

> 核心问题：**在 `BabyAI-GoToLocalS6N3-v0`（及零样本外推 `S7N4-v0`）离线模仿学习中，高样本复杂度究竟源于低样本下的梯度步数欠训练、弱表征对无关方向的过拟合，还是视觉-语言融合架构无法承载对称归纳偏置？仅用 `10%–20%` 的专家轨迹能否追平甚至超越 100%（4,500 ep）`Concat BC30`？**

---

## 0. 一句话结论

通过 **3,900 步梯度对齐** 剥离欠训练假象、经由对 `BabyAIBot` **实证零误差验证（0 mismatches）的精确状态-动作联合 $D_4 = C_4 \rtimes \mathbb{Z}_2$ 等变增广** 与 **8-路群平均 TTA** 消融，我们在 `5%–100%`（`225–4,500 ep`）全缩放律与 `20%`（`900 ep`）13 路受控消融（96 个新训练运行 + S6N3/S7N4 闭环评估 + 反事实语义接地探针 + 失败构成拆解）上揭示了 **$D_4$ 群等变偏置 × 跨模态架构的显著双重分离（Double Dissociation）**：
- **FiLM + D4 + TTA 实现 10× 样本压缩超越 100% `BC30`**：仅用 **10% 数据（450 ep ≈ 1,652 transitions）**，`film_d4` 达 S6N3 **`0.818 ± 0.008`** / S7N4 **`0.727 ± 0.053`**，`film_d4_tta` 达 S6N3 **`0.828 ± 0.008`** / S7N4 **`0.739 ± 0.044`**，已双双超越 **100% `BC30`（4,500 ep = 16,534 transitions：`0.814 / 0.697`）**；在 **20% 数据（900 ep ≈ 3,308 transitions）** 上，`film_d4_tta` 进一步升至 S6N3 **`0.870 ± 0.014`** / S7N4 **`0.798 ± 0.047`**，实现 **5× 样本压缩追平 100% `GRU-small`（`0.868 / 0.803`）与 `B3 Concat DAgger R1`（约 22k 标注：`0.867 / 0.747`）**。
- **`Concat` 与 $D_4$ 在大样本下发生负协同（坍塌 `-0.122`）**：虽然 `bc_d4` 在极低样本 `5%`（225 ep）比步数对齐的 `bc_step` 小幅提升 `+0.033`（`0.694 → 0.728`），但随着样本升至 `20%/50%/100%`，`bc_d4` 不仅停止增长，在 **100%（4,500 ep）反而暴跌至 `0.692 ± 0.004`**（比 `bc_step 0.814` 低 **`-0.122`**，3/3 seed 全负，S6 失败局从 `55.7` 反弹至 `92.3`）；反事实探针显示 `bc_d4@100%` 的综合指令敏感度降至 `0.014`（颜色敏感度 `0.012`，类型敏感度 `0.017`），证明 $C_4$ 旋转抹除了 `Concat` 末层依赖的全局空间方位捷径后，其晚期线性拼接结构无法建立真实的颜色-形状绑定，而 **FiLM 通道调制正是解锁 $D_4$ 样本效率的结构前提**。

---

## 1. 目标与成功判据

### 1.1 研究目标
1. **去除实验混杂因子**：原手册基线 `b5_defrac0.2/0.5`（及朴素子集抽样 `bc_naive`）以固定 epoch 数训练更少轨迹，导致 `5%` 仅更新 `210` 步而 `100%` 更新 `3,900` 步（`18.6×` 优化预算差）。必须先通过 **步数对齐（固定 `3,900` 步 = `30 epochs × 16,534 steps/epoch`）** 分离「优化欠训练」与「信息不足」。
2. **推导并实证验证 MiniGrid 第一视角精确对称群**：建立严格保持 `obs["image"]`（7×7 egocentric 栅格）、`obs["direction"]` 与 `BabyAIBot` 专家动作精确等变的 $D_4$ 群变换（训练期增广 + 测试期零额外训练 $D_4$-TTA 策略平均）。
3. **验证架构×对称归纳偏置交互效应**：系统对比 `Concat (BCNet)`、`GRU (B1)` 与 `FiLM (B2)` 在相同小样本与步数预算下的样本缩放曲线（`5%, 10%, 20%, 50%, 100%`）及 `20%` 受控消融（正则化 `bc_reg`、高权重衰减 `bc_wd`、仅 `Z2` 镜像、仅 `C4` 旋转、动作平衡采样 `bc_bal`、`gru_small_step`、`film_step`、`film_d4`、`d4_tta`）。
4. **反事实机械理解与跨支线全局统一对比**：用反事实指令置换探针与闭环失败分类器追踪不同规模下语言接地的涌现与坍塌阈值，并在「总监督 Oracle 标注转移数」统一坐标下绘制 B0–B5 全局 Pareto 前沿。

### 1.2 成功判据（实验启动前预设）
- 所有状态-动作 $D_4$ 群变换在 `BabyAIBot` 真实专家轨迹上的步进重建达到 **100% 精确一致（0 mismatches）**。
- 在严格不重选 checkpoint（仅用固定 500-ep 验证集 `val_action_acc` 早停）、不污染冻结基线及原手册 `reports/b5_data_efficiency_eval.json` / `reports/data_efficiency.csv` 的前提下：
  - 小样本（`≤20%`，900 ep）端到端超越 `100% BC30`（S6N3 `0.814` / S7N4 `0.697`）；
  - 量化并机制性解释「步数对齐、强正则、`GRU`、`D4` 增广、FiLM 融合、测试时 `D4-TTA`」各自独立净贡献；
  - 一键复现脚本 `SKIP_TRAIN=1 bash branches/b5_data_eff/run.sh` 端到端通过，且冻结区 `git diff` 为空。

---

## 2. 调研（实现之前）

### 2.1 文献
1. **BabyAI 样本效率基准**：Chevalier-Boisvert et al. (ICLR 2019, *BabyAI: A Platform to Study the Sample Efficiency of Grounded Language Learning*) 指出，在端到端栅格指令任务中，`GoToLocal` 及组合任务需要大量专家演示才能收敛，样本效率是核心瓶颈之一。
2. **MDP 状态-动作联合群等变**：van der Pol et al. (NeurIPS 2020, *MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning*) 严格证明，当策略与环境动力学存在离散群对称 $G$ 时，正确的等变形式不是孤立对图像做变换，而是满足状态变换 $L_g(s)$ 与动作诱导置换 $K_g^s(a)$ 同步成立的 **MDP Homomorphism**：$\pi(K_g^s(a) \mid L_g(s)) = \pi(a \mid s)$。
3. **强数据增广的 RL/IL 收益**：Laskin et al. (NeurIPS 2020, *RAD: Reinforcement Learning with Augmented Data*) 证明无需修改复杂主干的正确数据增广即可大幅压缩视觉策略学习的样本需求，同时警告打破任务几何不变性的增广（如错误翻转动作极性）会系统性破坏策略学习。

### 2.2 提炼的设计约束（MiniGrid `obs["image"]` + `direction` 精确 $D_4$ 群推导）
对 MiniGrid `BabyAI-GoToLocal` 离线转移 `(image[7,7,3], direction∈{0..3}, mission, action∈{0..6})` 实测检查显示：
1. **全局 $C_4$ 指南针旋转（旋转房间 90° × $k$, $k \in \{0,1,2,3\}$）**：
   - 由于 `obs["image"]` 是 **以智能体朝向为正上方、相对智能体局域坐标系的 egocentric 7×7 视野**，全局绕房间中心旋转 90° 时，智能体面前的相对栅格、颜色、物体类型与下一步局部动作（`left/right/forward`）**均保持完全不变**！唯一改变的是绝对朝向信号 `direction ← (direction + k) % 4`。
   - 此外检查训练集中 `image[:, :, 2]`（物体状态通道）在 `GoToLocal` 中 **全为 0**（无锁门状态编码），不会引入额外方位伪差。
2. **全局水平镜像 $\mathbb{Z}_2$**：
   - 将局部 egocentric 视野沿宽度轴镜像 `image ← image[:, ::-1, :]` 时，智能体左前方物体变到右前方、右前方变到左前方，**因此专家动作必须同步置换 `left(0) ↔ right(1)`**，而 `forward(2)` 与其他动作不变；为保证与 $C_4$ 生成元闭合构成完整的 8 元二面体群 $D_4 = C_4 \rtimes \mathbb{Z}_2$，`direction` 在 8 群成员 `(flip ∈ {0,1}, rot ∈ {0..3})` 下覆盖镜像后的全局朝向映射。
3. **精确实证检验**：直接在 `data_eff_utils.py` 中编写 `verify_d4_against_bot(20)`，将真实 `BabyAI-GoToLocalS6N3-v0` 环境底层网格（`grid`, `agent_pos`, `agent_dir`）物理旋转/镜像后重采样 `obs` 并重新询问 `BabyAIBot` 专家动作，对比变换函数输出：**`verify_d4_against_bot(20)` 与前序 `50` 局检验均实现 0 误差通过**。

### 2.3 已知陷阱
- **步数未对齐陷阱**：`n_epochs=30` 下若训练集从 4,500 ep 降至 225 ep（5%），优化步数骤降 `18.6×`（`3,900 → 210` steps），Naive 子集评估几乎完全测成「优化未收敛」而非样本容量不足。
- **验证集规模漂移陷阱**：`5%` 若顺带按比例缩小验证集会导致早停噪声剧增。本支线强制复用全集固定 `500` 条未见验证轨迹（seed 4500 之前的 `1,844` transitions），仅对训练 `4,500` 轨迹做确定性抽样。
- **动作极性破坏陷阱**：图像左右翻转若忘记同步 `left(0) ↔ right(1)`，会让 BC 直接崩溃到 50% 随机转向；而在测试期 `D4-TTA` 时同样必须对翻转分支输出的动作概率执行 `P(left) ↔ P(right)` 逆置换后才能在群上做对数概率均值。

---

## 3. 实现（含关键决策与理由）

所有支线代码独立置于 `branches/b5_data_eff/`，零修改冻结区及兄弟支线目录，模型输出写入独立前缀 `runs/b5_<method>_<ftag>_seed{0,1,2}`。

### 3.1 代码架构（`branches/b5_data_eff/`）
- [`data_eff_utils.py`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/branches/b5_data_eff/data_eff_utils.py)：
  - `load_split_episodes`（确定性切分 4,500 train / 500 固定 val，调用 `common.assert_no_leak` 硬防泄漏）；
  - `transform_obs_and_action` / `inverse_permute_logits` 实现 exact $D_4 = C_4 \rtimes \mathbb{Z}_2$（含 `C4` 旋转子群与 `Z2` 镜像子群接口及 `verify_d4_against_bot` 自检）；
  - `build_training_table` 支持 `none / z2 / c4 / d4` 扩展及 `balance_actions` 重平衡采样；
  - `sample_epoch_indices` 确保所有步数对齐方法每轮精确采样 `N_epoch = 16,534` 条（`30 epochs = 3,900 steps`）。
- [`train_data_eff.py`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/branches/b5_data_eff/train_data_eff.py)：
  - 统一实现 11 种受控训练配方（`bc_naive`, `bc_step`, `bc_wd`, `bc_reg`, `bc_bal`, `bc_z2`, `bc_c4`, `bc_d4`, `gru_small_step`, `film_step`, `film_d4`）；
  - 单线程 `torch.set_num_threads(1)` 可复现训练，仅按固定 500-ep 验证集 `val_action_acc` 保存 `best_model.pt`，同时记录 `history.json`、`train_acc_at_best` 与 `generalization_gap`。
- [`data_eff_policy.py`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/branches/b5_data_eff/data_eff_policy.py)：
  - 加载 Concat / GRU / FiLM 检查点；可选 `d4_tta=True` 对单步观测并行构造 8 个 $D_4$ 群输入，批处理前向后经逆动作置换 $\rho_a(g)^{-1}$ 平均 log-softmax 概率再取 `argmax`（记为 `bc_d4_tta` / `film_d4_tta`，无需任何额外训练标注）。
- [`eval_data_eff.py`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/branches/b5_data_eff/eval_data_eff.py)：对全部 96 个新训练模型 + 6 个原手册 `b5_defrac{0.2,0.5}` + 6 个扩展区 TTA 共 `129` 组（S6N3 `4500:4800` 及 S7N4 `10000:10300` 各 300 局）并行闭环评估，自动汇总扩展曲线、`20%` 受控消融、配对 seed 增量与采样压缩里程碑。
- [`probe_data_eff_grounding.py`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/branches/b5_data_eff/probe_data_eff_grounding.py)：在 600 个验证观测筛选出的 `183` 个真实多物体可切换目标状态（`GoToLocal` 中存在 ≥2 种不同目标候选）上，反事实替换 `mission` 颜色词与物体类别词，定量测度策略条件动作翻转率（颜色敏感度 / 类型敏感度 / 综合敏感度）。
- [`compare_failures.py`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/branches/b5_data_eff/compare_failures.py)：调用冻结基线诊断分类器将 S6N3 失败局拆解为 `表征失败 (attribute_confusion + distractor_priority + approach_then_lost)` 与 `状态覆盖失败 (turn_oscillation + orientation_lock + forward_stall + wander_timeout)` 两大组 7 细分项。
- [`make_summary.py`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/branches/b5_data_eff/make_summary.py) 与 [`make_figures.py`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/branches/b5_data_eff/make_figures.py)：全自动生成 [`reports/b5_data_eff_summary.md`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/reports/b5_data_eff_summary.md) 与 [`reports/figures/b5_data_eff/`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/reports/figures/b5_data_eff/) 四张论文级图表。

### 3.2 实验矩阵（共 96 个新训练运行 × 3 seeds = 0,1,2）
- **5 档缩放律曲线**（`5%=225 ep ≈ 820 trans`, `10%=450 ep ≈ 1,652`, `20%=900 ep ≈ 3,308`, `50%=2,250 ep ≈ 8,278`, `100%=4,500 ep = 16,534`）：
  - `bc_naive`, `bc_step`, `bc_reg`, `bc_d4`, `film_step`, `film_d4` 全 5 档（注：`bc_naive` / `bc_step` / `film_step` 在 100% 复用共享池对应训练，`bc_reg` / `bc_d4` / `film_d4` 跑全 5 档）；
  - zero-training TTA 评估 `bc_d4_tta`, `film_d4_tta` 在 `5%, 10%, 20%` 低数据区评测。
- **`20%`（`900 ep`）机制精细消融**：追加 `bc_z2`（仅镜像 2×）、`bc_c4`（仅旋转 4×）、`bc_wd`（强权重衰减 `wd=1e-2`）、`bc_bal`（动作均衡采样）、`gru_small_step`（B1 最佳语言编码器在小样本下表现），共 14 路对照（含 `b5_defrac`）。

---

## 4. 结果（全数来自 `reports/b5_data_eff_{eval,s7n4,sensitivity,failures}.json`，每组 `n=3`）

### 4.1 样本缩放律主曲线：S6N3（已见域）与 S7N4（未见外推）闭环成功率（`mean ± SD`）

| 方法 | 说明 | 5% (225 ep) | 10% (450 ep) | 20% (900 ep) | 50% (2250 ep) | 100% (4500 ep) |
|---|---|---:|---:|---:|---:|---:|
| `bc_naive` | Concat 无步数匹配 | 0.614±0.030 (S7: 0.448) | 0.684±0.033 (0.507) | 0.713±0.021 (0.630) | 0.717±0.021 (0.652) | 0.814±0.045 (0.697) |
| `bc_step` | Concat 3900 步匹配 | 0.694±0.035 (0.524) | 0.722±0.027 (0.564) | 0.742±0.026 (0.627) | 0.751±0.023 (0.634) | 0.814±0.045 (0.697) |
| `bc_reg` | Drop0.2+LS0.05+Cos | 0.619±0.061 (0.439) | 0.719±0.030 (0.514) | 0.718±0.015 (0.607) | 0.790±0.027 (0.632) | 0.733±0.031 (0.688) |
| `bc_d4` | Concat + D4 8×增广 | 0.728±0.067 (0.541) | 0.734±0.033 (0.558) | 0.714±0.025 (0.598) | 0.723±0.035 (0.662) | **0.692±0.004** (0.691) |
| `bc_d4_tta` | Concat + D4 + TTA | 0.710±0.069 (0.547) | 0.737±0.024 (0.566) | 0.729±0.021 (0.600) | — | — |
| `film_step` | FiLM 步数匹配 (无增广) | 0.696±0.034 (0.552) | 0.733±0.022 (0.633) | 0.814±0.023 (0.750) | **0.929±0.018 (0.873)** | **0.959±0.006 (0.900)** |
| `film_d4` | FiLM + D4 8×增广 | **0.736±0.018 (0.599)** | **0.818±0.008 (0.727)** | **0.863±0.014 (0.781)** | 0.927±0.022 (0.862) | 0.953±0.005 (0.894) |
| `film_d4_tta` | FiLM + D4 + 8× TTA | **0.740±0.016 (0.624)** | **0.828±0.008 (0.739)** | **0.870±0.014 (0.798)** | — | — |

- 跨缩放档 `mean_return / median_len_success / timeouts` 摘要（S6N3 / S7N4）：
  - `bc_naive@5%`：`val_acc=0.7798`，S6 `0.614 / ret=0.543 / len=4.0 / to=115.7`；S7 `0.448 / ret=0.405 / len=4.0 / to=165.7`；
  - `film_d4_tta@10%`：`val_acc=0.8729`，S6 `0.828 / ret=0.739 / len=3.7 / to=51.7`；S7 `0.739 / ret=0.665 / len=4.3 / to=78.3`（全部关键指标均已超过 `100% BC30`：`S6 0.814 / ret=0.723 / len=4.0 / to=55.7`，`S7 0.697 / ret=0.629 / len=4.7 / to=91.0`）；
  - `film_d4_tta@20%`：`val_acc=0.9167`，S6 `0.870 / ret=0.784 / len=3.0 / to=39.0`；S7 `0.798 / ret=0.727 / len=4.0 / to=60.7`；
  - `bc_d4@100%`：`val_acc=0.8219`，S6 `0.692 / ret=0.608 / len=4.0 / to=92.3`；S7 `0.691 / ret=0.614 / len=5.0 / to=92.7`。

### 4.2 `20%` 极低样本（`900 ep`）14 路受控消融完整指标

| 方法 | `val_acc` | `train_acc` | `gen_gap` | S6N3 Success (`ret / len / to`) | S7N4 Success (`ret / len / to`) | 颜色 / 类型敏感度 | S6 失败 (表征+覆盖) |
|---|---:|---:|---:|---|---|---:|---|
| `film_d4_tta` | **0.9167** | 0.9972 | +0.0806 | **0.870±0.014** (`0.784 / 3.0 / 39.0`) | **0.798±0.047** (`0.727 / 4.0 / 60.7`) | **0.1063 / 0.1638** | **39.0 (26.0+13.0)** |
| `film_d4` | **0.9167** | 0.9972 | +0.0806 | **0.863±0.014** (`0.778 / 3.0 / 41.0`) | **0.781±0.055** (`0.712 / 4.0 / 65.7`) | **0.1063 / 0.1921** | **41.0 (26.0+15.0)** |
| `film_step` | 0.8744 | 0.9992 | +0.1248 | 0.814±0.023 (`0.723 / 4.0 / 55.7`) | 0.750±0.052 (`0.672 / 5.0 / 75.0`) | 0.0725 / 0.1638 | 55.7 (37.7+18.0) |
| `bc_step` | 0.8068 | 0.9081 | +0.1013 | 0.742±0.026 (`0.644 / 4.7 / 77.3`) | 0.627±0.021 (`0.555 / 5.0 / 112.0`) | 0.0145 / 0.0339 | 77.3 (43.3+34.0) |
| `bc_wd` | 0.8102 | 0.8992 | +0.0890 | 0.741±0.040 (`0.642 / 4.7 / 77.7`) | 0.612±0.016 (`0.548 / 4.7 / 116.3`) | 0.0097 / 0.0169 | 77.7 (41.3+36.3) |
| `b5_defrac` | 0.8095 | 0.8649 | +0.0554 | 0.732±0.007 (`0.640 / 4.5 / 80.3`) | 0.622±0.014 (`0.556 / 5.0 / 113.3`) | — | — |
| `bc_c4` | 0.8068 | 0.8559 | +0.0491 | 0.730±0.019 (`0.635 / 4.3 / 81.0`) | 0.584±0.044 (`0.523 / 4.8 / 124.7`) | 0.0241 / 0.0339 | 81.0 (43.3+37.7) |
| `gru_small_step` | 0.8134 | 0.9991 | **+0.1856** | 0.730±0.011 (`0.641 / 4.0 / 81.0`) | 0.613±0.023 (`0.547 / 5.0 / 116.0`) | 0.0531 / 0.0960 | 81.0 (43.7+37.3) |
| `bc_d4_tta` | 0.8151 | 0.8929 | +0.0778 | 0.729±0.021 (`0.638 / 4.7 / 81.3`) | 0.600±0.067 (`0.538 / 4.3 / 120.0`) | 0.0072 / 0.0508 | 81.3 (46.7+34.7) |
| `bc_z2` | 0.8131 | 0.9006 | +0.0875 | 0.722±0.010 (`0.630 / 4.0 / 83.3`) | 0.603±0.011 (`0.541 / 5.0 / 119.0`) | 0.0266 / 0.0452 | 83.3 (48.0+35.3) |
| `bc_bal` | 0.8024 | 0.9008 | +0.0983 | 0.718±0.007 (`0.626 / 4.3 / 84.7`) | 0.600±0.021 (`0.536 / 5.0 / 120.0`) | 0.0145 / 0.0169 | 84.7 (48.0+36.7) |
| `bc_reg` | 0.8084 | 0.8901 | +0.0817 | 0.718±0.015 (`0.627 / 4.3 / 84.7`) | 0.607±0.028 (`0.542 / 5.0 / 118.0`) | 0.0169 / 0.0056 | 84.7 (49.3+35.3) |
| `bc_d4` | 0.8151 | 0.8929 | +0.0778 | 0.714±0.025 (`0.627 / 4.3 / 85.7`) | 0.598±0.060 (`0.535 / 4.7 / 120.7`) | 0.0242 / 0.0452 | 85.7 (46.7+39.0) |
| `bc_naive` | 0.8131 | 0.8678 | +0.0547 | 0.713±0.021 (`0.623 / 4.3 / 86.0`) | 0.630±0.016 (`0.562 / 5.0 / 111.0`) | 0.0217 / 0.0169 | 86.0 (46.3+39.7) |

---

## 5. 结果分析：哪里符合预期、哪里不符合

### 5.1 符合预期的发现
1. **梯度步数混杂被完美分离**：在 5%（225 ep）上，`bc_step − bc_naive = +0.080`（3/3 seed 全正：`0.614 → 0.694`，S7N4 `0.448 → 0.524`），在 10%/20%/50% 上也持续贡献 `+0.029 ~ +0.038`。说明原小样本研究如果不对齐优化步数，至少三分之一的性能差距纯粹是没训够步数。
2. **FiLM + D4 + TTA 形成高回报闭环协同**：在 FiLM 架构上，D4 群增广在低数据区 3/3 一致大幅增益：`5% +0.040`（`0.696 → 0.736`）、`10% +0.084`（`0.733 → 0.818`）、`20% +0.049`（`0.814 → 0.863`）；叠加零额外训练的 `D4-TTA` 后再稳步增加 S6N3 `+0.004 ~ +0.010` 及 **S7N4 `+0.012 ~ +0.025`**（如 `5%` S7N4 `0.599 → 0.624`，`20%` `0.781 → 0.798`），最终在 **450 ep（10%）超越 100% BC30**。
3. **语义敏感度随数据规模呈现双相涌现，且 FiLM+D4 大幅提前涌现点**：
   - `film_d4@10%`（450 ep）综合反事实敏感度已达 **`0.103`**（颜色 `0.077`，类型 `0.164`），直接持平 **`bc_step@100%`（4,500 ep）的 `0.103`**；`film_d4@20%` 升至 `0.132`、`50%` 达 `0.212`、`100%` 达 `0.266`。
   - 失败机制拆解显示，对比 `bc_naive@20%` (`86.0 = 46.3 表征 + 39.7 状态覆盖`)，`film_d4_tta@20%` 将总失败压减超过一半至 **`39.0 (26.0 表征 + 13.0 状态覆盖)`**，且中位成功步长从 `4.3` 缩短到最优 **`3.0`** 步。
4. **`gru_small_step@20%` 严重语言过拟合被精准捕获**：在 100% 数据下 `gru_small` 是 B1 表现最好的语言编码器（`0.868`），但在 `20%`（900 ep）小样本上，其 `train_acc` 在 15 轮内飙至 `0.9991` 而 `val_acc` 停在 `0.8134`（全消融最大泛化间隙 **`+0.1856`**），闭环 S6N3 跌回 **`0.730`**（反不如简单 `bc_step 0.742`），证实高自由度序列编码器在不结合 FiLM 空间条件调制时在小样本下极易靠背诵指令文本过拟合。

### 5.2 反直觉 / 不符合朴素预期的重大发现
**为什么 `bc_d4`（Concat + D4）在 5% 有正收益（`+0.033`），却在 `20%/50%` 停滞并在 `100%` 灾难性坍塌至 `0.692`（`-0.122 vs bc_step`）？**
我们结合 `history.json`、反事实探针与 `D4` 群子群消融（`bc_z2 / bc_c4`）定位出双重因果机制：
1. **结构性失配 + 空间方向捷径被 $C_4$ 破坏**：在 `bc_step`（无增广）上，随着数据升至 100%，`Concat` 的 `type_sensitivity` 能缓慢爬升到 `0.226`（综合 `0.103`），部分依赖原始房间绝对朝向 `direction` 与物体局部布局的相关捷径达成 `0.814`；一旦施加包含 `C4` 旋转的 `D4` 增广，`direction` 被均匀映射到 `{0,1,2,3}`，这些伪相关被彻底抹除，`bc_d4@100%` 的 `type_sensitivity` 从 `0.226` 骤降至 **`0.017`**、`color_sensitivity` 降至 **`0.012`**，暴露出末层线性拼接 `Linear(1568+32+4→128)` 根本无法学会真实「指令通道筛选目标视觉物体」的绑定关系。
2. **固定步数预算下的增广样本稀释效应**：在 `100%` 数据下，`bc_d4` 将训练集扩充为 `8 × 16,534 = 132,272` 条虚拟样本，但步数对齐机制仍限制总更新为 `30 epochs × 16,534 = 3,900 steps`（即全库仅被遍历 `3.75` 次）。低归纳偏置的 `Concat` 与高学习自由度在 `3.75` 遍 CosineLR 周期内尚未消化完增广方差（`bc_d4@100%` 的 `val_acc` 仅 `0.8219` vs `bc_step` 的 `0.8733`），同样现象也出现在强正则 `bc_reg@100%`（`0.733` vs `0.814`）；而 `FiLM` 将条件调制前置到 `conv1` 后通过低自由度 `(1+Δγ)·x + Δβ` 直接对特征通道执行选择滤波，因此在相同 `3,900` 步内既利用了 `D4` 对称性压低过拟合间隙（`20%`：`film_step +0.125 → film_d4 +0.081`），又避免在 `100%` 下受损（`film_d4 0.953` vs `film_step 0.959` 统计持平，S6 均稳居 `>0.95`）。

---

## 6. 补充调研（基于实测现象）

针对上述「同样一组完全精确的 $D_4$ 对称增广，为何在 `FiLM` 上形成 10× 压缩，却在 `Concat` 与低样本 `GRU` 上失效甚至负迁移」的反直觉实测现象，补充检索到以下关键理论工作：
1. **Lyle, Kwiatkowska & Gal (2019; 扩展版 Lyle et al., JMLR 2020, *An Analysis of the Effect of Invariance on Generalization in Neural Networks*)**：从 PAC-Bayes 泛化界与梯度方差角度比较「模型内建对称（Feature Averaging / Equivariant Layers）」与「训练期数据增广（Data Augmentation）」——前者严格压缩有效假设空间维度并降低方差而不增加优化样本负担，而后者在模型未具备对应不变架构偏置时，会将群轨道方差全部推给有限步长的 SGD 去逐样拟合，若优化预算或表征结构匹配不足，增广反而导致欠拟合。这一理论精确解释了为何 `bc_d4@100%`（`3.75` 遍增广表，`val_acc=0.822`）落后于 `bc_step`（`30` 遍原始表，`val_acc=0.873`），以及为何测试期直接做群轨道整合的 `film_d4_tta` 在低样本与跨房间 `S7N4` 上无需再训练即给出稳定 `+0.010 ~ +0.025` 增益。
2. **Balestriero, Bottou & LeCun (NeurIPS 2022, *The Effects of Regularization and Data Augmentation are Class Dependent*)**：证明数据增广与全局强正则（权重衰减、Dropout、Label Smoothing）对模型能力有限或结构错配的架构并非均匀有益，而是可能优先摧毁依赖特定方向/边缘分布的精细决策边界。这与我们在 `20%` 下观察到 `bc_reg`（`-0.024`）、`bc_c4`（`-0.012`）与 `100%` 下 `bc_reg`（`-0.081`）、`bc_d4`（`-0.122`）的系统性退化高度吻合：正则与增广清除了 `Concat` 赖以为生的绝对朝向快捷方式，却没有提供 FiLM 那样的乘性特征门控来替代学习语义目标选择。

---

## 7. 改进方向与下一轮计划

1. **架构内建真等变感知骨干（替代纯粹外部增广）**：参照 van der Pol et al. (NeurIPS 2020) 与群等变卷积网络（Cohen & Welling, ICML 2016），若将 `Z2` 镜像约束直接写入共享权重 `C1/C2` 并行分支并与 `FiLM` 结合，有望在不付出 8× 训练表扩展与 TTA 8× 推理延迟的前提下保留全部对称性收益。
2. **「10%–20% `film_d4_tta` + B4 多尺度数据 + B3 1轮轻量 FiLM DAgger」组合路线**：当前 `film_d4_tta` 用 `20%` 数据（`3,308` 离线标注）已达 `0.870 / 0.798`；后续若只需额外查询 `~1,500–2,000` 条聚焦 `turn_oscillation` 与 `attribute_confusion`（占当前 `film_d4_tta@20%` 剩余 `39` 失败局的 `~62%`）的在线纠错标注（总量控制在 `<5,500` transitions，仍不到完整基准 `16,534` 的 1/3），预期可直接跨越 `S6N3 > 0.94`、`S7N4 > 0.88`。

---

## 8. 复现命令

全部命令于 `/home/zhangyuming/course_project/Embodied_AI/p3-babyai` 目录下通过 `env_embodied` 环境执行（单线程可复现）：

```bash
# 1) 冻结区自检 + 重跑全套 S6N3/S7N4 闭环评估、语义探针、失败拆解、报告与四图（约 45 秒）
SKIP_TRAIN=1 bash branches/b5_data_eff/run.sh

# 2) 从零全流程复现（重新训练全部 96 个 B5 运行 + 评估 + 探针 + 图表）
bash branches/b5_data_eff/run.sh

# 3) 单独验证状态-动作 D4 群转换与 BabyAIBot 的零误差一致性
source env_embodied/bin/activate
python branches/b5_data_eff/data_eff_utils.py

# 4) 单独训练、评估或出图示例
OMP_NUM_THREADS=1 python branches/b5_data_eff/train_data_eff.py --method film_d4 --frac 0.20 --seed 0 --epochs 30 --threads 1 --out runs/b5_film_d4_f020_seed0
OMP_NUM_THREADS=1 python branches/b5_data_eff/eval_data_eff.py --env-id BabyAI-GoToLocalS6N3-v0 --seeds test --workers 16 --out reports/b5_data_eff_eval.json
OMP_NUM_THREADS=1 python branches/b5_data_eff/make_summary.py
OMP_NUM_THREADS=1 python branches/b5_data_eff/make_figures.py
```

产出图表位于 [`reports/figures/b5_data_eff/`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/reports/figures/b5_data_eff/)：
- [`b5_fig1_scaling_curves_s6_s7.png`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/reports/figures/b5_data_eff/b5_fig1_scaling_curves_s6_s7.png)（图 1｜5%–100% 样本缩放律：S6N3 与 S7N4 双重分离）
- [`b5_fig2_step_overfit_dynamics.png`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/reports/figures/b5_data_eff/b5_fig2_step_overfit_dynamics.png)（图 2｜20% 数据 30-Epoch 训练过拟合动态与 14 路消融对比）
- [`b5_fig3_grounding_emergence.png`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/reports/figures/b5_data_eff/b5_fig3_grounding_emergence.png)（图 3｜反事实指令敏感度涌现与 S6N3 失败构成拆解）
- [`b5_fig4_cross_branch_pareto.png`](file:///home/zhangyuming/course_project/Embodied_AI/p3-babyai/reports/figures/b5_data_eff/b5_fig4_cross_branch_pareto.png)（图 4｜跨支线 B0–B5 全局效率统一坐标 Pareto 前沿）

---

## 9. 与其它支线的关系（全局交叉验证）

结合前四条支线的实测结果与图 4 全局 Pareto 曲线，五条研究支线共同构成对 `BabyAI-GoToLocal` 离线到交互式策略学习的完整立体刻画：

1. **与 `B1（语言编码器）` 的互补与修正**：B1 在 100% 数据下发现 `gru_small` 将 `Concat BC30` 推升至 `0.868 / 0.803`；但 B5 在 `20%`（900 ep）下测得 `gru_small_step` 遭遇 `+0.1856` 的最大过拟合间隙且闭环仅 `0.730`（不及简单均值池化的 `bc_step 0.742`）。**修正结论：`GRU` 语言编码器的优势具有「样本阈值条件」**，在缺乏 FiLM 跨模态绑定约束时，提升语言端容量只会加快小样本死记硬背。
2. **与 `B2（FiLM 融合）` 的强因果闭环**：B2 证明在 100% 数据下 `film_nwd` 实现 `0.959 / 0.900` 跃升；B5 的交叉消融进一步确立——**FiLM 并非仅仅在满数据下刷高上限的「锦上添花」，而是承接 $D_4$ 群对称归纳偏置的结构前提**：没有 FiLM，`D4` 在 Concat 上于 100% 坍塌 `-0.122`；拥有 FiLM，`D4 + TTA` 让 `10%` 样本即可击败 `100% BC30`。
3. **与 `B3（DAgger 交互纠错）` 的效率接力**：在 B3 中，单纯对 `Concat` 执行两轮高开销在线 DAgger 需消耗 **`32,668` 总 Oracle 标注（`b3_v2_r2_w1`）** 才达到 S6N3 `0.909` / S7N4 `0.828`；而 B5 的 `film_d4_tta@20%` 在 **纯离线、`3,308` 标注**（不到前者的 **1/9**）上已取得 S6N3 `0.870` / S7N4 `0.798`，`film_step@50%`（`8,278` 标注）已达 `0.929 / 0.873` **全面超越所有 `Concat + DAgger` 版本**。B5 填平了 `<10,000` 标注区的全球效率前沿，而在 `>16,534` 标注上限区与 `B3` 最佳在线模型 **`b3_v2_film_dag_w1`（`22,539` 标注：S6N3 `0.969` / S7N4 `0.918`）** 无缝衔接。
4. **与 `B4（课程学习 / 多尺度迁移）` 的正交叠加空间**：B4 通过 `pace_easy`（`16,534` 标注：S6 `0.857` / S7 `0.766`）和 `film_multiscale_prog`（`17,156` 标注：**S7N4 全项目最高 `0.924`**）解决跨地图尺度的视觉不变性；B5 则在固定单域极低预算下通过 $D_4$ 群等变与 TTA 把样本需求压缩 `5×–10×`。图 4 右图表明，**B5（极低样本群等变前置先验）+ B2（FiLM 通道绑定）+ B4（跨尺度空间课程）+ B3（高收益纠错 DAgger）** 共同构成了完整、可无缝组合的 Pareto 最优技术栈。
