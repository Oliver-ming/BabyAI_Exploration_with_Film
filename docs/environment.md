# 环境、硬件与运行溯源记录（手册 §2.1 要求）

> 手册 §2.1 要求：「冻结软件版本，并记录所用硬件、配置与代码 commit」。
> 本文件完整记录实验服务器硬件规格、冻结软件环境、CPU 单线程确定性配置、主修改 `b2_film` 及全部支线耗时实测与逐运行 `git_commit` 溯源；完整 36 个锁定包版本见 `requirements-lock.txt`。

---

## 1. 服务器硬件规格（`hkust-ESC8000A-E11`）

| 项目 | 实测规格 |
|---|---|
| 主机 / 架构 | `hkust-ESC8000A-E11`（`x86_64`） |
| CPU | 双路 AMD EPYC 7453 28-Core Processor（2 Sockets × 28 Cores × 2 Threads = **112 逻辑核**，最高 3.49 GHz） |
| 内存 | 503 GiB DDR4（训练/评估期间可用 > 400 GiB） |
| GPU | NVIDIA GeForce RTX 4090 × 8（每卡 24 GiB，驱动 `575.57.08`） |
| 存储 | NVMe SSD `/dev/nvme1n1p1`（3.6 TB） |
| 操作系统 / 内核 | Ubuntu 20.04.6 LTS，Linux `5.15.0-139-generic`（`glibc 2.31`） |

**重要说明：全项目所有训练与闭环评估均在纯 CPU 上完成，GPU 全程未使用。**

```python
torch.cuda.is_available()  # False —— 锁定的环境特意安装了纯 CPU 版 wheel (torch==2.15.0+cpu)
```

理由：GoToLocal 数据集为紧凑离散网格状态（4,500 条训练轨迹展开仅 16,534 个转移样本，模型参数量约 22.1 万～29.2 万），单线程 CPU 每轮训练仅需 ~2 秒；使用纯 CPU 单线程还能彻底规避 GPU 非确定性原子归约与多核 OpenMP 线程竞争，实现严格的逐字节复现。

---

## 2. 冻结软件栈（`env_embodied/`）

| 组件 | 锁定版本 | 作用说明 |
|---|---|---|
| Python | `3.12.14` | 由 `uv` 管理的独立解释器 |
| `uv` | `0.12.17` | 环境与依赖锁定工具 |
| 虚拟环境目录 | `env_embodied/` | 冻结虚拟环境（禁止在实验归档后执行任何额外 `pip install`） |
| `torch` | **`2.15.0+cpu`** | 模型训练与推理（CPU wheel） |
| `minigrid` | **`3.2.0`** | 决定 `BabyAI-GoToLocalS6N3-v0` / `S7N4-v0` 观测编码、终止语义与 `BabyAIBot` 行为 |
| `gymnasium` | **`1.4.0`** | 环境 step / reset 接口 |
| `numpy` | **`2.5.4`** | 数组与随机数 |
| `pygame-ce` / `imageio` | `2.5.8` / `2.38.1` | 离屏轨迹渲染与 GIF 合成 |
| `matplotlib` / `pandas` | `3.11.2` / `3.1.1` | 汇总统计与科学作图 |

完整 36 个包的精确锁定版本见根目录 `requirements-lock.txt`（`uv pip freeze` 原始导出）。

---

## 3. 确定性运行配置（复现关键）

| 配置项 | 设定值 | 为什么必须固定 |
|---|---|---|
| `torch.set_num_threads(1)` | **单线程 (`1`)** | 多线程浮点并行归约会导致同一随机种子的 `val_acc` 在 `0.880–0.890` 之间波动并改变选中 checkpoint；单线程下同一 seed 重复训练产出的 `best_model.pt` **逐字节一致（byte-identical）** |
| `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` | **`1`** | 避免 112 核服务器上多进程并行训练时发生 OpenMP 自旋锁竞争 |
| `SDL_VIDEODRIVER=dummy`、`SDL_AUDIODRIVER=dummy` | **必须导出** | 无显示器服务器离屏初始化 `minigrid` / `pygame` 必需（已写入 `env_embodied/bin/activate` 及各 `run*.sh`） |
| 全链路随机种子固定 | `seed ∈ {0, 1, 2}` | `src/common.py:set_all_seeds` 同步固定 `random`、`numpy`、`torch` 与 `DataLoader(generator=g)` |
| 环境 seed 划分与防泄漏断言 | `src/common.py` | 每次训练入口自动执行 `assert_no_leak()`，硬断言训练/验证数据不含任何 `4500:4800` 或 `10000:10300` 测试种子 |

---

## 4. 墙钟耗时实测（CPU 单线程，全部取自 `runs/*/config.json` 与评估 JSON）

| 任务阶段 / 模型组 | 实测耗时（CPU 单线程） | 数据来源 |
|---|---|---|
| 生成 5,000 条 `BabyAIBot` 专家轨迹（`data/expert_5000.pkl`） | **8.8 秒** | `src/gen_expert.py` |
| 原冻结基线 `baseline_seed{0,1,2}`（`patience=5`，实跑 30 / 13 / 14 ep） | **57.2s / 26.3s / 26.8s** | `runs/baseline_seed*/config.json` |
| 受控基线 `BC30`（`b3_bc30_seed{0,1,2}`，跑满 30 ep，223,071 参数） | **62.2s / 62.6s / 58.0s**（均值 60.9s） | `runs/b3_bc30_seed*/config.json` |
| **★ 主修改 `b2_film_nwd_seed{0,1,2}`**（30 ep，225,183 参数，`wd=0`） | **67.7s / 62.3s / 68.0s**（均值 **66.0s**） | `runs/b2_film_nwd_seed*/config.json` |
| **★ 主修改 `b2_film_seed{0,1,2}`**（`film_wd`，30 ep，225,183 参数） | **69.5s / 71.2s / 71.5s**（均值 **70.7s**） | `runs/b2_film_seed*/config.json` |
| **★ 纯调制 `b2_film_only_seed{0,1,2}`**（30 ep，221,087 参数） | **71.6s / 70.7s / 69.1s**（均值 **70.5s**） | `runs/b2_film_only_seed*/config.json` |
| 等参数拼接对照 `b2_concat_wide_seed{0,1,2}`（30 ep，224,683 参数） | **61.0s / 62.0s / 62.8s**（均值 **61.9s**） | `runs/b2_concat_wide_seed*/config.json` |
| 单模型 S6N3 闭环评估（300 未见 episode） | **`b2_film*`：1.2–1.4 秒**；基线：1.9–3.0 秒 | `reports/b2_film_eval.json`（FiLM 超时少故更快） |
| 单模型 S7N4 零样本迁移评估（300 未见 episode） | **`b2_film*`：1.8–2.5 秒**；基线：3.1–4.5 秒 | `reports/b2_film_s7n4.json` |
| 一键完整基线（`bash run_baseline.sh`） | **~2 分钟** | 3 seed 并行 |
| **一键完整主修改 B2 FiLM（`bash run_film_main.sh`）** | **~12–15 分钟**（单线程顺序跑满 21 个 run + 双环境评测 + 探针 + 作图） | `branches/b2_film/run.sh` |
| 主修改只读核验（`bash run_film_main.sh --eval-only`） | **< 1 秒** | 直接核验存盘 JSON |

---

## 5. 代码 `git_commit` 完整溯源说明

每个 `runs/*/config.json` 均自动写入训练当时的 `git_commit` 哈希值，全仓库三批产物的对应关系如下：

| 运行目录组 | 记录的 `git_commit` | 时点与等价性说明 |
|---|---|---|
| `runs/baseline_seed{0,1,2}`、`runs/ablation_*` | `018a695f04d0a92d50b72d3ba1073783dc4e2652` | 原基线与视频冻结时点 |
| **`runs/b2_*`（全部 21 个 B2 FiLM 与对照模型）**、`runs/b3_bc30_seed*`、`runs/b3_dagger_seed*`、`runs/b5_defrac*` | `6d91f9c165f2ef8c9986d3cd6a36a2c4df9b7a01` | 主修改 B2 FiLM、受控 `BC30` 基线及基线协议数据效率完成时点 |
| `runs/b1_*`、`runs/b3_v2_*`、`runs/b4_*`、`runs/b5_{bc,film,gru}_*` | `84dfe11897d4f7bdbd8e05427d86e60ee07add86` | 提交 B2 主修改全部代码与产物后，并行完成 B1 / B3-v2 / B4 / B5 四条辅助支线时点 |

**冻结核心区一致性证明**：从 `018a695f` 到 `6d91f9c1` 及 `84dfe118`，`src/` 目录下的唯一改动是在 `src/common.py` 与 `src/train_bc.py` 增加了只读防泄漏断言 `assert_no_leak()`（`git diff 018a695 84dfe11 -- src/` 仅 12 行只读校验，不消耗任何随机数、不改动任何模型层或评估逻辑；且 `6d91f9c` 与 `84dfe11` 之间 `src/` 差异为 **0 行**）。因此全仓库所有基线、主修改 B2 与辅助支线 B1/B3/B4/B5 的训练数据与闭环评估语义 **100% 严格一致**。
