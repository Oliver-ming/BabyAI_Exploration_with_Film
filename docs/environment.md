# 环境与硬件记录（手册 §2.1 要求）

> 手册要求「冻结软件版本，并记录所用硬件、配置与代码 commit」。
> 本文件是硬件与运行配置的唯一记录处；软件版本见 `requirements-lock.txt`。

## 1. 硬件

| 项目 | 值 |
|---|---|
| CPU | AMD EPYC 7453 28-Core Processor（**112 逻辑核**，双路 × 28 核 × 2 超线程） |
| 内存 | 503 GiB（评估时可用 424 GiB） |
| GPU | NVIDIA GeForce RTX 4090 × 4（24 GiB each） |
| GPU 驱动 | 575.57.08 |
| 存储 | `/dev/nvme1n1p1`，3.6 TB（训练/评估时可用 419 GB） |
| OS / 内核 | Linux 5.15.0-139-generic，`x86_64-with-glibc2.31` |

**重要：GPU 全程未使用。** 本任务的所有训练与评估都在 **CPU** 上完成：

```python
torch.cuda.is_available()  # False —— 因为安装的是 CPU 版 torch（+cpu wheel）
```

理由：任务规模极小（16534 个训练转移、223k 参数、单 epoch < 2 秒），
CPU 单线程已足够，且**多线程会破坏可复现性**（见 §3）。

## 2. 软件（冻结）

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.12.14 | 由 `uv` 管理的独立解释器 |
| uv | 0.12.17 | 环境管理 |
| 虚拟环境 | `env_embodied/`（非默认 `.venv`） | 手册/指导文档要求的环境名 |
| torch | **2.15.0+cpu** | CPU wheel，`--index-url https://download.pytorch.org/whl/cpu` |
| minigrid | **3.2.0** | 决定环境行为（观测编码、step 语义） |
| gymnasium | **1.4.0** | 同上 |
| numpy | **2.5.4** | 同上 |
| pygame-ce | 2.5.8 | 仅渲染用（`SDL_VIDEODRIVER=dummy`） |
| matplotlib / pandas / imageio | 3.11.2 / 3.1.1 / 2.38.1 | 仅作图与视频 |

完整 36 个包的精确版本见 `requirements-lock.txt`（`uv pip freeze` 输出）。

## 3. 运行配置（影响可复现性）

| 配置 | 值 | 为什么重要 |
|---|---|---|
| `torch.set_num_threads(1)` | **单线程** | 多线程改变浮点归约顺序，同一 seed 的 `val_acc` 会在 0.880–0.890 间抖动（实测），Success 随之变化。单线程下同 seed 两次训练的 `best_model.pt` **逐字节相同** |
| `SDL_VIDEODRIVER=dummy` | 必须 | 无显示器服务器上否则 `gym.make` 卡住或 pygame 报错。已写入 `env_embodied/bin/activate` 末尾 |
| `SDL_AUDIODRIVER=dummy` | 必须 | 同上 |
| 随机种子 | `random` / `numpy` / `torch` / `DataLoader generator` 全部显式设定 | 见 `common.set_all_seeds` |
| 环境 seed 划分 | `common.py` 是唯一出处 | 见 `reports/data_audit.md` §0 |

## 4. 耗时实测（供复现者预估）

| 任务 | 耗时（CPU 单线程） |
|---|---|
| 生成 5000 条专家 episode | 8.8 秒 |
| BC 训练 1 个 seed（30 epoch / 16534 转移） | ~55 秒 |
| 训练 1 个 seed（+6000 DAgger 转移） | ~80 秒 |
| 闭环评估 1 个模型（300 episode） | 2–4 秒 |
| 完整基线复现（`bash run_baseline.sh`） | **~2 分钟** |
| 完整 DAgger 支线（`bash branches/b3_dagger/run.sh`） | **~15 分钟** |

## 5. 代码 commit

每个 run 目录的 `config.json` 都记录了 `git_commit`：

| run 组 | git commit | 说明 |
|---|---|---|
| `runs/baseline_seed*`、`runs/ablation_*` | `018a695f04d0a92d50b72d3ba1073783dc4e2652` | 基线冻结时点 |
| `runs/b3_*` | `6d91f9c165f2ef8c9986d3cd6a36a2c4df9b7a01` | 支线开始时点 |

两个 commit 之间对 `src/` 的改动**只有一处新增的防护性断言**：

```python
# src/common.py  新增 assert_no_leak()；src/train_bc.py 在训练入口调用它
assert_no_leak(e["seed"] for e in train_eps)
assert_no_leak(e["seed"] for e in val_eps)
```

它只做**校验**（断言训练数据里不含测试 seed），**不改变任何随机数、模型结构或训练逻辑**。
因此两组 run 的基线代码在语义上完全一致；`git diff 018a695 6d91f9c -- src/` 可以自行核对。

> 说明这一点是为了避免评审误以为"基线是在另一份代码上跑的"。
