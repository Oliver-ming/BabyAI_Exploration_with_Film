# AIAA 4220 · Project 3 · 赛道 2（BabyAI 语言模仿学习）
## 服务器开发指导文档 · v1.0

> **本文档的读者是服务器上的 AI Agent。** 目标是按优先级逐个完成开发项。
> 开始前必须先读 §0（环境准备）和 §1（不可协商的约束），其余按 §2 的优先级顺序推进。
> 所有数值标注来源：`【实测】` = 上游验证过的事实；`【规定】` = 手册硬性要求；`【推断】` = 从代码推出的结论。

---

# §0 环境准备：uv + env_embodied

## 0.1 安装 uv

```bash
# 官方安装脚本
curl -LsSf https://astral.sh/uv/install.sh | sh
# 让当前 shell 生效（重新打开终端，或执行）
source $HOME/.local/bin/env
uv --version
```

- uv 官方文档：https://docs.astral.sh/uv/
- uv 安装脚本：https://astral.sh/uv/install.sh

## 0.2 创建项目环境 `env_embodied`

**要求：环境名必须叫 `env_embodied`。**

```bash
cd /path/to/workspace

# 1) 初始化项目（生成 pyproject.toml，供 uv 管理依赖）
uv init --name p3-babyai --python 3.12

# 2) 创建名为 env_embodied 的虚拟环境
uv venv env_embodied --python 3.12
```

**创建后目录结构：**

```
workspace/
├── pyproject.toml
├── uv.lock
└── env_embodied/          ← 虚拟环境（不是默认的 .venv）
    ├── bin/activate
    └── lib/python3.12/site-packages/
```

**之后所有命令二选一：**

```bash
# 方式 A：激活
source env_embodied/bin/activate

# 方式 B：显式指定解释器（适合脚本/CI）
uv pip install --python env_embodied/bin/python ...
env_embodied/bin/python src/train_bc.py ...
```

**注意**：`uv venv env_embodied` 之后，`uv run` 默认会找 `.venv`，所以要么用 `uv run --python env_embodied/bin/python`，要么先 activate。**推荐统一 activate，避免混淆。**

## 0.3 安装依赖（锁版本）

**两级策略：核心依赖精确 pin，辅助包给下限。**

```bash
source env_embodied/bin/activate

# ① PyTorch：服务器无 GPU 用 CPU 版（本任务不需要 GPU）
uv pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cpu

# ② 核心依赖：必须精确 pin，三个 seed 必须跑在同一套版本上
uv pip install \
  "minigrid==3.2.0" \
  "gymnasium==1.4.0" \
  "numpy==2.5.4"

# ③ 辅助依赖：给下限即可（装了之后立刻 freeze）
uv pip install \
  "pygame-ce>=2.5.0" \
  "matplotlib>=3.10" \
  "pandas>=2.2" \
  "tqdm>=4.66" \
  "imageio>=2.35" \
  "tensorboard>=2.16"
```

**为什么这样分**：`minigrid / gymnasium / numpy / torch` 的版本会直接影响环境行为（观测编码、step 语义、随机数流），必须钉死。其余只影响工具链，给下限避免"版本不存在"的报错。

**装完立刻锁定**（§0.7），`requirements-lock.txt` 才是最终的可复现凭证。

**如果服务器有 NVIDIA GPU 且想用 GPU 训练**（本任务非必需，但更快）：

```bash
# CUDA 12.1 版本
uv pip install "torch==2.15.0" --index-url https://download.pytorch.org/whl/cu121
```

## 0.4 国内镜像（服务器在国内时强烈建议）

```bash
export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple/"
export UV_INDEX_STRATEGY="unsafe-best-match"
# 或写入 ~/.config/uv/uv.toml
```

- 阿里云 PyPI：https://mirrors.aliyun.com/pypi/simple/
- 清华 PyPI：https://pypi.tuna.tsinghua.edu.cn/simple/

## 0.5 验证环境

```bash
source env_embodied/bin/activate
python - <<'PY'
import torch, gymnasium, minigrid, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("gymnasium", gymnasium.__version__)
print("minigrid", minigrid.__version__)
print("numpy", numpy.__version__)
PY
```

**预期输出**（版本号必须与上面 pin 的完全一致）：

```
torch 2.15.0 cuda False        # 或 True（有 GPU 时）
gymnasium 1.4.0
minigrid 3.2.0
numpy 2.5.4
```

## 0.6 无显示器服务器必须设的变量

```bash
export SDL_VIDEODRIVER=dummy     # 防止 pygame 尝试开窗口
export SDL_AUDIODRIVER=dummy
```

**没设这个会导致 `gym.make("BabyAI-...")` 卡住或报 pygame 显示错误。** 建议写进 `env_embodied/bin/activate` 末尾或 `.envrc`。

## 0.7 锁定环境

```bash
uv pip freeze > requirements-lock.txt
cat requirements-lock.txt
```

**`requirements-lock.txt` 是提交物之一**（手册要求"锁定的环境规格"）。

---

# §1 不可协商的约束

## 1.1 手册硬性规定【规定】

| 项目 | 规定值 |
|---|---|
| 规定环境 | `BabyAI-GoToLocalS6N3-v0` |
| 专家数据 | **最多** 5,000 条**成功**专家 episode |
| 训练预算 | **最多** 30 epoch（含早停） |
| 初始化 seed | **3 个** |
| 评估 | 每模型 **300 个未见环境 seed** |
| 数据划分 | **按环境 seed 划分**，测试 seed 不进训练 |
| 模型选择 | **仅用验证集动作准确率** |
| 主指标 | 未见 seed 上的**闭环 episode Success 率** |
| 次指标 | 平均回报、成功 episode 长度中位数、留出动作准确率、`BabyAI-GoToLocalS7N4-v0` Success、数据效率 |
| 规定证据 | 训练/验证损失、3 seed Success 对比、**按 mission 类型分解**、**≥3 个语言接地失败案例** |

## 1.2 三条最容易违反的规则

**① 不许用闭环 Success 选 checkpoint。**
必须用**验证集动作准确率**选。闭环 Success 只能用于最终报告。违反这条会在"实验协议"一项直接扣分。

**② 不许随机划分 episode。**
必须**按环境 seed** 切分。原因见 §1.3。

**③ 主指标是 Success，不是动作准确率。**
【实测】如果模型"永远预测 forward"，动作准确率可达 **57.0%**（因为专家数据里 forward 占 57%），但闭环 Success 会是 0。手册明确写"单独的动作准确率不是任务指标"。

## 1.3 数据划分的具体做法（必须照做）

```python
TRAIN_SEEDS   = list(range(0, 4000))        # 用于生成 5000 条专家演示
VAL_SEEDS     = list(range(4000, 4500))     # 用于选 checkpoint
TEST_SEEDS    = list(range(4500, 4800))     # 300 个，用于最终评估
# S7N4 迁移用独立的 seed 段
TEST_SEEDS_S7 = list(range(10000, 10300))
```

**为什么必须按 seed 划分**：【实测】同一个 seed 生成的网格是唯一的（2000 seeds → 2000 unique grids），但相邻 seed 之间高度相似。随机划分 episode 会导致近乎重复的样本同时出现在训练和测试集，产生数据泄漏。

**测试 seed 绝不能出现在 TRAIN_SEEDS 里**，包括 DAgger 的第二轮采集。

---

# §2 开发优先级

> **严格按这个顺序做。** 每完成一项，运行该项的验收命令，通过后再进入下一项。
> P0 是"没做完其余全部作废"。

## 优先级总览

| 优先级 | 任务 | 产出 | 依赖 |
|---|---|---|---|
| **P0-1** | 仓库克隆 + 环境验证 | `env_embodied` 可 import | §0 |
| **P0-2** | 专家数据生成器 | `data/expert_5000.pkl` | P0-1 |
| **P0-3** | 数据集统计审计 | `reports/data_audit.md` | P0-2 |
| **P1-1** | BC 基线模型 + 训练循环 | `runs/baseline_seed{0,1,2}/` | P0-3 |
| **P1-2** | 闭环评估脚本 | `reports/baseline_eval.json` | P1-1 |
| **P1-3** | 回放/可视化 | `videos/*.gif` | P1-2 |
| **P2-1** | mission 类型分解 | `reports/mission_breakdown.csv` | P1-2 |
| **P2-2** | S7N4 迁移评估 | `reports/s7n4_transfer.json` | P1-2 |
| **P2-3** | 失败案例采集 | `reports/failures/*.png` + `.json` | P1-3 |
| **P3-1** | 主修改：DAgger-lite | `runs/dagger_seed{0,1,2}/` | P2-1 |
| **P3-2** | 消融：数据效率 20/50/100% | `reports/data_efficiency.csv` | P3-1 |
| **P4-1** | 汇总表 + 作图 | `reports/figures/*.png` | P3-2 |
| **P4-2** | 复现包整理 | `README.md` + `requirements-lock.txt` | P4-1 |

---

# §3 P0-1：仓库克隆与环境验证

## 3.1 必需的仓库与链接

| 仓库 | 链接 | 用途 | 建议 commit |
|---|---|---|---|
| **Minigrid**（主仓库） | https://github.com/Farama-Foundation/Minigrid | 环境 + BabyAIBot 专家机器人 | 与 `minigrid==3.2.0` 对应的 tag |
| Minigrid 文档：GoToLocal | https://minigrid.farama.org/environments/babyai/GoToLocal/ | 环境参数与注册名 | — |
| Minigrid 文档：环境列表 | https://minigrid.farama.org/environments/babyai/ | 全部 BabyAI 变体 | — |
| **BabyAIBot 参考测试** | https://github.com/Farama-Foundation/Minigrid/blob/main/tests/test_baby_ai_bot.py | 专家机器人正确用法 | — |
| Minigrid 训练教程 | https://minigrid.farama.org/content/training/ | SB3 用法参考（本任务不用 PPO，但 feature extractor 可参考） | — |
| **原始 BabyAI 项目** | https://github.com/mila-iqia/babyai | 任务设计与语言模板的原始实现 | — |
| **BabyAI 论文** | https://arxiv.org/abs/1810.08272 | 任务动机、指令生成机制 | — |
| uv 文档 | https://docs.astral.sh/uv/ | 环境管理 | — |

## 3.2 克隆命令

```bash
mkdir -p third_party && cd third_party

# 1) Minigrid（只读参考 + 直接看 BabyAIBot 源码）
git clone --depth 1 --branch v3.2.0 \
  https://github.com/Farama-Foundation/Minigrid.git minigrid-src

# 2) 原始 BabyAI（参考任务设计与指令模板，不要用它来跑实验）
git clone --depth 1 https://github.com/mila-iqia/babyai.git babyai-orig
```

**注意**：`pip install minigrid==3.2.0` 已经提供了运行所需的一切。`third_party/` 只是**读源码的参考**，**不要把它们加进 PYTHONPATH**，否则版本冲突。

## 3.3 验收命令

```bash
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy

python - <<'PY'
import gymnasium as gym, minigrid
from minigrid.utils.baby_ai_bot import BabyAIBot

env = gym.make("BabyAI-GoToLocalS6N3-v0")
obs, _ = env.reset(seed=0)
u = env.unwrapped

assert obs["image"].shape == (7, 7, 3), obs["image"].shape
assert obs["image"].dtype.name == "uint8"
assert env.action_space.n == 7
assert u.max_steps == 36
assert isinstance(obs["mission"], str)

bot = BabyAIBot(u)
a = bot.replan(None)
assert a is not None
print("PASS", obs["mission"], "| first expert action:", a)
PY
```

预期输出：`PASS go to the blue key | first expert action: <某个 0-6 的整数>`

---

# §4 P0-2：专家数据生成器

## 4.1 已核实的机制（实现前必须理解）

### ① 成功条件：**相邻 + 面向**，不是"站在物体上"【实测】

```python
# verifier.py:309  GoToInstr.verify_action
for pos in self.desc.obj_poss:
    if np.array_equal(pos, self.env.front_pos):   # front_pos = 面前那一格
        return "success"
```

【实测】把 agent 放在钥匙旁边并面向它，**动作是 left/right 也会触发成功**——因为 `verify` 在每步之后都检查状态，不看你做了什么动作。

**这一点极其重要**：`Key` 的 `can_overlap()` 是 `False`，所以你**永远无法站在钥匙上**。必须站在它旁边并转身面向它。

### ② `max_steps=36` 的来源【实测】

```python
# roomgrid_level.py:80
nav_time_room = self.room_size ** 2          # 36
nav_time_maze = nav_time_room * num_rows * num_cols   # 36 * 1 * 1
num_navs = self.num_navs_needed(self.instrs)  # GoToLocal -> 1
self.max_steps = num_navs * nav_time_maze     # 36
```

### ③ 专家 episode 极短【实测】

- 200 seeds → 200 成功，**764 个转移**，平均 **3.82 步/episode**
- 丢弃率：0/200（专家全成功）

**含义：5000 条 episode 只有约 19,100 个 (obs, action) 样本。** 这个数字必须写进报告。

### ④ 动作分布【实测，300 episode】

| 动作 | 占比 |
|---|---|
| `forward` | **57.0%** |
| `left` | 21.9% |
| `right` | 21.1% |
| `pickup/drop/toggle/done` | **0%** |

**"永远预测 forward"就能拿到 57% 的准确率**——这是报告里必须呈现的"平凡基线"。

### ⑤ 观测是 agent 中心坐标系，但 `direction` 仍需输入【实测】

在同一个世界状态下改变 `agent_dir`，7×7 patch **会变化**（因为视野朝向变了），所以：

- patch 本身已经包含了"前方是画面下方"这一信息
- 但 `direction` 提供了**绝对朝向**，对"要往哪边转"这个决策是必需的

**结论：三个输入（image / mission / direction）全部必需，不能省。**

### ⑥ mission 有两套模板，`a` vs `the` 有语义【实测，4000 seeds】

```
"go to the X"   → 89.7%   （场景里只有 1 个匹配物体）
"go to a X"     → 10.3%   （场景里有 >1 个匹配物体，任一个都算成功）
```

**总共 36 种 mission 字符串**（6 颜色 × 3 类型 × 2 冠词），不是 18 种。

`surface()` 逻辑（`verifier.py:97-101`）：

```python
if len(self.obj_set) > 1:
    s = "a " + s      # 多个匹配 -> "a"
else:
    s = "the " + s    # 唯一匹配 -> "the"
```

### ⑦ 网格多样性【实测】

2000 seeds → **2000 个唯一网格**，但只有 **1419 个唯一首帧观测**。说明同一观测可能对应不同的绝对朝向（因为 `direction` 是单独字段）。

### ⑧ reset 会往 stdout 打噪声【实测】

约 0.01 行/reset 的 `Sampling rejected: unreachable object at (x, y)`。生成 5000 条时会有几十行噪声，**用 `contextlib.redirect_stdout` 包住或直接忽略**。

## 4.2 生成脚本规范

创建 `src/gen_expert.py`，要求：

```python
"""生成 5000 条成功专家 episode。"""
import contextlib, io, pickle
from pathlib import Path
import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.utils.baby_ai_bot import BabyAIBot

ENV_ID = "BabyAI-GoToLocalS6N3-v0"
N_EPISODES = 5000
TRAIN_SEEDS = range(0, 4000)

def gen_one(env, seed):
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    bot = BabyAIBot(u)
    transitions, imgs, dirs, missions = [], [], [], []
    last_action = None
    for _ in range(u.max_steps):
        action = bot.replan(last_action)
        if action is None:
            break
        transitions.append(action)
        imgs.append(obs["image"].copy())
        dirs.append(int(obs["direction"]))
        missions.append(obs["mission"])
        obs, r, terminated, truncated, _ = env.step(action)
        last_action = action
        if terminated or truncated:
            break
    if not terminated:
        return None                      # 只保留成功
    return dict(
        seed=seed,
        mission=missions[0],
        images=imgs,
        directions=dirs,
        actions=transitions,
        episode_len=len(transitions),
    )

def main():
    out = Path("data"); out.mkdir(exist_ok=True)
    env = gym.make(ENV_ID)
    episodes, seed = [], 0
    with contextlib.redirect_stdout(io.StringIO()):   # 屏蔽 Sampling rejected 噪声
        while len(episodes) < N_EPISODES:
            seed += 1
            if seed not in TRAIN_SEEDS:
                continue
            ep = gen_one(env, seed)
            if ep is not None:
                episodes.append(ep)
    with open(out / "expert_5000.pkl", "wb") as f:
        pickle.dump(episodes, f)
    n_tr = sum(e["episode_len"] for e in episodes)
    print(f"{len(episodes)} episodes, {n_tr} transitions, avg {n_tr/len(episodes):.2f} steps")

if __name__ == "__main__":
    main()
```

## 4.3 验收命令

```bash
python src/gen_expert.py
python - <<'PY'
import pickle, collections
eps = pickle.load(open("data/expert_5000.pkl","rb"))
assert len(eps) == 5000
tr = sum(e["episode_len"] for e in eps)
c = collections.Counter(a for e in eps for a in e["actions"])
print(f"episodes={len(eps)} transitions={tr} avg_len={tr/len(eps):.2f}")
print("action dist:", {k: f"{v/tr:.1%}" for k,v in c.most_common()})
forbidden = {3,4,5,6}
assert not (set(c) & forbidden), f"expert used forbidden actions: {set(c) & forbidden}"
print("PASS")
PY
```

**预期**：`episodes=5000 transitions≈19000 avg_len≈3.8`，`forward≈57%`，`left/right≈22%/21%`，无 pickup/drop/toggle/done。

---

# §5 P0-3：数据集统计审计

生成 `reports/data_audit.md`，必须包含：

1. episode 数、转移总数、平均长度分布（min/median/max）
2. 动作分布（对照 §4.1 ④）
3. **mission 覆盖表**：36 种字符串各自的 episode 数（检查是否有 0 出现的）
4. 冠词分布：`the` vs `a`
5. 每个 episode 的 `obj_poss` 数量分布（1 个 vs 多个匹配物体）
6. 观测唯一性：唯一 `(image, direction)` 组合数 / 总转移数

**验收**：36 种 mission 中至少有 30 种在 5000 条里出现频次 ≥ 20。若某些太少，在报告中说明并将它们单独列为"低覆盖 mission"。

---

# §6 P1-1：BC 基线模型

## 6.1 必须实现的架构【规定】

```
image (7,7,3) uint8
  │
  ├─→ 小型 CNN 或 MLP ──→ v (视觉特征)
  │
mission (string)
  │
  ├─→ 固定词表 + 平均 32 维词嵌入 ──→ l (32 维)
  │
direction (int 0-3)
  │
  └─→ one-hot(4) ──→ d (4 维)

concat(v, l, d) ──→ Linear(→128) ──→ ReLU ──→ Linear(128→7) ──→ logits
                                                                    ↓
                                                          CrossEntropyLoss
```

## 6.2 实现要点

**① 图像通道顺序**

`obs["image"]` 是 `(7,7,3)`（HWC）。PyTorch Conv2d 要 `(C,H,W)`。**必须显式 `permute(2,0,1)`**，不要依赖任何自动转置（这里没有 SB3）。

**② 图像值是类别索引，不是颜色**

`image` 的值域是 `{0,1,2,4,5}` 等类别 id。**必须归一化到 [0,1]**（除以 255 或用 Embedding 层）。直接用 0-10 的原始值喂 CNN 会让网络拟合无意义的数值距离。

建议：用 `nn.Embedding(num_objects, k)` 处理通道 0，或对每个通道做 one-hot（见 §8 的"表征"改动）。

**③ 词表必须固定且覆盖全部 36 种 mission**

```python
COLORS = ["red","green","blue","purple","yellow","grey"]
TYPES  = ["ball","box","key"]
VOCAB = ["<pad>", "go", "to", "the", "a"] + COLORS + TYPES
# 大小 = 5 + 6 + 3 = 14
```

**【重要】** 从数据里动态构建词表是可以的，但**必须保存到 checkpoint 旁边**，否则评估时无法加载。

**④ 词嵌入的平均必须处理变长**

`"go to a red key"` 是 5 个词，`"go to the red key"` 也是 5 个词，但 padding 策略要统一。用 `mean(emb[valid_tokens])` 而不是 `mean(emb[all_tokens])`。

**⑤ 早停规则**

监控**验证集动作准确率**，patience 建议 5 epoch，最大 30 epoch。**不许监控闭环 Success。**

## 6.3 训练脚本规范

创建 `src/train_bc.py`，必须支持：

```bash
python src/train_bc.py --seed 0 --epochs 30 --out runs/baseline_seed0
python src/train_bc.py --seed 1 --epochs 30 --out runs/baseline_seed1
python src/train_bc.py --seed 2 --epochs 30 --out runs/baseline_seed2
```

每个 run 目录必须含：

```
runs/baseline_seed0/
├── config.json          # 全部超参 + 数据路径 + git commit
├── vocab.json           # 词表（推理时需要）
├── best_model.pt        # 按验证准确率选出的 checkpoint
├── history.json         # 每 epoch 的 train_loss/val_loss/train_acc/val_acc
└── predictions_val.npz  # 验证集逐样本预测（用于分析）
```

## 6.4 验收命令

```bash
for s in 0 1 2; do
  python src/train_bc.py --seed $s --epochs 30 --out runs/baseline_seed$s
done

python - <<'PY'
import json, glob
for d in sorted(glob.glob("runs/baseline_seed*")):
    h = json.load(open(f"{d}/history.json"))
    best = max(h, key=lambda e: e["val_acc"])
    print(f"{d}: best val_acc={best['val_acc']:.3f} @ epoch {best['epoch']}")
PY
```

**预期**：验证集动作准确率 **0.70–0.95**。如果 < 0.60 说明实现有问题（检查通道顺序、归一化、词表）。

**注意**：验证准确率高**不代表**闭环 Success 高（见 §1.2 ③）。这两个数字会在 §7 里同时出现，报告里必须解释它们的差距。

---

# §7 P1-2：闭环评估

## 7.1 这是最重要的脚本

`src/eval_closed_loop.py` 必须：

```python
def evaluate(model, vocab, env_id, seeds, max_episodes=None):
    """
    对每个 seed: reset -> 逐步 model.predict -> 记录 terminated/truncated
    返回: success_rate, mean_return, median_len_if_success, per_mission_stats
    """
```

## 7.2 关键实现细节

**① 必须用 `torch.no_grad()` + `model.eval()`**

**② 必须是闭环**：每步把 `model` 的预测喂回环境，**不能用 teacher forcing**

**③ 必须记录 `terminated` 而非 `truncated`**

【实测】BabyAI 里 `terminated=True` 就是成功（`GoToInstr` 只有 success 才会 `terminated`，没有 failure 分支）。`truncated=True` 是超时（36 步用完）。

**④ 必须记录 verbosity**

【实测】`reset()` 会打 `Sampling rejected` 噪声。评估 900 个 episode 会有十几行，用 `redirect_stdout` 屏蔽。

**⑤ 单 seed 评估成本【推断】**

- 300 seeds × 平均 3.8 步（成功）或 36 步（失败）
- 最坏 300 × 36 = 10,800 步
- 模型前向 ~0.5 ms/step → **约 5 秒/模型**
- 3 个模型 → 15 秒

**所以评估非常便宜，可以频繁跑。**

## 7.3 验收命令

```bash
python src/eval_closed_loop.py --runs runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
  --env-id BabyAI-GoToLocalS6N3-v0 --seeds 4500:4800 --out reports/baseline_eval.json
```

`reports/baseline_eval.json` 必须含每个 run 的：

```json
{
  "run": "runs/baseline_seed0",
  "env_id": "BabyAI-GoToLocalS6N3-v0",
  "seeds": "4500:4800",
  "n_episodes": 300,
  "val_action_acc": 0.842,
  "closed_loop_success": 0.31,
  "mean_return": 0.28,
  "median_len_success": 5,
  "per_mission": { "go to the blue key": {"n": 9, "success": 0.44}, ... },
  "action_histogram": { "forward": 0.61, "left": 0.19, "right": 0.20 },
  "invalid_action_rate": 0.0
}
```

## 7.4 必须分析的"两个数字的差距"

报告里必须回答：

> **验证集动作准确率 0.84，为什么闭环 Success 只有 0.31？**

这就是**分布偏移**（covariate shift）：

- 训练时，模型看到的是**专家访问的状态**（专家从不站在死角）
- 评估时，模型的小错误把它带到**专家从未到过的状态**，然后它不知道该做什么
- 错误累积 → 36 步耗尽 → 失败

**这句话是这条赛道的核心论点，也是 P3-1（DAgger）的动机。**

---

# §8 P1-3：回放与可视化

## 8.1 需要产出的视频

- **≥2 条成功轨迹** `videos/success_*.gif`
- **≥2 条失败轨迹** `videos/failure_*.gif`
- 时长 30-60 秒（合并成一条演示视频时）

## 8.2 渲染方法

```python
# 无显示器服务器用 rgb_array，不要用 human
env = gym.make("BabyAI-GoToLocalS6N3-v0", render_mode="rgb_array")
frame = env.render()          # (H, W, 3) uint8
```

用 `imageio` 或 `PIL` 逐帧存成 gif/mp4。**`render_mode="human"` 在服务器上会失败。**

## 8.3 每个失败案例必须记录的信息

```json
{
  "seed": 4523,
  "mission": "go to the green box",
  "steps": [
    {"t":0, "action":"forward", "agent_pos":[3,1], "agent_dir":1},
    ...
  ],
  "final_distance_to_target": 2,
  "failure_type": "attribute_confusion",
  "note": "走向 green ball 而非 green box"
}
```

`failure_type` 从这三类里选（见 §10.3）：

- `attribute_confusion`：颜色/类型对错了
- `distractor_priority`：走向最近的物体而非指定物体
- `orientation_lock`：在目标旁但朝向不对，反复原地转

**`orientation_lock` 是 GoToLocal 特有的**【实测】：因为成功条件是"站在旁边 + 面向它"，agent 可能站对了位置但转错方向，然后一直转圈直到超时。这是很好的一种失败案例。

---

# §9 P2：分析与迁移

## 9.1 P2-1 mission 类型分解

按 §4.1 ⑥ 的 36 种字符串分组报成功率。输出 `reports/mission_breakdown.csv`：

```csv
mission,color,type,article,n_train_episodes,n_test_episodes,success_rate
go to the blue key,blue,key,the,203,9,0.44
go to a red box,red,box,a,15,1,0.00
...
```

**必须标出**哪些 mission 训练样本 < 20（低覆盖），并在报告中说明它们的成功率不可靠。

## 9.2 P2-2 S7N4 迁移

```bash
python src/eval_closed_loop.py --runs runs/baseline_seed0 runs/baseline_seed1 runs/baseline_seed2 \
  --env-id BabyAI-GoToLocalS7N4-v0 --seeds 10000:10300 \
  --out reports/s7n4_transfer.json
```

**注意**：S7N4 的 `max_steps=49`，房间 7×7、干扰物 4 个。观测空间仍是 `(7,7,3)`，所以**模型可以直接加载，不需要任何适配**。

**预期**：S7N4 上的 Success 会低于 S6N3（视野覆盖率下降 + 干扰物变多），这个下降幅度本身是"泛化能力"的证据。

## 9.3 P2-3 失败案例采集

从 §8.3 的记录里，按 `failure_type` 各挑至少 1 个，共 ≥3 个。每个必须：

1. 有轨迹 gif
2. 有逐步动作日志
3. 有**明确的归因**（"它把 X 错认成了 Y"），不是"它失败了"
4. 说明**修改方案能否解决它**（如果不能，要诚实说）

---

# §10 P3：主修改与消融

## 10.1 主修改：DAgger-lite【规定选项之一】

### 为什么选它

【推断】从 §7.4 的分析看，基线的核心瓶颈是**分布偏移**而不是表达能力。DAgger 直接针对这一点，是所有选项里机制最硬的一个。

### 实现流程

```
Round 1: 用 5000 条专家数据训练基线 BC  → 得到 π_1
Round 2: 用 π_1 在 TRAIN_SEEDS 上跑闭环，收集它访问到的状态
         （注意：用 π_1 自己产生的轨迹，不是专家的）
         在这些状态上，用 BabyAIBot 标注"专家会怎么做"
         得到 N_dagger 条修正样本
Round 3: 用 (专家数据 + DAgger 数据) 重新训练 → π_2
评估:   用 TEST_SEEDS 评估 π_2
```

### 关键实现细节

**① 标注时 bot 需要正确的环境状态**

`BabyAIBot(u)` 直接读 `u.grid`。所以在采集 DAgger 数据时：

```python
obs, _ = env.reset(seed=seed)
bot = BabyAIBot(env.unwrapped)      # 每个 episode 新建
for t in range(max_steps):
    expert_a = bot.replan(last_expert_a)   # 专家在这个状态会做什么
    policy_a, _ = model.predict(obs)       # 策略实际做什么（用于推进环境）
    dagger_buffer.append((obs, expert_a))  # 存的是专家动作
    obs, r, te, tr, _ = env.step(policy_a) # 环境按策略动作推进
    last_expert_a = expert_a               # 注意：传给 bot 的是专家动作链
    if te or tr: break
```

**② `last_action` 传什么？**

`bot.replan(action_taken)` 的参数是"上一步实际发生了什么"。在 DAgger 采集里，**环境执行的是策略动作**，但 bot 的内部规划是基于专家动作链的。两种做法：

- 保守（推荐）：传 `expert_a`（专家视角的动作链），让 bot 内部状态一致
- 激进：传 `policy_a`，让 bot 知道"实际偏离了"

**建议先用保守做法**，在报告里说明。如果 bot 频繁报错（`DisappearedBoxError` 等），改用激进做法。

**③ 数据量分配**

手册限制"最多 5000 条成功专家 episode"。DAgger 数据**不占这 5000 条预算**（因为它不是"专家 episode"），但**必须在报告里明确写出用了多少条 DAgger 样本**，否则会被质疑"偷偷加数据"。

建议：DAgger 采集 1000-2000 条轨迹，约 5000-10000 个转移（与专家数据同量级）。

**④ 不能碰 TEST_SEEDS**

DAgger 采集只能用 `TRAIN_SEEDS` + `VAL_SEEDS`。

### 验收

```bash
python src/train_dagger.py --seed 0 --out runs/dagger_seed0
python src/eval_closed_loop.py --runs runs/dagger_seed0 \
  --env-id BabyAI-GoToLocalS6N3-v0 --seeds 4500:4800 \
  --out reports/dagger_eval.json
```

## 10.2 消融：数据效率 20/50/100%

用**同样的基线架构**，只用 20% / 50% / 100% 的专家数据训练，看 Success 如何变化。

```bash
for frac in 0.2 0.5 1.0; do
  python src/train_bc.py --seed 0 --data-frac $frac --out runs/defrac${frac}_seed0
done
```

输出 `reports/data_efficiency.csv`：

```csv
frac,n_episodes,val_acc,closed_loop_success
0.2,1000,0.71,0.08
0.5,2500,0.79,0.21
1.0,5000,0.84,0.31
```

**这个表的用途**：回答"DAgger 的收益是来自更好的分布覆盖，还是仅仅因为多了数据？"

如果 `dagger(1000 专家 + DAgger)` 的表现 > `bc(5000 专家)`，那说明分布覆盖比数据量更重要——这是 DAgger 的核心论点。

## 10.3 五个官方选项的取舍（供参考，不要求全做）

| 选项 | 机制 | 推荐度 | 理由 |
|---|---|---|---|
| **DAgger-lite** | 消除分布偏移 | ★★★★★ | 直击基线核心瓶颈，手册明确列出 |
| 数据效率 | 改正则化/采样 | ★★★★ | 有连续实验设计，能画漂亮曲线 |
| 跨模态融合（FiLM） | 视觉以 mission 为条件 | ★★★ | 技术"高级"但本任务难度低，收益可能不明显 |
| 语言编码器（GRU） | 保留词序 | ★★ | **语言空间只有 14 词 / 36 种组合，平均嵌入够用** |
| 课程学习（S5N2→S6N3） | 先易后难 | ★ | S5N2 和 S6N3 的 mission 空间**完全相同**，更像域随机化 |

---

# §11 P4：汇总与交付

## 11.1 必须的图表

| 图 | 内容 | 数据来源 |
|---|---|---|
| 图 1 | 训练/验证损失曲线（3 seed，mean ± SD） | `history.json` |
| 图 2 | **动作准确率 vs 闭环 Success 的散点/柱状对比** | `baseline_eval.json` |
| 图 3 | 3 seed Success 柱状图（基线 vs 修改） | 两份 eval json |
| 图 4 | mission 类型分解热图（36 种 × 成功率） | `mission_breakdown.csv` |
| 图 5 | 数据效率曲线（20/50/100%） | `data_efficiency.csv` |
| 图 6 | S6N3 vs S7N4 迁移对比 | `s7n4_transfer.json` |
| 图 7 | 成功/失败轨迹拼图 | `videos/*.gif` + 帧 |

**图 2 是这条赛道的"灵魂图"**——它直观展示"准确率不等于成功率"。

## 11.2 提交物清单

```
p3-babyai/
├── README.md                      # 五组命令（安装/生成数据/训练/评估/作图）
├── requirements-lock.txt          # uv pip freeze 的输出
├── pyproject.toml + uv.lock       # uv 管理
├── src/
│   ├── gen_expert.py
│   ├── train_bc.py
│   ├── train_dagger.py
│   ├── eval_closed_loop.py
│   ├── render_episodes.py
│   └── make_figures.py
├── data/expert_5000.pkl           # 或生成脚本 + 说明（文件可能较大）
├── runs/*/                        # 每个 run 的 config + history + model
├── reports/
│   ├── data_audit.md
│   ├── baseline_eval.json
│   ├── mission_breakdown.csv
│   ├── s7n4_transfer.json
│   ├── data_efficiency.csv
│   ├── failures/                  # 失败案例 gif + json
│   └── figures/
├── videos/
└── docs/                          # 本文档 + 手册
```

## 11.3 README 必须包含的五组命令

```bash
# 1) 安装
uv venv env_embodied --python 3.12 && source env_embodied/bin/activate
uv pip install -r requirements-lock.txt

# 2) 生成专家数据
python src/gen_expert.py

# 3) 训练基线
python src/train_bc.py --seed 0 --out runs/baseline_seed0

# 4) 评估
python src/eval_closed_loop.py --runs runs/baseline_seed0 --env-id BabyAI-GoToLocalS6N3-v0 --seeds 4500:4800

# 5) 作图
python src/make_figures.py --reports reports --out reports/figures
```

---

# §12 常见故障排查

| 现象 | 原因 | 修复 |
|---|---|---|
| `pygame.error: No available video device` | 服务器无显示器 | `export SDL_VIDEODRIVER=dummy` |
| `gym.make` 卡住 | 同上 | 同上 |
| 训练 loss 不降 | 图像未归一化 / 通道顺序错 | 检查 `permute(2,0,1)` + `/255` |
| 验证准确率 0.95 但 Success 0 | 过拟合专家状态分布 | **正常现象**，这是 DAgger 的动机 |
| `KeyError` 在词表查找 | 词表没覆盖某 mission | 用 §6.2 ③ 的固定词表 |
| `BabyAIBot` 报 `DisappearedBoxError` | bot 内部状态与 env 不一致 | 每 episode 新建 bot；检查 `last_action` 传递 |
| `Sampling rejected` 刷屏 | 环境正常行为 | `redirect_stdout` 屏蔽 |
| 图像 shape 报错 | `(7,7,3)` 直接喂 Conv2d | `permute(2,0,1)` |
| 评估很慢 | 用了 teacher forcing 或 batch=1 | 闭环评估本来就慢；3 模型约 15 秒可接受 |
| 视频全黑 | `render_mode` 不对 | 用 `rgb_array`，不要 `human` |
| uv 装 torch 卡住 | 默认 PyPI 慢 | 用 `--index-url` 指定 PyTorch 官方或国内镜像 |

---

# §13 一句话总结

这条赛道的核心矛盾是：**用"专家在最优状态下做的动作"训练，却要在"模型自己犯错后的状态"上执行**。所有有效的改动（尤其 DAgger）本质上都在缩小这个 gap；而所有无效的分析（比如只看动作准确率）本质上都在忽略这个 gap。
