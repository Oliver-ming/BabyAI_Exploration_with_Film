# 个案分析：一例成功 + 五例语言接地失败 + 三例状态覆盖失败及 B2 FiLM 逐例核验（手册 §2.1 / §3.2）

> **手册要求**：§2.1 要求「至少解释一例成功与一例失败」；§3.2（赛道 2）要求「至少三个**语言接地**失败案例」。
> **核验范围与模型溯源**：本文件不仅对基线模型 `runs/baseline_seed0`（与受控 30-epoch `runs/b3_bc30_seed0` 均选中第 26 轮 `val_acc=0.8899`，闭环轨迹逐字节一致）的 **1 例典型成功（`seed 4501`）**、**5 例语言接地失败（`seed 4516, 4565, 4570, 4582, 4560`，分存于 `reports/failures/` 与 `branches/_shared/lang_grounding_failures/`）** 以及 **3 例状态覆盖类失败（`seed 4630, 4716, 4568`）** 给出逐步因果归因，还使用冻结环境 `BabyAI-GoToLocalS6N3-v0` **实测核验了唯一主修改 B2 FiLM**（主配置 `runs/b2_film_nwd_seed{0,1,2}` 与正则配置 `runs/b2_film_seed{0,1,2}` [`film_wd`]，并附纯调制 `b2_film_only_seed0` 与辅助支线 B3 DAgger 对照）在每一个具体案例上的精确步数、回报与完整动作序列。
>
> **核心实测结论**：`b2_film_nwd_seed0` 将基线的 **5 例语言接地失败 100%（5/5）全部修复为 2～8 步的最短/近优成功轨迹**，同时修复了 2 例因干扰物诱发转向失误的状态覆盖失败（`4630, 4716`）；而唯一仍超时的 **`seed 4568`** 则诚实展示了无记忆反应式策略（即便具备 FiLM 接地）在目标位于初始背后 7×7 视野盲区时的原地振荡边界，以及为何正交叠加 `FiLM + DAgger` 能将其进一步修复。

---

## 一、典型成功案例：`seed 4501` · `go to the blue key`

- **场景真值**（6×6 房间，内部可通行坐标 `1..4 × 1..4`）：目标 `blue key` 在 `(3,1)`；干扰物 `yellow box` 在 `(3,2)`（正挡在北向走廊中央）、`green box` 在 `(4,3)`；智能体初始在 `(3,4)` 朝东 `→(0)`。
- **基线表现（`runs/baseline_seed0` / `b3_bc30_seed0`）**：**成功（`terminated=True`），8 步，回报 `0.800`**。
- **主修改 B2 FiLM 表现**：
  - `runs/b2_film_nwd_seed0`、`runs/b2_film_only_seed0`、`runs/b2_film_seed2`：**成功，8 步，回报 `0.800`**，动作序列与基线**完全一致**：`['left', 'forward', 'left', 'forward', 'right', 'forward', 'forward', 'right']`；`runs/b2_film_nwd_seed2` 以 **10 步（回报 `0.750`）** 成功。
  - **多 seed 诚实披露**：`b2_film_nwd_seed1` 与 `b2_film_seed0,1` 在该局因初始 `(3,4) →` 视野内未见北侧 `(3,1)` 目标而触发靠墙 `forward_stall` 超时（36 步），体现了多 seed 如实报告不挑最优单次运行的原则。

### 场景示意与 8 步精确轨迹（`baseline_seed0` 与 `b2_film_nwd_seed0` 共同轨迹）

```text
     i=0   i=1   i=2      i=3          i=4     i=5
j=0   #     #     #        #            #       #
j=1   #     .    (2,1)→ [blue key]      .       #   ← t=7 在 (2,1) 右转朝东面向 (3,1) 成功
j=2   #     .     ↑     yellow box      .       #   ← (3,2) 干扰物封堵 x=3 直行北通道
j=3   #     .    (2,3)←  (3,3)↑     green box   #   ← t=2 在 (3,3) 见 yellow box 左转绕行
j=4   #     .     .     agent →(0)      .       #   ← t=0 起点 (3,4)
j=5   #     #     #        #            #       #
```

| 步 $t$ | 动作前位置 | 动作前朝向 | 面前格 (`front_pos`) | 执行动作 | 动作后位置 & 朝向 | 动作后面前格 |
|---|---|---|---|---|---|---|
| 0 | `(3,4)` | `→(0)` | `(4,4)` empty | `left` | `(3,4)` `↑(3)` | `(3,3)` empty |
| 1 | `(3,4)` | `↑(3)` | `(3,3)` empty | `forward` | `(3,3)` `↑(3)` | **`(3,2)` yellow box** |
| 2 | `(3,3)` | `↑(3)` | **`(3,2)` yellow box** | `left` | `(3,3)` `←(2)` | `(2,3)` empty |
| 3 | `(3,3)` | `←(2)` | `(2,3)` empty | `forward` | `(2,3)` `←(2)` | `(1,3)` empty |
| 4 | `(2,3)` | `←(2)` | `(1,3)` empty | `right` | `(2,3)` `↑(3)` | `(2,2)` empty |
| 5 | `(2,3)` | `↑(3)` | `(2,2)` empty | `forward` | `(2,2)` `↑(3)` | `(2,1)` empty |
| 6 | `(2,2)` | `↑(3)` | `(2,1)` empty | `forward` | `(2,1)` `↑(3)` | `(2,0)` grey wall |
| 7 | `(2,1)` | `↑(3)` | `(2,0)` grey wall | `right` | `(2,1)` `→(0)` | **`(3,1)` blue key (`terminated=True`)** |

### 机制剖析
1. **结束判定语义**：GoToLocal 的成功条件是**站在目标相邻格且朝向目标**（`front_pos == target_pos`），而非踩上目标格（`Key.can_overlap() == False`）。第 7 步在 `(2,1)` 右转面向 `(3,1)` 即刻触发终止。
2. **正确的类型排斥与绕障**：`t=2` 面前正对干扰物 `yellow box`，策略未在其面前停滞，而是左转借道 `x=2` 纵廊北上绕过障碍物。
3. **回报与步长的耦合**：成功仅需 8 步（回报 $1 - 0.9 \times 8/36 = 0.800$），而一旦接地失败陷入振荡则必跑满 36 步（回报 `0.000`）。

---

## 二、五例语言接地失败案例及主修改 B2 FiLM 逐例修复实测

> 归因准则（冻结于 `src/render_episodes.py:classify_failure`）：超时结束时面前正对**同色异类**或**同类异色**干扰物记为 `attribute_confusion`；停在离干扰物距离小于目标距离处记为 `distractor_priority`。

### 案例 1 · `seed 4516` · `go to the purple box`（`attribute_confusion`：同色异类混淆）

- **场景物体**：目标 **`purple box` 在 `(1,3)`**；干扰物 **`purple key` 在 `(3,3)`**（同色异类）、**`grey box` 在 `(2,2)`**（同类异色）。起点 `(4,2)` 朝东 `→(0)`（存档：`reports/failures/seed4516.json`）。
- **基线失败过程（`baseline_seed0`，36 步超时，回报 `0.0`）**：
  - `t=0..2` 执行 `right, right, forward` 走到 `(3,2)` 朝西 `←(2)`，正对 `grey box (2,2)`；
  - `t=3` 左转朝南 `↓(1)` 正对同色干扰物 **`purple key (3,3)`**；随后在 `t=3..35` 连续 33 步原地 `left/right` 交替摆头于 `grey box (2,2)` 与 `purple key (3,3)` 之间直至超时！两个干扰物分别共享目标的「类型（box）」与「颜色（purple）」，Concat 线性加性特征被两侧局部匹配同时吸引而死锁。
  - **辅助支线 B3 对照**：`b3_dagger_seed0` 与 `b3_v2_unstick_seed0` **均同样 36 步超时失败**，证实仅靠交互补状态无法拆解特征层面的属性纠缠。
- **★ 主修改 B2 FiLM 实测结果（已解决 ✅）**：
  - **`runs/b2_film_nwd_seed{0,1,2}`（3/3 全成功）** 及 `b2_film_seed0,2`（2/3 成功）、`b2_film_only_seed0`：**8 步成功，回报 `0.800`**！
  - **精确动作序列（8 步）**：`['right', 'forward', 'forward', 'right', 'forward', 'forward', 'forward', 'right']`。
  - **修复机制轨迹**：`t=0` 在 `(4,2)` 右转朝南 `↓(1)` → `t=1..2` 连走两步至右下角 `(4,4)` → `t=3` 右转朝西 `←(2)` → `t=4..6` 沿南墙底廊 `(4,4)→(3,4)→(2,4)→(1,4)` 径直略过 `(3,3)` 的 `purple key` → `t=7` 在 `(1,4)` 右转朝北 `↑(3)` 精准面向 `(1,3)` 的 **`purple box`**！FiLM 的 `(1+Δγ)` 乘性门控要求通道同时满足 `purple` 与 `box`，彻底消除了半匹配干扰物的伪峰值。

### 案例 2 · `seed 4565` · `go to the yellow box`（`attribute_confusion`：双同色干扰物包围）

- **场景物体**：目标 **`yellow box` 在 `(2,2)`**；场景内同时存在**两个同色异类干扰物 `yellow ball` 分别位于 `(1,2)` 与 `(2,3)`**！起点 `(3,4)` 朝西 `←(2)`（存档：`branches/_shared/lang_grounding_failures/seed4565.json`）。
- **基线失败过程（`baseline_seed0`，36 步超时，回报 `0.0`）**：
  - `t=0` 前进至 `(2,4)`，`t=1` 右转朝北 `↑(3)` 正对眼前的 **`yellow ball (2,3)`**（挡在目标 `(2,2)` 正南方）；`t=2..35` 在 `(2,4)` 原地 `left/right` 振荡 34 步超时（`b3_dagger_seed0` 亦 36 步超时）。
- **★ 主修改 B2 FiLM 实测结果（主配置已解决 ✅）**：
  - **`runs/b2_film_nwd_seed0` 与 `seed2`（2/3 成功）**、`b2_film_seed1`、`b2_film_only_seed0`、`b3_v2_film_dag_seed0`：**4 步近优成功，回报 `0.900`**！
  - **精确动作序列（4 步）**：`['right', 'forward', 'forward', 'left']`。
  - **修复机制轨迹**：在起点 `(3,4) ←(2)` 处，7×7 视野同时看到 `(2,3)` 的 `yellow ball` 与 `(2,2)` 的 `yellow box`。`b2_film_nwd_seed0` 不再像基线那样盲目西进落入 `(2,4)` 陷阱，而是 `t=0` 立即 `right` 转北 `↑(3)` → `t=1..2` 沿畅通的 `x=3` 走廊北上 `(3,4)→(3,3)→(3,2)` 绕开南侧挡路的 `yellow ball (2,3)` → `t=3` 在 `(3,2)` 执行 `left` 转西 `←(2)` 直接面向 `(2,2)` 的 **`yellow box`**！（注：全局 `wd=1e-4` 的 `b2_film_seed0,2` 及 `nwd_seed1` 仍会被双 `yellow ball` 困在 `(1,3)`/`(2,4)`，这也从个案层面印证了窄主干上加全局 L2 会削弱细粒度形状区分度）。

### 案例 3 · `seed 4570` · `go to the grey ball`（`attribute_confusion`：同类异色颜色盲区）

- **场景物体**：目标 **`grey ball` 在 `(4,4)`**；两个**同类异色干扰物** `red ball` 在 `(3,1)`、`blue ball` 在 `(4,3)`。起点 `(1,2)` 朝南 `↓(1)`（存档：`branches/_shared/lang_grounding_failures/seed4570.json`）。
- **基线失败过程（`baseline_seed0`，36 步超时，回报 `0.0`）**：
  - 基线完全验证了反事实探针揭示的「颜色敏感度仅 0.051–0.065」病理：场景里三个物体全是 `ball`，基线在 `t=0..3` 走到 `(3,3)` 朝东正对 `blue ball (4,3)`，接着 `t=4..5` 转北走到 `(3,2)` 正对 **`red ball (3,1)`**，并在 `(3,2)` 原地左右摆头 30 步直到超时——只要是球就停，完全不读 `grey`！
- **★ 主修改 B2 FiLM 实测结果（6/6 全种子一致解决 ✅）**：
  - **`runs/b2_film_nwd_seed{0,1,2}`（3/3）与 `runs/b2_film_seed{0,1,2}`（3/3）以及 `b2_film_only_seed0` 全部以最短 5 步成功，回报 `0.875`！**
  - **精确动作序列（5 步）**：`['forward', 'forward', 'left', 'forward', 'forward']`。
  - **修复机制轨迹**：`t=0..1` 从 `(1,2) ↓(1)` 连走两步直抵南墙 `(1,4)` → `t=2` 左转朝东 `→(0)` → `t=3..4` 沿 `y=4` 底行 `(1,4)→(2,4)→(3,4)` 一步不偏地直达 `(3,4) →(0)`，正对角落 `(4,4)` 的 **`grey ball`**，彻底无视上方近在咫尺的 `blue ball (4,3)` 与 `red ball (3,1)`。

### 案例 4 · `seed 4582` · `go to the blue box`（`attribute_confusion`：相近颜色 `blue` vs `purple` 混淆）

- **场景物体**：目标 **`blue box` 在 `(3,2)`**；干扰物 **`purple box` 在 `(2,2)`**（紧邻目标左侧，同类异色）、`yellow key` 在 `(2,3)`。起点 `(4,4)` 朝南 `↓(1)`（存档：`branches/_shared/lang_grounding_failures/seed4582.json`）。
- **基线失败过程（`baseline_seed0`，36 步超时，回报 `0.0`）**：
  - `t=0` 右转朝西 `←(2)`，随后连走三步穿过 `(3,4)` 直奔西墙 `(1,4)`，再北上至 `(1,2)` 朝东正对 **`purple box (2,2)`** 原地振荡至 36 步超时——尽管 `t=1` 经过 `(3,4)` 时真正的目标 `blue box (3,2)` 就在正北方 2 格处！
- **★ 主修改 B2 FiLM 实测结果（6/6 全种子一致解决 ✅）**：
  - **`runs/b2_film_nwd_seed0,2` 与 `runs/b2_film_seed{0,1,2}`、`b2_film_only_seed0` 全部以最短 4 步成功，回报 `0.900`**（`b2_film_nwd_seed1` 亦以 12 步成功，回报 `0.700`）！
  - **精确动作序列（4 步）**：`['right', 'forward', 'right', 'forward']`。
  - **修复机制轨迹**：`t=0` 在 `(4,4)` 右转朝西 `←(2)` → `t=1` 前进一步至 `(3,4)`，此时 7×7 局部观测内 `(3,2)` 的 `blue box` 与 `(2,2)` 的 `purple box` 并排呈现；FiLM 精确放大 `blue` 通道抑制 `purple` 通道，`t=2` 立即在 `(3,4)` 果断 `right` 转北 `↑(3)` → `t=3` `forward` 至 `(3,3)` 正对 **`blue box (3,2)`** 完成任务！

### 案例 5 · `seed 4560` · `go to the green ball`（`distractor_priority`：近距同类异色干扰物抢占注意力）

- **场景物体**：目标 **`green ball` 在 `(4,1)`**；近距同类异色干扰物 **`grey ball` 在 `(3,1)`**，`purple key` 在 `(1,2)`。起点 `(4,3)` 朝西 `←(2)`（存档：`reports/failures/seed4560.json`）。
- **基线失败过程（`baseline_seed0`，36 步超时，回报 `0.0`）**：
  - 起点 `(4,3)` 朝西时，正北方向 `(4,1)` 的 `green ball` 其实仅距 2 格；但基线在 `t=0` 顺着初始朝向西进至 `(3,3)`，`t=1..2` 右转北进至 `(3,2)` 正对 **`grey ball (3,1)`**，随后被这个距离仅 1 格的同类球锁定，在 `(3,2)` 原地 `left/right` 摆头 33 步直至超时。
- **★ 主修改 B2 FiLM 实测结果（6/6 全种子一致解决 ✅）**：
  - **`runs/b2_film_nwd_seed{0,1,2}`（3/3）与 `runs/b2_film_seed0,1`、`b2_film_only_seed0` 全部以理论最短 2 步成功，回报 `0.950`**（`b2_film_seed2` 以 6 步成功，回报 `0.850`）！
  - **精确动作序列（2 步）**：`['right', 'forward']`。
  - **修复机制轨迹**：`t=0` 在起点 `(4,3) ←(2)` 处，7×7 视野右侧已包含 `(4,1)` 的 `green ball` 与 `(3,1)` 的 `grey ball`；FiLM 在第 0 步就直接识别出右侧的绿色球才是目标，当即执行 `right` 转北 `↑(3)` → `t=1` 执行 `forward` 进至 `(4,2) ↑(3)` 正对 **`green ball (4,1)`**，2 步秒解！

---

## 三、补充核验：`reports/failures/` 其余三例状态覆盖失败与 B2 FiLM 的能力边界

为做到完整无遗漏，我们在同一评估环境下核验了冻结目录 `reports/failures/` 中归入状态覆盖类的其余 3 个案例（`4630, 4716, 4568`），结果揭示了极具启发性的机制分野：

1. **`seed 4630` · `go to the grey key`（原归类 `approach_then_lost`，B2 FiLM 6/6 全解决 ✅）**：
   - **场景**：目标 `grey key` 在 `(4,2)`，干扰物 `grey box (3,3)`、`purple key (3,4)`，起点 `(3,1) ↑(3)`。基线（36 步）与 `b3_dagger_seed0` / `b3_v2_unstick_seed0`（均 36 步）都在西侧走廊 `(2,1)..(2,3)` 迷失打转。
   - **B2 FiLM 实测**：`b2_film_nwd_seed{0,1,2}` 与 `b2_film_seed{0,1,2}` **6/6 全部仅用 4 步成功（回报 `0.900`）**，动作 `['left', 'left', 'forward', 'left']`：从 `(3,1) ↑` 连转两次朝南 `↓` → 前进至 `(3,2)` → 左转朝东 `→` 直接面向 `grey key (4,2)`！说明许多表面上的「路过未停」根源仍是目标特征不够显著。
2. **`seed 4716` · `go to the yellow key`（原归类 `orientation_lock`，B2 FiLM 主配置 3/3 全解决 ✅）**：
   - **场景**：目标 `yellow key` 在 `(2,3)`，干扰物 `red ball (4,3)`、`purple ball (1,3)`，起点 `(3,2) ↓(1)`。基线在 `t=0` 前进至 `(3,3)` 后，`t=1` **错误地 `left` 转东面向 `red ball (4,3)`** 并卡死 34 步。
   - **B2 FiLM 实测**：`b2_film_nwd_seed{0,1,2}`（3/3）及 `b2_film_seed0,1` **仅用 2 步成功（回报 `0.950`）**，动作 `['forward', 'right']`：`t=0` 进至 `(3,3)` 后，`t=1` 精准 `right` 转西面向 `yellow key (2,3)`！
3. **`seed 4568` · `go to the green ball`（`turn_oscillation` —— ★ B2 FiLM 的真实失败案例剖析 ❌）**：
   - **场景与视野盲区验证**：目标 `green ball` 在东北角 **`(4,1)`**，干扰物 `blue box (1,1)`、`purple key (1,3)`，智能体起点在 **`(3,2)` 朝西 `←(2)`**。由于 MiniGrid 的 7×7 观测是以智能体为底边中点的**前向扇区**，实测验证：
     - 当站在 `(3,2)` 朝西 `←(2)` 时，视野内仅有 `purple key` 与 `blue box`，**身后的 `(4,1)` `green ball` 完全不可见**；
     - 当 `t=0` 执行 `left` 转南 `↓(1)` 后，东北侧 `(4,1)` 落在智能体左后方一格，视野内仅剩 `purple key`，**`green ball` 依然处于 7×7 视野盲区之外**！
   - **B2 FiLM 为什么在此失败**：`b2_film_nwd_seed{0,1,2}`（3/3 全部 36 步超时，也是 `b2_film_nwd_seed0` 全测试集 300 局中仅有的 11 局失败之一）在 `(3,2) ←(2)` 执行 `left` 到 `(3,2) ↓(1)`，而在 `(3,2) ↓(1)` 盲区观测下贪心策略又输出 `right` 回到 `(3,2) ←(2)`，陷入二周期无记忆极限环 `['left', 'right', 'left', 'right', ...]`。
   - **谁能修它（正交互补的直接物证）**：由于该失败源于**部分可观测盲区下的无记忆确定性死锁**而非视觉-语言接地错误（视野里根本没有目标可调制！），交互式纠偏 **`b3_dagger_seed0` / `b3_v2_unstick_seed0`（8 步成功，回报 `0.800`）** 与正交组合模型 **`runs/b3_v2_film_dag_seed0`（FiLM + DAgger，4 步成功 `['left', 'left', 'forward', 'left']`，回报 `0.900`）** 均顺利打破死锁！这从单轨迹机制上完美解释了为什么 `B2 FiLM` 能把表征失败从 `38.3` 压到 `6.3`，但仍残留 `6.0` 局状态覆盖失败，以及为什么叠加 `DAgger` 能进一步把总失败降到 `9.0`。

---

## 四、全部 9 个核验案例对照总表

| Seed | 指令 (`mission`) | 基线分类 (`baseline_seed0`) | 基线结果 | **主修改 `b2_film_nwd_seed0`**（及 3-seed 通过数） | `b2_film_seed0` (`wd=1e-4`, 及 3-seed) | 辅助 `b3_dagger_seed0` | 正交 `FiLM+DAgger` (`seed0`) | 因果结论 |
|---|---|---|---|---|---|---|---|---|
| `4501` | `go to the blue key` | 成功对照 | ✅ 8 步 (`0.80`) | **✅ 8 步 (`0.80`)** [2/3] | ❌ 36 步 [1/3] | ✅ 8 步 | ✅ 8 步 | 保持同路径绕障行为 |
| `4516` | `go to the purple box` | `attribute_confusion`（同色异类） | ❌ 36 步 | **✅ 8 步 (`0.80`)** [3/3] | ✅ 8 步 [2/3] | ❌ 36 步 | ❌ 36 步 | **FiLM 解开颜色/类型双半匹配死锁** |
| `4565` | `go to the yellow box` | `attribute_confusion`（双同色球） | ❌ 36 步 | **✅ 4 步 (`0.90`)** [2/3] | ❌ 36 步 [1/3] | ❌ 36 步 | ✅ 4 步 | **FiLM 绕开挡路同色球直取目标** |
| `4570` | `go to the grey ball` | `attribute_confusion`（三异色球） | ❌ 36 步 | **✅ 5 步 (`0.875`)** [3/3] | ✅ 5 步 [3/3] | ✅ 7 步 | ✅ 5 步 | **FiLM 彻底消除基线颜色盲区** |
| `4582` | `go to the blue box` | `attribute_confusion`（蓝/紫近似色） | ❌ 36 步 | **✅ 4 步 (`0.90`)** [3/3] | ✅ 4 步 [3/3] | ✅ 8 步 | ✅ 4 步 | **FiLM 在 1 像素级精准区分蓝/紫框** |
| `4560` | `go to the green ball` | `distractor_priority`（近距灰球） | ❌ 36 步 | **✅ 2 步 (`0.95`)** [3/3] | ✅ 2 步 [3/3] | ✅ 6 步 | ✅ 2 步 | **FiLM 抑制「最近物体优先」捷径** |
| `4630` | `go to the grey key` | `approach_then_lost` | ❌ 36 步 | **✅ 4 步 (`0.90`)** [3/3] | ✅ 4 步 [3/3] | ❌ 36 步 | ✅ 4 步 | 强化目标显著度后直接转向命中 |
| `4716` | `go to the yellow key` | `orientation_lock` | ❌ 36 步 | **✅ 2 步 (`0.95`)** [3/3] | ✅ 2 步 [2/3] | ✅ 2 步 | ✅ 2 步 | 消除侧方红球诱偏，一步转对方向 |
| `4568` | `go to the green ball` | `turn_oscillation`（背后盲区） | ❌ 36 步 | **❌ 36 步 (`0.00`)** [0/3] | ✅ 4 步 [1/3] | **✅ 8 步** | **✅ 4 步** | **盲区无目标属记忆/覆盖边界，由 DAgger 互补修复** |

---

## 五、复现与逐例核验命令

```bash
source env_embodied/bin/activate
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy

# 1. 基线冻结案例存档位置（只读）
ls reports/failures/seed*.json branches/_shared/lang_grounding_failures/seed*.json

# 2. 只读运行 B2 FiLM 主模型对比上述全部 9 个测试 seed 的逐步轨迹（不改写任何冻结文件）
python - <<'PY'
import sys
sys.path.insert(0, "branches/b2_film"); sys.path.insert(0, "branches/_shared"); sys.path.insert(0, "src")
import gymnasium as gym, minigrid
from film_policy import load_any_run, make_policy_fn

ACTS = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
env = gym.make("BabyAI-GoToLocalS6N3-v0")
for run in ["runs/baseline_seed0", "runs/b2_film_nwd_seed0", "runs/b2_film_seed0"]:
    m, _, _ = load_any_run(run); pol = make_policy_fn(m)
    print(f"=== {run} ===")
    for s in [4501, 4516, 4565, 4570, 4582, 4560, 4630, 4716, 4568]:
        obs, _ = env.reset(seed=s); acts = []; term = trunc = False
        while not (term or trunc):
            a = int(pol(obs)); acts.append(ACTS[a])
            obs, _, term, trunc, _ = env.step(a)
        print(f"  seed {s} ({obs['mission']}): {'SUCCESS' if term else 'TIMEOUT'} in {len(acts)} steps | {acts[:10]}")
env.close()
PY
```
