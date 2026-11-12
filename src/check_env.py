"""P0-1 · 环境验收（指导文档 §3.3）。

用法:
    python src/check_env.py

校验：
  - obs["image"] 形状 (7,7,3) uint8、action_space.n == 7、max_steps == 36
  - BabyAIBot 能给出第一个动作
  - 纯随机的 3 条 episode 统计（自检 step 语义）
  - 关键依赖版本是否与 requirements-lock.txt 的核心 pin 一致
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

import minigrid  # noqa: F401  (注册 BabyAI-*)
from minigrid.utils.baby_ai_bot import BabyAIBot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENV_ID, ENV_ID_S7, MAX_STEPS_S6N3, MAX_STEPS_S7N4  # noqa: E402

EXPECTED = {
    "torch": "2.15.0",
    "gymnasium": "1.4.0",
    "minigrid": "3.2.0",
    "numpy": "2.5.4",
}


def main() -> None:
    ok = True

    # ---------- §3.3 验收 ----------
    env = gym.make(ENV_ID)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=0)
    u = env.unwrapped

    checks = [
        (obs["image"].shape == (7, 7, 3), f'obs["image"].shape == (7,7,3)（实际 {obs["image"].shape}）'),
        (obs["image"].dtype.name == "uint8", f'obs["image"].dtype == uint8（实际 {obs["image"].dtype}）'),
        (env.action_space.n == 7, f"action_space.n == 7（实际 {env.action_space.n}）"),
        (u.max_steps == MAX_STEPS_S6N3, f"max_steps == {MAX_STEPS_S6N3}（实际 {u.max_steps}）"),
        (isinstance(obs["mission"], str), f'obs["mission"] 是 str（实际 {type(obs["mission"])}）'),
        ("direction" in obs, "obs 含 direction 字段"),
    ]

    bot = BabyAIBot(u)
    a = bot.replan(None)
    checks.append((a is not None, f"BabyAIBot 给出首个动作（{a}）"))

    for good, desc in checks:
        print(f"[{'PASS' if good else 'FAIL'}] {desc}")
        ok &= good

    print(f"\nPASS {obs['mission']} | first expert action: {a}")

    # ---------- S7N4 也验一遍 ----------
    env7 = gym.make(ENV_ID_S7)
    with contextlib.redirect_stdout(io.StringIO()):
        obs7, _ = env7.reset(seed=10000)
    u7 = env7.unwrapped
    checks7 = [
        (obs7["image"].shape == (7, 7, 3), f"S7N4 obs shape (7,7,3)（实际 {obs7['image'].shape}）"),
        (u7.max_steps == MAX_STEPS_S7N4, f"S7N4 max_steps == {MAX_STEPS_S7N4}（实际 {u7.max_steps}）"),
    ]
    for good, desc in checks7:
        print(f"[{'PASS' if good else 'FAIL'}] {desc}")
        ok &= good
    env7.close()

    # ---------- 版本核对 ----------
    import torch
    import gymnasium as g
    import minigrid as m
    import numpy as np2

    actual = {
        "torch": torch.__version__.split("+")[0],
        "gymnasium": g.__version__,
        "minigrid": m.__version__,
        "numpy": np2.__version__,
    }
    print("\n依赖版本：")
    for k, want in EXPECTED.items():
        got = actual[k]
        good = got == want
        print(f"[{'PASS' if good else 'WARN'}] {k}: {got}（pin {want}）")
        ok &= good
    print(f"       torch.cuda.is_available() = {torch.cuda.is_available()}")

    # ---------- 自检：随机策略跑 3 条 ----------
    print("\n自检（均匀随机策略，3 条 episode）：")
    env2 = gym.make(ENV_ID)
    rng = np.random.default_rng(0)
    lines = []
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in (4500, 4501, 4502):
            obs, _ = env2.reset(seed=seed)
            steps, te, tr = 0, False, False
            while not (te or tr):
                obs, r, te, tr, _ = env2.step(int(rng.integers(0, 3)))
                steps += 1
            lines.append(f"  seed={seed} steps={steps} terminated={te} truncated={tr}")
    env2.close()
    env.close()
    for line in lines:
        print(line)

    print("\n" + ("=" * 60))
    print("P0-1 验收通过 ✅" if ok else "P0-1 验收未通过 ❌（见上方 FAIL/WARN）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
