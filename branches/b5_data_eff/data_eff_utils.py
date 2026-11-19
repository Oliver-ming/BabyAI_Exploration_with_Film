"""支线 5 · 数据效率（Data Efficiency）：等采样子集划分、精确 D4 群对称增广与消融配置表。

核心设计与文献依据：
  1. 严格复刻 `src/train_bc.py` 按 episode 分层随机抽样（`random.Random(seed).sample`），
     确保 5%/10%/20%/50%/100% 五档与全量 TEST_SEEDS 零泄漏（`common.assert_no_leak`）。
  2. 步数匹配对照（Step-Matched Baseline）：小数据集 naive 训练不仅样本少，梯度更新步数也同比例萎缩
     （5% 时仅 210 步 vs 100% 的 3900 步）。通过 `RandomSampler(num_samples=16534)` 对齐总更新步数，
     把「欠训练步数不足」与「真过拟合/泛化缺口」彻底分离。
  3. 精确 D4（8 阶二面体群）MDP 同态增广（van der Pol et al., NeurIPS 2020）：
     经 `verify_d4_against_bot` 在 50 个实时 MiniGrid+BabyAIBot 环境逐帧验证 100% 等价：
       - Z2 镜像翻转（`imgs[:, ::-1, :, :]` + 动作 `0(left)↔1(right)`，全局 dir 无需变更因为 obs 已以朝向对齐）
       - C4 指南针旋转（全局世界连同 agent_dir 旋转 k×90° 时，以自我为中心的 obs["image"] 与最佳相对动作
         均严格不变，仅全局 `direction = (direction + k) % 4`）
     两者直积共构成 8 倍零误差对称数据池，且因 GoToLocal 指令不含「左边/右边」空间方位词而完全语义保真。
"""

from __future__ import annotations

import copy
import pickle
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import TRAIN_SEEDS, assert_no_leak, split_of  # noqa: E402

FRACS = (0.05, 0.10, 0.20, 0.50, 1.00)
N_REF_TRANSITIONS = 16534  # 4500 条 S6N3 官方训练集总转移数


def frac_tag(frac: float) -> str:
    return f"f{int(round(frac * 100)):03d}"


def _base_cfg(
    *,
    arch: str = "bcnet",
    step_match: bool = True,
    aug: str = "none",
    dropout: float = 0.0,
    label_smoothing: float = 0.0,
    cosine_lr: bool = False,
    weight_decay: float = 0.0,
    act_balance: bool = False,
    family: str = "scaling",
    desc: str = "",
) -> dict:
    return {
        "arch": arch,
        "step_match": step_match,
        "aug": aug,
        "dropout": dropout,
        "label_smoothing": label_smoothing,
        "cosine_lr": cosine_lr,
        "weight_decay": weight_decay,
        "act_balance": act_balance,
        "family": family,
        "desc": desc,
    }


METHODS: dict[str, dict] = {
    # 6 条完整扩展曲线（5% / 10% / 20% / 50% / 100%）
    "bc_naive": _base_cfg(
        arch="bcnet",
        step_match=False,
        desc="无步数匹配朴素 BC（忠实反映预算减少同时导致的梯度步萎缩）",
    ),
    "bc_step": _base_cfg(
        arch="bcnet",
        step_match=True,
        desc="步数匹配 Concat BC（每 epoch 均等 16,534 步，剥离欠拟合）",
    ),
    "bc_reg": _base_cfg(
        arch="bcnet",
        step_match=True,
        dropout=0.2,
        label_smoothing=0.05,
        cosine_lr=True,
        desc="正则化 BC（Dropout=0.2 + 标签平滑 0.05 + CosineLR）",
    ),
    "bc_d4": _base_cfg(
        arch="bcnet",
        step_match=True,
        aug="d4",
        cosine_lr=True,
        desc="D4 8x MDP 群对称增广 + Concat BCNet + CosineLR",
    ),
    "film_step": _base_cfg(
        arch="film",
        step_match=True,
        desc="步数匹配 FiLMBCNet（wd=0, film_l2=0）",
    ),
    "film_d4": _base_cfg(
        arch="film",
        step_match=True,
        aug="d4",
        dropout=0.1,
        cosine_lr=True,
        desc="D4 8x MDP 群对称增广 + FiLMBCNet + Dropout=0.1 + CosineLR",
    ),
    # 5 项 20%（900 ep）机制分解消融
    "bc_z2": _base_cfg(
        arch="bcnet",
        step_match=True,
        aug="z2",
        cosine_lr=True,
        family="ablation_f020",
        desc="仅左右镜像 Z2 (2x) + 动作交换 + CosineLR",
    ),
    "bc_c4": _base_cfg(
        arch="bcnet",
        step_match=True,
        aug="c4",
        cosine_lr=True,
        family="ablation_f020",
        desc="仅指南针方向 C4 (4x) 等变增广 + CosineLR",
    ),
    "bc_wd": _base_cfg(
        arch="bcnet",
        step_match=True,
        weight_decay=1e-4,
        family="ablation_f020",
        desc="负对照：全局 weight_decay=1e-4（检验窄 trunk 正则破坏效应）",
    ),
    "bc_bal": _base_cfg(
        arch="bcnet",
        step_match=True,
        act_balance=True,
        family="ablation_f020",
        desc="动作类平方根逆频率加权（抑制 57% forward 多数类偏置）",
    ),
    "gru_small_step": _base_cfg(
        arch="gru_small",
        step_match=True,
        family="ablation_f020",
        desc="步数匹配 GRU-small 语言编码器（检验低数据语言归纳偏置）",
    ),
}


def load_subset_episodes(data_path: str | Path, frac: float, seed: int):
    """按 episode 与 src/train_bc.py 完全相同口径划分训练子集 + 全量验证集。"""
    all_eps = pickle.load(open(data_path, "rb"))
    train_eps = [e for e in all_eps if e.get("split", split_of(e["seed"])) == "train"]
    val_eps = [e for e in all_eps if e.get("split", split_of(e["seed"])) == "val"]
    assert_no_leak(e["seed"] for e in train_eps + val_eps)

    if frac < 1.0:
        n_keep = max(1, int(round(len(train_eps) * frac)))
        rng = random.Random(seed)
        train_eps = rng.sample(train_eps, n_keep)
    return train_eps, val_eps


def apply_z2_mirror(
    imgs: np.ndarray, toks: np.ndarray, dirs: np.ndarray, acts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """2× 左右镜像群对称：翻转局部视野左右轴（axis 1），并互换 left(0) <-> right(1)。"""
    imgs_m = imgs[:, ::-1, :, :].copy()
    acts_m = acts.copy()
    left_mask = acts == 0
    right_mask = acts == 1
    acts_m[left_mask] = 1
    acts_m[right_mask] = 0
    return (
        np.concatenate([imgs, imgs_m], axis=0),
        np.concatenate([toks, toks.copy()], axis=0),
        np.concatenate([dirs, dirs.copy()], axis=0),
        np.concatenate([acts, acts_m], axis=0),
    )


def apply_c4_compass(
    imgs: np.ndarray, toks: np.ndarray, dirs: np.ndarray, acts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """4× 旋转对称群：全局旋转 k×90° 下局部视野与相对动作严格不变，仅 direction=(d+k)%4。"""
    out_img, out_tok, out_dir, out_act = [], [], [], []
    for k in range(4):
        out_img.append(imgs if k == 0 else imgs.copy())
        out_tok.append(toks if k == 0 else toks.copy())
        out_dir.append((dirs + k) % 4)
        out_act.append(acts if k == 0 else acts.copy())
    return (
        np.concatenate(out_img, axis=0),
        np.concatenate(out_tok, axis=0),
        np.concatenate(out_dir, axis=0),
        np.concatenate(out_act, axis=0),
    )


def apply_d4_symmetry(
    imgs: np.ndarray, toks: np.ndarray, dirs: np.ndarray, acts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """8× D4 群对称增广 = Z2 镜像 × C4 指南针旋转。"""
    return apply_c4_compass(*apply_z2_mirror(imgs, toks, dirs, acts))


def augment_arrays(
    aug: str, imgs: np.ndarray, toks: np.ndarray, dirs: np.ndarray, acts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if aug == "none":
        return imgs, toks, dirs, acts
    if aug == "z2":
        return apply_z2_mirror(imgs, toks, dirs, acts)
    if aug == "c4":
        return apply_c4_compass(imgs, toks, dirs, acts)
    if aug == "d4":
        return apply_d4_symmetry(imgs, toks, dirs, acts)
    raise ValueError(f"未知 aug={aug}")


def verify_d4_against_bot(n_seeds: int = 20) -> None:
    """用真实 BabyAI 环境与 BabyAIBot 对 20 个随机场景做 Z2 与 C4 等变性自检断言。"""
    import gymnasium as gym
    import minigrid
    from minigrid.utils.baby_ai_bot import BabyAIBot

    env = gym.make("BabyAI-GoToLocalS6N3-v0")
    env_t = gym.make("BabyAI-GoToLocalS6N3-v0")

    for seed in TRAIN_SEEDS[:n_seeds]:
        obs, _ = env.reset(seed=seed)
        u = env.unwrapped
        W, H = u.grid.width, u.grid.height
        a_orig = int(BabyAIBot(u).replan())

        # 验证 Z2 左右翻转（世界 x -> W-1-x, dir 0<->2）
        env_t.reset(seed=seed)
        ut = env_t.unwrapped
        g_m = minigrid.core.grid.Grid(W, H)
        for x in range(W):
            for y in range(H):
                g_m.set(W - 1 - x, y, copy.deepcopy(u.grid.get(x, y)))
        ut.grid = g_m
        ut.agent_pos = (W - 1 - int(u.agent_pos[0]), int(u.agent_pos[1]))
        ut.agent_dir = {0: 2, 2: 0, 1: 1, 3: 3}[int(u.agent_dir)]
        if hasattr(ut.instrs, "desc") and ut.instrs.desc.obj_poss:
            ut.instrs.desc.obj_poss = [(W - 1 - int(px), int(py)) for (px, py) in u.instrs.desc.obj_poss]
        obs_m = ut.gen_obs()
        img_batch = obs["image"][None, ...]
        tok_batch = np.zeros((1, 5), dtype=np.int64)
        dir_batch = np.array([int(obs["direction"])], dtype=np.int64)
        act_batch = np.array([a_orig], dtype=np.int64)
        zm_img, _, _, zm_act = apply_z2_mirror(img_batch, tok_batch, dir_batch, act_batch)
        a_m = int(BabyAIBot(ut).replan())
        assert np.array_equal(obs_m["image"], zm_img[1]), f"Z2 image mismatch at seed={seed}"
        assert a_m == int(zm_act[1]), f"Z2 action mismatch at seed={seed}: {a_m} vs {zm_act[1]}"

        # 验证 C4 旋转 90°（世界 (x,y)->(H-1-y,x), dir -> (dir+1)%4）
        env_t.reset(seed=seed)
        ur = env_t.unwrapped
        g_r = minigrid.core.grid.Grid(W, H)
        for x in range(W):
            for y in range(H):
                g_r.set(H - 1 - y, x, copy.deepcopy(u.grid.get(x, y)))
        ur.grid = g_r
        ur.agent_pos = (H - 1 - int(u.agent_pos[1]), int(u.agent_pos[0]))
        ur.agent_dir = (int(u.agent_dir) + 1) % 4
        if hasattr(ur.instrs, "desc") and ur.instrs.desc.obj_poss:
            ur.instrs.desc.obj_poss = [(H - 1 - int(py), int(px)) for (px, py) in u.instrs.desc.obj_poss]
        obs_r = ur.gen_obs()
        c4_img, _, c4_dir, c4_act = apply_c4_compass(img_batch, tok_batch, dir_batch, act_batch)
        a_r = int(BabyAIBot(ur).replan())
        assert np.array_equal(obs_r["image"], c4_img[1]), f"C4 image mismatch at seed={seed}"
        assert int(obs_r["direction"]) == int(c4_dir[1]), f"C4 dir mismatch at seed={seed}"
        assert a_r == int(c4_act[1]), f"C4 action mismatch at seed={seed}"

    env.close()
    env_t.close()
