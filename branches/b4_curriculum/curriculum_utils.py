"""支线 4 · 课程学习难度评分、13 组等预算数据集构造与阶段调度器。

所有变体严格遵守三条协议铁律：
  1. 总专家训练池严格等于 4,500 条唯一 seed episode（来自 TRAIN_SEEDS[0..4499]），
     加 500 条冻结 S6N3 验证集 = 5,000 专家轨迹预算上限；绝不接触 TEST_SEEDS / TEST_SEEDS_S7。
  2. 同一多环境混合池内切分互不相交的 TRAIN_SEEDS 切片，保证 4,500 个 seed 互不重复。
  3. 当某个课程阶段暴露的子集样本数少于全池转移数时，使用 RandomSampler(replacement=True,
     num_samples=N_full) 保持每个 epoch 梯度步数严格相等，剥离「少练了几步」的混杂因素。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from common import assert_no_leak  # noqa: E402

OBJECT_TO_IDX = {"key": 5, "ball": 6, "box": 7}
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}

VARIANTS = (
    # Block A: 同域 S6N3 样本级排序课程（BCNet Concat）
    "pace_easy",
    "pace_hard",
    "pace_rand",
    # Block B: 跨环境课程学习 vs 等数据均匀混合域随机化（BCNet Concat）
    "env_easy_stage",
    "env_easy_cumul",
    "env_easy_mix",
    "env_s6n4_prog",
    "env_s6n4_mix",
    "env_multiscale_prog",
    "env_multiscale_mix",
    # Block C: 架构 × 课程交互（FiLMBCNet, wd=0, film_l2=0）
    "film_easy_stage",
    "film_multiscale_prog",
    "film_multiscale_mix",
)

FILM_VARIANTS = ("film_easy_stage", "film_multiscale_prog", "film_multiscale_mix")


def score_episode_difficulty(ep: dict) -> float:
    """仅利用训练片段自身观测与专家动作计算的零泄漏固有难度分。

    考量四个机制维度的困难来源：
      - 可混淆干扰物密度（7x7 局部视野单帧最多同时出现的同色/同类非目标物体数，截断于 3）
      - 是否为唯一目标 `the`（相比 `a`，多目标容错降为 1 个真目标）
      - 最优专家轨迹长度 `episode_len`
      - 转向次数 `n_turns`（非纯直行）
    """
    parts = ep["mission"].split()
    article, color_str, type_str = parts[2], parts[-2], parts[-1]
    t_type = OBJECT_TO_IDX[type_str]
    t_color = COLOR_TO_IDX[color_str]

    imgs = np.asarray(ep["images"], dtype=np.uint8)
    obj = imgs[..., 0]
    col = imgs[..., 1]
    is_item = (obj == 5) | (obj == 6) | (obj == 7)
    is_target = (obj == t_type) & (col == t_color)
    is_confusable = is_item & (~is_target) & ((obj == t_type) | (col == t_color))
    max_conf = int(is_confusable.sum(axis=(1, 2)).max()) if len(imgs) > 0 else 0

    is_the = 1.0 if article == "the" else 0.0
    ep_len = float(len(ep["actions"]))
    n_turns = float(sum(1 for a in ep["actions"] if a in (0, 1)))
    return float(2.0 * min(3, max_conf) + 1.5 * is_the + 0.4 * ep_len + 0.3 * n_turns)


def _tag_env(eps: list[dict], default_env: str) -> list[dict]:
    out = []
    for ep in eps:
        c = dict(ep)
        c.setdefault("env_id", default_env)
        out.append(c)
    return out


def _load_pools(data_dir: str | Path) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    d = Path(data_dir)
    all_official = pickle.load(open(d / "expert_5000.pkl", "rb"))
    s6n3_train = _tag_env([e for e in all_official if e.get("split") == "train"], "BabyAI-GoToLocalS6N3-v0")
    s6n3_val = _tag_env([e for e in all_official if e.get("split") == "val"], "BabyAI-GoToLocalS6N3-v0")
    aux = {}
    for tag, default_eid in [
        ("objs6", "BabyAI-GoToObjS6-v1"),
        ("s5n2", "BabyAI-GoToLocalS5N2-v0"),
        ("s6n4", "BabyAI-GoToLocalS6N4-v0"),
        ("s8n6", "BabyAI-GoToLocalS8N6-v0"),
    ]:
        p = d / f"b4_curriculum_{tag}_2250.pkl"
        if p.exists():
            aux[tag] = _tag_env(pickle.load(open(p, "rb")), default_eid)
    return s6n3_train, s6n3_val, aux


def build_curriculum(
    variant: str,
    seed: int = 0,
    data_dir: str | Path = "data",
):
    """返回 (pool_eps, s6n3_val, schedule_fn, meta)。

    schedule_fn(epoch: int) -> tuple[str, list[int]] 返回 (阶段名称, 暴露的 pool_eps 下标列表)。
    """
    if variant not in VARIANTS:
        raise ValueError(f"未知 variant={variant}，可选 {VARIANTS}")

    s6n3_train, s6n3_val, aux = _load_pools(data_dir)
    assert len(s6n3_train) == 4500

    if variant in ("pace_easy", "pace_hard", "pace_rand"):
        pool_eps = s6n3_train
        n = len(pool_eps)
        n40 = round(0.40 * n)  # 1800
        n70 = round(0.70 * n)  # 3150

        if variant == "pace_easy":
            scores = [(score_episode_difficulty(ep), i) for i, ep in enumerate(pool_eps)]
            scores.sort(key=lambda x: (x[0], x[1]))
            order = [i for _, i in scores]
            stage_desc = "ep1-10: easiest 40% (1800); ep11-20: easiest 70% (3150); ep21-30: all 100% (4500)"
            score_stats = {
                "easy40_score_max": round(scores[n40 - 1][0], 3),
                "mid70_score_max": round(scores[n70 - 1][0], 3),
                "all_score_range": [round(scores[0][0], 3), round(scores[-1][0], 3)],
            }
        elif variant == "pace_hard":
            scores = [(score_episode_difficulty(ep), i) for i, ep in enumerate(pool_eps)]
            scores.sort(key=lambda x: (-x[0], x[1]))
            order = [i for _, i in scores]
            stage_desc = "ep1-10: hardest 40% (1800); ep11-20: hardest 70% (3150); ep21-30: all 100% (4500)"
            score_stats = {
                "hard40_score_min": round(scores[n40 - 1][0], 3),
                "hard70_score_min": round(scores[n70 - 1][0], 3),
                "all_score_range": [round(scores[-1][0], 3), round(scores[0][0], 3)],
            }
        else:
            rng = np.random.RandomState(1000 + seed)
            order = rng.permutation(n).tolist()
            stage_desc = "ep1-10: random 40% (1800); ep11-20: nested random 70% (3150); ep21-30: all 100% (4500)"
            score_stats = {"random_perm_seed": 1000 + seed}

        s1 = order[:n40]
        s2 = order[:n70]
        s3 = list(range(n))

        def schedule_fn(epoch: int) -> tuple[str, list[int]]:
            if epoch <= 10:
                return f"{variant}_stage1_40pct", s1
            if epoch <= 20:
                return f"{variant}_stage2_70pct", s2
            return f"{variant}_stage3_100pct", s3

        meta = {
            "block": "A_sample_pacing",
            "curriculum_family": "within_s6n3_pacing",
            "stage_desc": stage_desc,
            "score_stats": score_stats,
        }

    elif variant in ("env_easy_stage", "film_easy_stage"):
        # [0:1500] S5N2 + [1500:4500] S6N3
        pool_eps = aux["s5n2"][:1500] + s6n3_train[1500:4500]
        idx_s5 = list(range(0, 1500))
        idx_s6 = list(range(1500, 4500))

        def schedule_fn(epoch: int) -> tuple[str, list[int]]:
            if epoch <= 10:
                return "stage1_s5n2_only", idx_s5
            return "stage2_s6n3_only", idx_s6

        meta = {
            "block": "C_film_curriculum" if variant in FILM_VARIANTS else "B_cross_env",
            "curriculum_family": "s5n2_to_s6n3_sequential",
            "stage_desc": "ep1-10: S5N2 only (1500); ep11-30: S6N3 only (3000) [non-cumulative]",
        }

    elif variant in ("env_easy_cumul", "env_easy_mix"):
        # [0:750] ObjS6 + [750:1500] S5N2 + [1500:4500] S6N3
        pool_eps = aux["objs6"][:750] + aux["s5n2"][750:1500] + s6n3_train[1500:4500]
        idx_1 = list(range(0, 750))
        idx_2 = list(range(0, 1500))
        idx_all = list(range(0, 4500))

        if variant == "env_easy_cumul":
            def schedule_fn(epoch: int) -> tuple[str, list[int]]:
                if epoch <= 6:
                    return "stage1_objs6", idx_1
                if epoch <= 14:
                    return "stage2_objs6_s5n2", idx_2
                return "stage3_all_easy_to_s6n3", idx_all
            stage_desc = "ep1-6: ObjS6 (750); ep7-14: ObjS6+S5N2 (1500); ep15-30: all 4500"
        else:
            def schedule_fn(epoch: int) -> tuple[str, list[int]]:
                return "uniform_mix_easy_s6n3", idx_all
            stage_desc = "ep1-30: uniform mix of ObjS6 (750) + S5N2 (750) + S6N3 (3000)"

        meta = {
            "block": "B_cross_env",
            "curriculum_family": "objs6_s5n2_s6n3",
            "stage_desc": stage_desc,
        }

    elif variant in ("env_s6n4_prog", "env_s6n4_mix"):
        # [0:1500] S6N4 + [1500:4500] S6N3
        pool_eps = aux["s6n4"][:1500] + s6n3_train[1500:4500]
        idx_s6n3 = list(range(1500, 4500))
        idx_all = list(range(0, 4500))

        if variant == "env_s6n4_prog":
            def schedule_fn(epoch: int) -> tuple[str, list[int]]:
                if epoch <= 12:
                    return "stage1_s6n3_base", idx_s6n3
                return "stage2_s6n3_plus_s6n4", idx_all
            stage_desc = "ep1-12: S6N3 (3000); ep13-30: S6N3 (3000) + S6N4 (1500)"
        else:
            def schedule_fn(epoch: int) -> tuple[str, list[int]]:
                return "uniform_mix_s6n3_s6n4", idx_all
            stage_desc = "ep1-30: uniform mix of S6N3 (3000) + S6N4 (1500)"

        meta = {
            "block": "B_cross_env",
            "curriculum_family": "s6n3_plus_s6n4",
            "stage_desc": stage_desc,
        }

    elif variant in ("env_multiscale_prog", "env_multiscale_mix", "film_multiscale_prog", "film_multiscale_mix"):
        # [0:750] S5N2 + [750:1500] S6N4 + [1500:2250] S8N6 + [2250:4500] S6N3
        pool_eps = (
            aux["s5n2"][:750]
            + aux["s6n4"][750:1500]
            + aux["s8n6"][1500:2250]
            + s6n3_train[2250:4500]
        )
        # stage 1: S5N2 (0:750) + S6N3 (2250:4500) = 3000 eps
        idx_s1 = list(range(0, 750)) + list(range(2250, 4500))
        idx_all = list(range(0, 4500))

        if variant.endswith("_prog"):
            def schedule_fn(epoch: int) -> tuple[str, list[int]]:
                if epoch <= 12:
                    return "stage1_s5n2_s6n3", idx_s1
                return "stage2_multiscale_all", idx_all
            stage_desc = "ep1-12: S5N2+S6N3 (3000); ep13-30: S5N2+S6N3+S6N4+S8N6 (4500)"
        else:
            def schedule_fn(epoch: int) -> tuple[str, list[int]]:
                return "uniform_mix_multiscale", idx_all
            stage_desc = "ep1-30: uniform mix of S5N2(750)+S6N3(2250)+S6N4(750)+S8N6(750)"

        meta = {
            "block": "C_film_curriculum" if variant in FILM_VARIANTS else "B_cross_env",
            "curriculum_family": "multiscale_s5_s6_s8",
            "stage_desc": stage_desc,
        }

    else:
        raise AssertionError(f"未处理的 variant={variant}")

    # 硬性合规校验：4500 条唯一 seed + 零测试泄漏
    assert len(pool_eps) == 4500, f"{variant}: pool len={len(pool_eps)} != 4500"
    pool_seeds = [e["seed"] for e in pool_eps]
    assert len(set(pool_seeds)) == 4500, f"{variant}: 发现重复 seed，独立 seed 数={len(set(pool_seeds))}"
    assert_no_leak(pool_seeds + [e["seed"] for e in s6n3_val])

    env_counts: dict[str, int] = {}
    for ep in pool_eps:
        eid = ep.get("env_id", "BabyAI-GoToLocalS6N3-v0")
        env_counts[eid] = env_counts.get(eid, 0) + 1
    meta["pool_env_counts"] = env_counts
    meta["use_film"] = variant in FILM_VARIANTS

    return pool_eps, s6n3_val, schedule_fn, meta
