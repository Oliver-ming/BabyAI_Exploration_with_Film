"""支线 1 · 词序敏感度探针 + 打乱指令闭环鲁棒性评估。

审计修复：
  1. 强制单线程 OpenMP/PyTorch + ProcessPoolExecutor 并行
  2. 默认 --n-states 300（匹配 TEST_SEEDS 长度 300）
  3. 严格区分 bc30 / b2_film_nwd / mean 等组名
  4. 支持 FiLMLangBCNet 与 b2_film FiLMBCNet 的 encode_language 接口
  5. 对所有严格置换不变编码器（mean / bc30 / mean_wide / dan / b2_film*）及
     固定长度线性对消编码器（mean_pos）做数学构造性 0 敏感度硬断言
  6. 新增 --eval-scrambled 模式：在闭环 TEST_SEEDS 上直接评测
     normal vs swap（go to <art> <type> <color>）vs move_front（<color> go to <art> <type>），
     检验 Sinha et al. (ACL 2021) 的顺序依赖 vs 伪词序退化假设

用法:
    OMP_NUM_THREADS=1 python branches/b1_langenc/probe_word_order.py \
        --runs runs/b3_bc30_seed* runs/b2_film_nwd_seed* runs/b1_*_seed* \
        --workers 12 --out reports/b1_word_order_sensitivity.json \
        --eval-scrambled --scrambled-out reports/b1_scrambled_eval.json
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b1_langenc"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import make_env  # noqa: E402
from common import ENV_ID, TEST_SEEDS, mission_to_ids  # noqa: E402
from eval_closed_loop import evaluate, parse_seeds  # noqa: E402
from eval_langenc import group_name  # noqa: E402
from lang_encoders import bag_of_words_equivariant, fixed_length_bow_invariant  # noqa: E402
from lang_policy import load_any_run, make_policy_fn  # noqa: E402


def collect_observations(n_states: int = 300, scan: int = 300) -> list[dict]:
    """在测试 seed 上收集 (image, direction, mission) 状态。"""
    env = make_env()
    states = []
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in TEST_SEEDS[:scan]:
            if len(states) >= n_states:
                break
            obs, _ = env.reset(seed=seed)
            states.append(
                {
                    "seed": seed,
                    "image": obs["image"].copy(),
                    "direction": int(obs["direction"]),
                    "mission": obs["mission"],
                }
            )
    env.close()
    return states


def swap_counterfactual(mission: str) -> str:
    """go to [art] [color] [type] -> go to [art] [type] [color]（词袋相同）。"""
    p = mission.split()
    if len(p) != 5:
        return mission
    return " ".join([p[0], p[1], p[2], p[4], p[3]])


def move_color_front(mission: str) -> str:
    """go to [art] [color] [type] -> [color] go to [art] [type]（词袋相同）。"""
    p = mission.split()
    if len(p) != 5:
        return mission
    return " ".join([p[3], p[0], p[1], p[2], p[4]])


def act(policy, image, direction, mission) -> int:
    return policy({"image": image, "direction": direction, "mission": mission})


def lang_embedding(model, mission: str) -> np.ndarray | None:
    """统一抽取各架构语言分支的输出向量。"""
    toks = (mission_to_ids(mission) + [0] * 5)[:5]
    t = torch.tensor([toks], dtype=torch.long)
    with torch.no_grad():
        if hasattr(model, "encode_language"):
            return model.encode_language(t)[0].numpy()
        if hasattr(model, "lang"):
            return model.lang(t)[0].numpy()
        if hasattr(model, "lang_emb"):
            mask = (t != 0).unsqueeze(-1).float()
            e = model.lang_emb(t)
            return ((e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1))[0].numpy()
    return None


def probe_run_task(task: tuple[str, list[dict]]) -> dict:
    run, states = task
    torch.set_num_threads(1)
    model, cfg, _ = load_any_run(run)
    policy = make_policy_fn(model)
    grp = group_name(run, cfg)

    swap_changed = swap_total = 0
    front_changed = front_total = 0
    emb_swap_dist = []
    emb_front_dist = []
    for s in states:
        base = act(policy, s["image"], s["direction"], s["mission"])
        e_base = lang_embedding(model, s["mission"])

        alt = swap_counterfactual(s["mission"])
        if alt != s["mission"]:
            swap_total += 1
            if act(policy, s["image"], s["direction"], alt) != base:
                swap_changed += 1
            e_alt = lang_embedding(model, alt)
            if e_base is not None and e_alt is not None:
                emb_swap_dist.append(float(np.abs(e_base - e_alt).mean()))

        alt2 = move_color_front(s["mission"])
        if alt2 != s["mission"]:
            front_total += 1
            if act(policy, s["image"], s["direction"], alt2) != base:
                front_changed += 1
            e_alt2 = lang_embedding(model, alt2)
            if e_base is not None and e_alt2 is not None:
                emb_front_dist.append(float(np.abs(e_base - e_alt2).mean()))

    return {
        "run": run,
        "group": grp,
        "variant": grp,
        "branch": cfg.get("branch"),
        "train_seed": cfg.get("seed"),
        "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
        "n_lang_params": cfg.get("arch", {}).get("n_lang_params"),
        "n_params": cfg.get("arch", {}).get("n_params"),
        "swap_sensitivity": round(swap_changed / max(1, swap_total), 4),
        "move_front_sensitivity": round(front_changed / max(1, front_total), 4),
        "swap_embedding_l1": round(float(np.mean(emb_swap_dist)), 6) if emb_swap_dist else None,
        "move_front_embedding_l1": round(float(np.mean(emb_front_dist)), 6) if emb_front_dist else None,
        "n_pairs": swap_total,
    }


def eval_scrambled_task(task: tuple[str, list[int], str]) -> dict:
    run, seeds, env_id = task
    torch.set_num_threads(1)
    model, cfg, _ = load_any_run(run)
    grp = group_name(run, cfg)
    base_policy = make_policy_fn(model)

    def make_transformed_policy(fn):
        if fn is None:
            return base_policy

        def wrapped(obs):
            return base_policy({**obs, "mission": fn(obs["mission"])})

        return wrapped

    out = {
        "run": run,
        "group": grp,
        "train_seed": cfg.get("seed"),
        "n_params": cfg.get("arch", {}).get("n_params"),
        "n_lang_params": cfg.get("arch", {}).get("n_lang_params"),
    }
    for cond, fn in [("normal", None), ("swap", swap_counterfactual), ("move_front", move_color_front)]:
        res = evaluate(make_transformed_policy(fn), env_id, seeds)
        res.pop("_trajectories", None)
        out[cond] = {
            "closed_loop_success": res["closed_loop_success"],
            "mean_return": res["mean_return"],
            "n_timeout": res["n_timeout"],
        }
    out["drop_swap"] = round(out["normal"]["closed_loop_success"] - out["swap"]["closed_loop_success"], 6)
    out["drop_move_front"] = round(out["normal"]["closed_loop_success"] - out["move_front"]["closed_loop_success"], 6)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n-states", type=int, default=300)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="reports/b1_word_order_sensitivity.json")
    ap.add_argument("--eval-scrambled", action="store_true",
                    help="同时在 TEST_SEEDS 闭环评估 normal / swap / move_front")
    ap.add_argument("--scrambled-out", default="reports/b1_scrambled_eval.json")
    ap.add_argument("--seeds", default="test")
    args = ap.parse_args()

    # 去重
    runs = list(dict.fromkeys(args.runs))
    states = collect_observations(args.n_states, scan=len(TEST_SEEDS))
    print(f"采集 {len(states)} 个测试状态（mission 全部 5 词）")

    tasks = [(r, states) for r in runs]
    if args.workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            results = list(ex.map(probe_run_task, tasks))
    else:
        results = [probe_run_task(t) for t in tasks]

    # --- 构造性硬自检断言 ---
    checks = []
    for r in results:
        grp = r["group"]
        if fixed_length_bow_invariant(grp):
            emb_ok = r["swap_embedding_l1"] is None or r["swap_embedding_l1"] < 1e-5
            act_ok = r["swap_sensitivity"] == 0.0
            if bag_of_words_equivariant(grp):
                front_ok = r["move_front_sensitivity"] == 0.0 and (
                    r["move_front_embedding_l1"] is None or r["move_front_embedding_l1"] < 1e-5
                )
            else:
                # mean_pos 在 L=5 无 padding 时，所有置换下 5 个位置都恰好出现 1 次
                front_ok = r["move_front_sensitivity"] == 0.0 and (
                    r["move_front_embedding_l1"] is None or r["move_front_embedding_l1"] < 1e-5
                )
            ok = emb_ok and act_ok and front_ok
            checks.append(
                {
                    "run": r["run"],
                    "group": grp,
                    "strict_bow": bag_of_words_equivariant(grp),
                    "swap_sensitivity": r["swap_sensitivity"],
                    "move_front_sensitivity": r["move_front_sensitivity"],
                    "swap_embedding_l1": r["swap_embedding_l1"],
                    "passed": ok,
                }
            )
            if not ok:
                raise AssertionError(f"置换不变自检失败: {r}")

    print(f"[自检] 共 {len(checks)} 个置换不变/线性对消模型通过构造性 0 敏感度断言 ✅")

    agg: dict[str, list] = collections.defaultdict(list)
    for r in results:
        agg[r["group"]].append(r)

    summary = {
        "n_states": len(states),
        "probe_definition": {
            "swap": "go to <art> <color> <type> -> go to <art> <type> <color>（词袋完全相同，顺序不同）",
            "move_front": "<color> go to <art> <type>（词袋相同，常量前缀被打乱）",
        },
        "self_checks_zero_order_invariant": checks,
        "runs": results,
        "by_variant": {
            v: {
                "n": len(rs),
                "swap_sensitivity_mean": round(float(np.mean([x["swap_sensitivity"] for x in rs])), 4),
                "swap_sensitivity_sd": round(float(np.std([x["swap_sensitivity"] for x in rs])), 4),
                "move_front_sensitivity_mean": round(float(np.mean([x["move_front_sensitivity"] for x in rs])), 4),
                "move_front_sensitivity_sd": round(float(np.std([x["move_front_sensitivity"] for x in rs])), 4),
                "swap_embedding_l1_mean": round(
                    float(np.mean([x["swap_embedding_l1"] for x in rs if x["swap_embedding_l1"] is not None])), 6
                )
                if any(x["swap_embedding_l1"] is not None for x in rs)
                else None,
                "move_front_embedding_l1_mean": round(
                    float(np.mean([x["move_front_embedding_l1"] for x in rs if x["move_front_embedding_l1"] is not None])),
                    6,
                )
                if any(x["move_front_embedding_l1"] is not None for x in rs)
                else None,
                "n_lang_params": next((x["n_lang_params"] for x in rs if x["n_lang_params"] is not None), None),
            }
            for v, rs in agg.items()
        },
    }

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'group':15s} {'n':>2s} {'lang_p':>7s} {'swap_act':>9s} {'front_act':>10s} {'swap_emb_L1':>12s}")
    for v, a in sorted(summary["by_variant"].items()):
        lp = a["n_lang_params"] if a["n_lang_params"] is not None else -1
        el1 = f"{a['swap_embedding_l1_mean']:.5f}" if a["swap_embedding_l1_mean"] is not None else "—"
        print(
            f"{v:15s} {a['n']:2d} {lp:7d} {a['swap_sensitivity_mean']:9.4f} "
            f"{a['move_front_sensitivity_mean']:10.4f} {el1:>12s}"
        )
    print(f"\nwrote {p}")

    if args.eval_scrambled:
        seeds = parse_seeds(args.seeds)
        stasks = [(r, seeds, ENV_ID) for r in runs]
        if args.workers > 1 and len(stasks) > 1:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(stasks))) as ex:
                sresults = list(ex.map(eval_scrambled_task, stasks))
        else:
            sresults = [eval_scrambled_task(t) for t in stasks]

        sagg: dict[str, list] = collections.defaultdict(list)
        for r in sresults:
            sagg[r["group"]].append(r)

        sby = {}
        for v, rs in sagg.items():
            norm = [x["normal"]["closed_loop_success"] for x in rs]
            swp = [x["swap"]["closed_loop_success"] for x in rs]
            frt = [x["move_front"]["closed_loop_success"] for x in rs]
            dsw = [x["drop_swap"] for x in rs]
            dfr = [x["drop_move_front"] for x in rs]
            # 断言 BoW 在闭环下掉点也严格为 0
            if fixed_length_bow_invariant(v):
                if any(abs(d) > 1e-9 for d in dsw + dfr):
                    raise AssertionError(f"闭环打乱自检失败（{v}）: dsw={dsw}, dfr={dfr}")
            sby[v] = {
                "n": len(rs),
                "normal_mean": round(float(np.mean(norm)), 4),
                "normal_sd": round(float(np.std(norm)), 4),
                "swap_mean": round(float(np.mean(swp)), 4),
                "swap_sd": round(float(np.std(swp)), 4),
                "move_front_mean": round(float(np.mean(frt)), 4),
                "move_front_sd": round(float(np.std(frt)), 4),
                "drop_swap_mean": round(float(np.mean(dsw)), 4),
                "drop_swap_sd": round(float(np.std(dsw)), 4),
                "drop_move_front_mean": round(float(np.mean(dfr)), 4),
                "drop_move_front_sd": round(float(np.std(dfr)), 4),
            }

        scr_out = {
            "env_id": ENV_ID,
            "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
            "runs": sresults,
            "by_variant": sby,
        }
        sp = Path(args.scrambled_out)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(scr_out, indent=2, ensure_ascii=False))
        print(f"\n[闭环打乱测试] {'group':15s} {'normal':>8s} {'swap':>8s} {'Δswap':>8s} {'front':>8s} {'Δfront':>8s}")
        for v, a in sorted(sby.items()):
            print(
                f"               {v:15s} {a['normal_mean']:8.3f} {a['swap_mean']:8.3f} "
                f"{-a['drop_swap_mean']:+8.3f} {a['move_front_mean']:8.3f} {-a['drop_move_front_mean']:+8.3f}"
            )
        print(f"wrote {sp}")


if __name__ == "__main__":
    main()
