"""支线 2 扩展消融 v2 · 全量闭环评估、低数据评估、反事实语言探针、失败拆解、机制诊断与 300-DPI 作图脚本。

运行:
    OMP_NUM_THREADS=1 env_embodied/bin/python branches/b2_attn/eval_v2_and_analyze.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import collections
import concurrent.futures
import contextlib
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b2_attn"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from attn_net import ATTN_VARIANTS, build as build_attn_v1  # noqa: E402
from attn_v2_net import ATTN_V2_VARIANTS, build as build_attn_v2  # noqa: E402
from branch_utils import make_env  # noqa: E402
from common import (  # noqa: E402
    ENV_ID,
    ENV_ID_S7,
    PAD_IDX,
    TEST_SEEDS,
    TEST_SEEDS_S7,
    encode_batch,
)
from compare_failures import MECHANISM, run_episode_with_policy  # noqa: E402
from eval_closed_loop import evaluate  # noqa: E402
from film_policy import load_any_run, make_policy_fn  # noqa: E402
from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT  # noqa: E402
from probe_language import collect_states  # noqa: E402
from render_episodes import classify_failure  # noqa: E402

ENV_ID_S8 = "BabyAI-GoToLocalS8N6-v0"
SEEDS = [0, 1, 2]
CASE_SEEDS = [4501, 4516, 4560, 4568, 4502]

F020_V2_VARIANTS = (
    "res_spatial_softmax",
    "mh_res_spatial_softmax",
    "attn_map_inject",
    "token_coattn_spatial",
    "mh_spatial_relational",
)

SHORT_LABELS = {
    "bc30": "BC30\n(Concat)",
    "film_nwd": "FiLM\n(Primary)",
    "spatial_attn_gate": "v1:Spatial\nGate",
    "spatial_softmax_attn": "v1:Spatial\nSoftmax",
    "cross_attn_mha": "v1:Cross\nMHA",
    "film_spatial_attn": "v1:FiLM+\nSpatial",
    "film_cross_mha": "v1:FiLM+\nMHA",
    "gated_attn_chaplot": "v2:Chaplot\nGA",
    "cross_mha_nopos": "v2:MHA\nNoPE",
    "res_spatial_softmax": "v2:Res\nSoftmax",
    "mh_res_spatial_softmax": "v2:MH-Res\nSoftmax",
    "attn_map_inject": "v2:Map\nInject",
    "token_coattn_spatial": "v2:Token\nCoAttn",
    "mh_spatial_relational": "v2:MH-Res\n+Rel",
}


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def load_run_universal(run_dir: str | Path):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    branch = cfg.get("branch")
    if branch in ("b2_attn", "b2_attn_v2"):
        vocab = json.loads((run_dir / "vocab.json").read_text())
        a = cfg["arch"]
        builder = build_attn_v2 if branch == "b2_attn_v2" else build_attn_v1
        model = builder(
            a["variant"],
            vocab_size=len(vocab["vocab"]),
            obj_emb_dim=a["obj_emb_dim"],
            lang_dim=a["lang_dim"],
            hidden=a.get("hidden", 128),
            dropout=a.get("dropout", 0.0),
        )
        state = torch.load(run_dir / "best_model.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model, cfg, vocab
    return load_any_run(run_dir)


def mean_sd(vals: list[float]) -> dict:
    vals = [float(v) for v in vals]
    return {
        "n": len(vals),
        "mean": round(st.mean(vals), 6),
        "sd": round(st.pstdev(vals), 6),
        "values": [round(v, 6) for v in vals],
    }


def _eval_worker(task: tuple[str, str, list[int], str]) -> dict:
    run_dir, env_id, seeds, key = task
    model, cfg, _ = load_run_universal(run_dir)
    t0 = time.time()
    res = evaluate(make_policy_fn(model), env_id, seeds)
    res.pop("_trajectories", None)
    return {
        "_key": key,
        "run": run_dir,
        "variant": cfg.get("arch", {}).get("variant", "concat"),
        "branch": cfg.get("branch", "baseline"),
        "train_seed": cfg.get("seed"),
        "frac": cfg.get("frac", 1.0),
        "val_action_acc": cfg.get("result", {}).get("best_val_acc"),
        "best_epoch": cfg.get("result", {}).get("best_epoch"),
        "n_params": cfg.get("arch", {}).get("n_params"),
        "train_seconds": cfg.get("result", {}).get("train_seconds"),
        "env_id": env_id,
        "eval_seconds": round(time.time() - t0, 2),
        **res,
    }


def summarize_group(runs: list[dict]) -> dict:
    runs = sorted(runs, key=lambda r: r["train_seed"])
    return {
        "n_params": runs[0]["n_params"],
        "closed_loop_success": mean_sd([r["closed_loop_success"] for r in runs]),
        "val_action_acc": mean_sd([r["val_action_acc"] for r in runs]),
        "mean_return": mean_sd([r["mean_return"] for r in runs]),
        "median_len_success": mean_sd([r["median_len_success"] for r in runs]),
        "n_timeout": mean_sd([r["n_timeout"] for r in runs]),
    }


def paired_delta(runs: list[dict], ref_runs: list[dict]) -> dict:
    rmap = {r["train_seed"]: r for r in runs}
    refmap = {r["train_seed"]: r for r in ref_runs}
    diffs, rows = [], []
    for s in SEEDS:
        d = round(rmap[s]["closed_loop_success"] - refmap[s]["closed_loop_success"], 6)
        diffs.append(d)
        rows.append(
            {
                "seed": s,
                "variant": rmap[s]["closed_loop_success"],
                "ref": refmap[s]["closed_loop_success"],
                "delta": d,
            }
        )
    return {
        "n_pairs": len(diffs),
        "delta_mean": round(st.mean(diffs), 6),
        "delta_sd": round(st.pstdev(diffs), 6),
        "n_positive": sum(1 for d in diffs if d > 0),
        "rows": rows,
    }


def run_main_eval() -> dict:
    print("== 1. Running v2 S6N3 / S7N4 / S8N6 closed-loop evaluations ==")
    b2_s6 = json.loads((ROOT / "reports/b2_film_eval.json").read_text())
    b2_s7 = json.loads((ROOT / "reports/b2_film_s7n4.json").read_text())
    ev1 = json.loads((ROOT / "reports/b2_attn_eval.json").read_text())

    bc30_s6 = [r for r in b2_s6["baseline_runs"] if "b3_bc30" in r["run"]]
    film_s6 = [r for r in b2_s6["runs"] if "b2_film_nwd" in r["run"]]
    bc30_s7 = [r for r in b2_s7["baseline_runs"] if "b3_bc30" in r["run"]]
    film_s7 = [r for r in b2_s7["runs"] if "b2_film_nwd" in r["run"]]
    bc30_s8 = [r for r in ev1["runs"] if "b3_bc30" in r["run"] and r["env_id"] == ENV_ID_S8]
    film_s8 = [r for r in ev1["runs"] if "b2_film_nwd" in r["run"] and r["env_id"] == ENV_ID_S8]

    ss_s6 = [r for r in ev1["runs"] if r.get("variant") == "spatial_softmax_attn" and r["env_id"] == ENV_ID]
    ss_s7 = [r for r in ev1["runs"] if r.get("variant") == "spatial_softmax_attn" and r["env_id"] == ENV_ID_S7]
    ss_s8 = [r for r in ev1["runs"] if r.get("variant") == "spatial_softmax_attn" and r["env_id"] == ENV_ID_S8]

    tasks = []
    for v in ATTN_V2_VARIANTS:
        for s in SEEDS:
            rdir = f"runs/b2_attn_v2_{v}_seed{s}"
            tasks.append((rdir, ENV_ID, TEST_SEEDS, f"{v}::S6N3"))
            tasks.append((rdir, ENV_ID_S7, TEST_SEEDS_S7, f"{v}::S7N4"))
            tasks.append((rdir, ENV_ID_S8, TEST_SEEDS_S7, f"{v}::S8N6"))

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    v2_runs: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, initializer=_worker_init) as ex:
        for res in ex.map(_eval_worker, tasks):
            k = res.pop("_key")
            grouped[k].append(res)
            v2_runs.append(res)

    results = {
        "branch": "b2_attn_v2",
        "sd_convention": "ddof=0 (population SD via statistics.pstdev)",
        "seeds_s6n3": "4500:4800",
        "seeds_s7n4": "10000:10300",
        "seeds_s8n6": "10000:10300",
        "references": {
            **ev1["references"],
            **{f"v1_{k}": v for k, v in ev1["variants"].items()},
        },
        "variants": {},
        "runs": v2_runs,
    }

    for v in ATTN_V2_VARIANTS:
        r_s6 = grouped[f"{v}::S6N3"]
        r_s7 = grouped[f"{v}::S7N4"]
        r_s8 = grouped[f"{v}::S8N6"]
        results["variants"][v] = {
            "S6N3": summarize_group(r_s6),
            "S7N4": summarize_group(r_s7),
            "S8N6": summarize_group(r_s8),
            "paired_vs_bc30": {
                "S6N3": paired_delta(r_s6, bc30_s6),
                "S7N4": paired_delta(r_s7, bc30_s7),
                "S8N6": paired_delta(r_s8, bc30_s8),
            },
            "paired_vs_film_nwd": {
                "S6N3": paired_delta(r_s6, film_s6),
                "S7N4": paired_delta(r_s7, film_s7),
                "S8N6": paired_delta(r_s8, film_s8),
            },
            "paired_vs_v1_spatial_softmax": {
                "S6N3": paired_delta(r_s6, ss_s6),
                "S7N4": paired_delta(r_s7, ss_s7),
                "S8N6": paired_delta(r_s8, ss_s8),
            },
        }
        s6m = results["variants"][v]["S6N3"]["closed_loop_success"]
        s7m = results["variants"][v]["S7N4"]["closed_loop_success"]
        s8m = results["variants"][v]["S8N6"]["closed_loop_success"]
        print(f"  [{v:24s}] S6N3={s6m['mean']:.3f}±{s6m['sd']:.3f} | S7N4={s7m['mean']:.3f}±{s7m['sd']:.3f} | S8N6={s8m['mean']:.3f}±{s8m['sd']:.3f}")

    (ROOT / "reports/b2_attn_v2_eval.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def run_f020_eval() -> dict:
    print("== 2. Running 20% step-matched low-data evaluations (v1 spatial_softmax + 5 v2 variants) ==")
    f20_v1 = json.loads((ROOT / "reports/b2_attn_f020_eval.json").read_text())

    tasks = []
    # 1. Missing v1 spatial_softmax_attn f020
    for s in SEEDS:
        rdir = f"runs/b2_attn_spatial_softmax_attn_f020_seed{s}"
        tasks.append((rdir, ENV_ID, TEST_SEEDS, "v1_spatial_softmax_attn::S6N3"))
        tasks.append((rdir, ENV_ID_S7, TEST_SEEDS_S7, "v1_spatial_softmax_attn::S7N4"))
    # 2. Five v2 variants f020
    for v in F020_V2_VARIANTS:
        for s in SEEDS:
            rdir = f"runs/b2_attn_v2_{v}_f020_seed{s}"
            tasks.append((rdir, ENV_ID, TEST_SEEDS, f"{v}::S6N3"))
            tasks.append((rdir, ENV_ID_S7, TEST_SEEDS_S7, f"{v}::S7N4"))

    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    runs_all: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, initializer=_worker_init) as ex:
        for res in ex.map(_eval_worker, tasks):
            k = res.pop("_key")
            grouped[k].append(res)
            runs_all.append(res)

    out = {
        "frac": 0.2,
        "n_episodes": 900,
        "step_matched_to": 16534,
        "references": {
            **f20_v1["references"],
            **{f"v1_{k}": v for k, v in f20_v1["variants"].items()},
        },
        "variants": {},
        "runs": runs_all,
    }

    keys = ["v1_spatial_softmax_attn", *F020_V2_VARIANTS]
    for v in keys:
        r_s6 = grouped[f"{v}::S6N3"]
        r_s7 = grouped[f"{v}::S7N4"]
        out["variants"][v] = {
            "S6N3": summarize_group(r_s6),
            "S7N4": summarize_group(r_s7),
        }
        s6m = out["variants"][v]["S6N3"]["closed_loop_success"]
        s7m = out["variants"][v]["S7N4"]["closed_loop_success"]
        print(f"  [{v:24s} f020] S6N3={s6m['mean']:.3f}±{s6m['sd']:.3f} | S7N4={s7m['mean']:.3f}±{s7m['sd']:.3f}")

    (ROOT / "reports/b2_attn_v2_f020_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def _probe_worker(task: tuple[str, str, list[dict]]) -> dict:
    name, run_dir, states = task
    model, cfg, _ = load_run_universal(run_dir)
    policy = make_policy_fn(model)
    c_chg = c_tot = t_chg = t_tot = 0
    for s in states:
        with torch.no_grad():
            base = policy({"image": s["image"], "direction": s["direction"], "mission": s["mission"]})
            if s["alt_color_mission"]:
                c_tot += 1
                if policy({"image": s["image"], "direction": s["direction"], "mission": s["alt_color_mission"]}) != base:
                    c_chg += 1
            if s["alt_type_mission"]:
                t_tot += 1
                if policy({"image": s["image"], "direction": s["direction"], "mission": s["alt_type_mission"]}) != base:
                    t_chg += 1
    return {
        "_group": name,
        "run": run_dir,
        "variant": cfg.get("arch", {}).get("variant", "concat"),
        "train_seed": cfg.get("seed"),
        "color_total": c_tot,
        "color_changed": c_chg,
        "color_swap_sensitivity": round(c_chg / max(1, c_tot), 6),
        "type_total": t_tot,
        "type_changed": t_chg,
        "type_swap_sensitivity": round(t_chg / max(1, t_tot), 6),
        "combined_sensitivity": round((c_chg + t_chg) / max(1, c_tot + t_tot), 6),
    }


def run_probe() -> dict:
    print("== 3. Running 183-state counterfactual language probe ==")
    states = collect_states(600, scan=300)
    pr1 = json.loads((ROOT / "reports/b2_attn_probe.json").read_text())

    tasks = [
        (v, f"runs/b2_attn_v2_{v}_seed{s}", states)
        for v in ATTN_V2_VARIANTS
        for s in SEEDS
    ]
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    runs_v2: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, initializer=_worker_init) as ex:
        for res in ex.map(_probe_worker, tasks):
            g = res.pop("_group")
            grouped[g].append(res)
            runs_v2.append(res)

    out = {
        "n_states": len(states),
        "references": pr1["by_variant"],
        "by_variant": {},
        "runs": runs_v2,
    }
    for v in ATTN_V2_VARIANTS:
        rs = sorted(grouped[v], key=lambda r: r["train_seed"])
        out["by_variant"][v] = {
            "color_swap_sensitivity": mean_sd([r["color_swap_sensitivity"] for r in rs]),
            "type_swap_sensitivity": mean_sd([r["type_swap_sensitivity"] for r in rs]),
            "combined_sensitivity": mean_sd([r["combined_sensitivity"] for r in rs]),
        }
        cs = out["by_variant"][v]["color_swap_sensitivity"]
        ts = out["by_variant"][v]["type_swap_sensitivity"]
        print(f"  [{v:24s}] color={cs['mean']:.3f}±{cs['sd']:.3f} | type={ts['mean']:.3f}±{ts['sd']:.3f}")

    (ROOT / "reports/b2_attn_v2_probe.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def _failure_worker(task: tuple[str, str]) -> tuple[str, dict, dict]:
    name, run_dir = task
    model, cfg, _ = load_run_universal(run_dir)
    policy = make_policy_fn(model)
    env = make_env(ENV_ID)
    trajs = []
    with contextlib.redirect_stdout(io.StringIO()):
        for seed in TEST_SEEDS:
            trajs.append(run_episode_with_policy(env, policy, seed))
    env.close()

    types = collections.Counter()
    groups = collections.Counter()
    cases = {}
    tmap = {t["seed"]: t for t in trajs}
    for t in trajs:
        if not t["success"]:
            ft, _ = classify_failure(t)
            types[ft] += 1
            groups[MECHANISM.get(ft, "other")] += 1
    for cs in CASE_SEEDS:
        t = tmap[cs]
        ft = None if t["success"] else classify_failure(t)[0]
        cases[str(cs)] = {
            "success": t["success"],
            "episode_len": t["episode_len"],
            "failure_type": ft,
            "mission": t["mission"],
            "final_pos": t["final_agent_pos"],
            "final_front_cell": t["final_front_cell"],
        }
    summary = {
        "run": run_dir,
        "variant": cfg.get("arch", {}).get("variant", "concat"),
        "train_seed": cfg.get("seed"),
        "n_success": sum(1 for t in trajs if t["success"]),
        "n_failure": sum(1 for t in trajs if not t["success"]),
        "failure_types": dict(types),
        "failure_groups": dict(groups),
    }
    return name, summary, cases


def run_failures() -> dict:
    print("== 4. Running 300-episode failure breakdown & 5-case audit ==")
    fl1 = json.loads((ROOT / "reports/b2_attn_failures.json").read_text())

    tasks = [(v, f"runs/b2_attn_v2_{v}_seed{s}") for v in ATTN_V2_VARIANTS for s in SEEDS]
    grouped: dict[str, list[tuple[dict, dict]]] = collections.defaultdict(list)
    with concurrent.futures.ProcessPoolExecutor(max_workers=8, initializer=_worker_init) as ex:
        for name, s_res, c_res in ex.map(_failure_worker, tasks):
            grouped[name].append((s_res, c_res))

    out = {
        "env_id": ENV_ID,
        "seeds": "4500:4800",
        "case_seeds": CASE_SEEDS,
        "references": {"aggregate": fl1["aggregate"], "cases": fl1["cases"]},
        "aggregate": {},
        "cases": {},
        "runs": [],
    }
    for v in ATTN_V2_VARIANTS:
        pairs = sorted(grouped[v], key=lambda x: x[0]["train_seed"])
        rs = [x[0] for x in pairs]
        cs_list = [x[1] for x in pairs]
        out["runs"].extend(rs)
        all_types = sorted({k for r in rs for k in r["failure_types"]})
        out["aggregate"][v] = {
            "n_failure": mean_sd([r["n_failure"] for r in rs]),
            "representation": mean_sd([r["failure_groups"].get("representation", 0) for r in rs]),
            "state_coverage": mean_sd([r["failure_groups"].get("state_coverage", 0) for r in rs]),
            "failure_types_mean": {
                k: round(sum(r["failure_types"].get(k, 0) for r in rs) / len(rs), 2) for k in all_types
            },
        }
        out["cases"][v] = {}
        for cs in CASE_SEEDS:
            k = str(cs)
            per_s = [c[k] for c in cs_list]
            out["cases"][v][k] = {
                "n_solved": sum(1 for x in per_s if x["success"]),
                "per_seed": per_s,
            }
        ag = out["aggregate"][v]
        print(
            f"  [{v:24s}] fail={ag['n_failure']['mean']:4.1f} "
            f"(repr={ag['representation']['mean']:4.1f}, cov={ag['state_coverage']['mean']:4.1f}) "
            + " ".join(f"s{cs}:{out['cases'][v][str(cs)]['n_solved']}/3" for cs in CASE_SEEDS)
        )

    (ROOT / "reports/b2_attn_v2_failures.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def _cell_multiplier_stats(obs: dict, smap: np.ndarray) -> dict:
    grid = obs["image"]
    mission = obs["mission"]
    parts = mission.split()
    tcol, ttype = parts[-2], parts[-1]
    res: dict[str, list[float]] = {"target": [], "blocking_yellow_box": [], "other_objects": [], "walls": [], "empty": []}
    for r in range(7):
        for c in range(7):
            oid, cid = int(grid[r, c, 0]), int(grid[r, c, 1])
            oname = IDX_TO_OBJECT.get(oid, "")
            cname = IDX_TO_COLOR.get(cid, "")
            val = float(smap[r, c])
            if oname == ttype and cname == tcol:
                res["target"].append(val)
            elif oname == "box" and cname == "yellow":
                res["blocking_yellow_box"].append(val)
            elif oname in ("key", "ball", "box"):
                res["other_objects"].append(val)
            elif oname == "wall":
                res["walls"].append(val)
            elif oname == "empty":
                res["empty"].append(val)
    return {k: (round(float(np.mean(vs)), 4) if vs else None) for k, vs in res.items()}


def run_diagnostics() -> dict:
    print("== 5. Running mechanistic diagnostics (4501 t=2 modulation, token weights, learned scalars) ==")
    env = make_env(ENV_ID)
    with contextlib.redirect_stdout(io.StringIO()):
        obs_4501_t2, _ = env.reset(seed=4501)
        for a in (0, 2):
            obs_4501_t2, _, _, _, _ = env.step(a)

    img_4501, tok_4501, dir_4501 = encode_batch([obs_4501_t2])
    seed4501_mod = {}
    compare_runs = [
        ("v1_spatial_softmax", "runs/b2_attn_spatial_softmax_attn"),
        ("v2_res_spatial_softmax", "runs/b2_attn_v2_res_spatial_softmax"),
        ("v2_mh_res_spatial_softmax", "runs/b2_attn_v2_mh_res_spatial_softmax"),
        ("v2_token_coattn_spatial", "runs/b2_attn_v2_token_coattn_spatial"),
        ("v2_mh_spatial_relational", "runs/b2_attn_v2_mh_spatial_relational"),
    ]
    for label, prefix in compare_runs:
        per_seed = []
        for s in SEEDS:
            m, _, _ = load_run_universal(f"{prefix}_seed{s}")
            sp = m.extract_spatial_map(img_4501, tok_4501, dir_4501)
            per_seed.append(_cell_multiplier_stats(obs_4501_t2, sp["spatial_map"]))
        seed4501_mod[label] = {
            k: mean_sd([x[k] for x in per_seed if x[k] is not None])
            for k in ("target", "blocking_yellow_box", "other_objects", "walls", "empty")
        }

    # Learned scalars across all 7 v2 variants × 3 seeds
    learned_scalars = {}
    for v in ATTN_V2_VARIANTS:
        seed_rows = []
        for s in SEEDS:
            cfg = json.loads((ROOT / f"runs/b2_attn_v2_{v}_seed{s}/config.json").read_text())
            seed_rows.append(cfg["result"].get("mod_stats_at_best", {}))
        learned_scalars[v] = seed_rows

    # Token-level attention weights for token_coattn_spatial averaged over 300 S6N3 test episodes (initial obs)
    sample_obs = []
    with contextlib.redirect_stdout(io.StringIO()):
        for ts in TEST_SEEDS:
            o, _ = env.reset(seed=ts)
            sample_obs.append(o)
    env.close()

    imgs, toks, dirs = encode_batch(sample_obs)
    n_len5 = 0
    head_tok_per_seed = []
    for s in SEEDS:
        m, _, _ = load_run_universal(f"runs/b2_attn_v2_token_coattn_spatial_seed{s}")
        w_accum = np.zeros((4, 5), dtype=np.float64)
        cnt = 0
        for i in range(len(sample_obs)):
            valid_len = int((toks[i] != PAD_IDX).sum().item())
            if valid_len != 5:
                continue
            sp = m.extract_diagnostics(imgs[i : i + 1], toks[i : i + 1], dirs[i : i + 1])
            w_accum += sp["token_weights"][:, :5]
            cnt += 1
        n_len5 = cnt
        head_tok_per_seed.append(w_accum / max(1, cnt))

    mean_tok = np.mean(head_tok_per_seed, axis=0)
    sd_tok = np.std(head_tok_per_seed, axis=0)

    out = {
        "seed4501_t2_mission": obs_4501_t2["mission"],
        "seed4501_t2_effective_modulation": seed4501_mod,
        "learned_scalars": learned_scalars,
        "token_coattn_word_weights": {
            "n_episodes_len5": n_len5,
            "slots": ["go", "to", "the", "<color>", "<type>"],
            "heads_mean_4x5": np.round(mean_tok, 4).tolist(),
            "heads_sd_4x5": np.round(sd_tok, 4).tolist(),
            "seed0_4x5": np.round(head_tok_per_seed[0], 4).tolist(),
        },
    }
    (ROOT / "reports/b2_attn_v2_diagnostics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def make_benchmark_figure(ev: dict, f20: dict, pr: dict, fl: dict) -> None:
    print("== 6a. Generating 300-DPI b2_attn_v2_benchmark.png ==")
    plt.rcParams.update({"font.size": 9.2, "axes.spines.top": False, "axes.spines.right": False})
    (ROOT / "reports/figures/b2_attn").mkdir(parents=True, exist_ok=True)

    order = [
        "bc30",
        "film_nwd",
        "spatial_attn_gate",
        "spatial_softmax_attn",
        "cross_attn_mha",
        "gated_attn_chaplot",
        "cross_mha_nopos",
        "res_spatial_softmax",
        "mh_res_spatial_softmax",
        "attn_map_inject",
        "token_coattn_spatial",
        "mh_spatial_relational",
    ]
    colors = [
        "#D97706",  # bc30
        "#059669",  # film_nwd
        "#60A5FA",  # v1 gate
        "#2563EB",  # v1 softmax
        "#818CF8",  # v1 mha
        "#94A3B8",  # v2 chaplot
        "#A78BFA",  # v2 nopos
        "#DC2626",  # v2 res_softmax
        "#B91C1C",  # v2 mh_res_softmax
        "#EA580C",  # v2 map_inject
        "#7C3AED",  # v2 token_coattn
        "#0F766E",  # v2 relational
    ]

    def get_full_ms(key: str, env_tag: str) -> tuple[float, float]:
        if key in ("bc30", "film_nwd"):
            d = ev["references"][key][env_tag]["closed_loop_success"]
        elif f"v1_{key}" in ev["references"]:
            d = ev["references"][f"v1_{key}"][env_tag]["closed_loop_success"]
        else:
            d = ev["variants"][key][env_tag]["closed_loop_success"]
        return d["mean"], d["sd"]

    fig = plt.figure(figsize=(15.2, 9.0), dpi=300)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.22)

    # Panel (a): Top full-width S6N3, S7N4, S8N6
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(order))
    w = 0.25
    for idx, (env_tag, hatch, alpha, lbl) in enumerate(
        [("S6N3", "", 0.94, "S6N3 (in-dist)"), ("S7N4", "///", 0.58, "S7N4 (zero-shot 7×7)"), ("S8N6", "...", 0.36, "S8N6 (zero-shot 8×8 N=6)")]
    ):
        means = [get_full_ms(k, env_tag)[0] for k in order]
        sds = [get_full_ms(k, env_tag)[1] for k in order]
        bars = ax.bar(
            x + (idx - 1) * w,
            means,
            w,
            yerr=sds,
            capsize=2.0,
            color=colors,
            alpha=alpha,
            hatch=hatch,
            edgecolor="#1F2937",
            linewidth=0.65,
            label=lbl,
        )
        if env_tag in ("S6N3", "S8N6"):
            for b, m in zip(bars, means):
                dy = 0.012 if env_tag == "S6N3" else 0.010
                ax.text(b.get_x() + b.get_width() / 2, m + dy, f"{m:.3f}", ha="center", va="bottom", fontsize=6.2, fontweight="bold" if env_tag == "S6N3" else "normal")
    ax.axvline(4.5, color="#6B7280", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[k] for k in order], fontsize=8.0)
    ax.set_ylim(0.40, 1.13)
    ax.set_ylabel("Closed-loop Success (mean ± SD, n=3)")
    ax.set_title("(a) Full-Data Performance & Cross-Scale Generalization: Frozen Baselines & v1 vs v2 Attention Designs", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.2, frameon=True, framealpha=0.92, facecolor="white", edgecolor="#E5E7EB", ncol=3, loc="upper center")

    # Panel (b): Step-matched 20% Low-Data (S6N3 & S7N4)
    ax_b = fig.add_subplot(gs[1, 0])
    f20_order = [
        ("bc30", "BC30"),
        ("film_nwd", "FiLM"),
        ("v1_spatial_attn_gate", "v1:Gate"),
        ("v1_spatial_softmax_attn", "v1:Softmax"),
        ("v1_cross_attn_mha", "v1:MHA"),
        ("res_spatial_softmax", "v2:ResSS"),
        ("mh_res_spatial_softmax", "v2:MH-Res"),
        ("attn_map_inject", "v2:MapInj"),
        ("token_coattn_spatial", "v2:CoAttn"),
        ("mh_spatial_relational", "v2:Rel"),
    ]
    f20_cols = ["#D97706", "#059669", "#60A5FA", "#2563EB", "#818CF8", "#DC2626", "#B91C1C", "#EA580C", "#7C3AED", "#0F766E"]

    def get_f20_ms(k: str, env_tag: str) -> tuple[float, float]:
        if k == "bc30":
            d = f20["references"]["bc_step_f020"][env_tag]
            return d["success_mean"], d["success_sd"]
        if k == "film_nwd":
            d = f20["references"]["film_step_f020"][env_tag]
            return d["success_mean"], d["success_sd"]
        if k in f20["references"]:
            d = f20["references"][k][env_tag]["closed_loop_success"]
            return d["mean"], d["sd"]
        d = f20["variants"][k][env_tag]["closed_loop_success"]
        return d["mean"], d["sd"]

    xb = np.arange(len(f20_order))
    wb = 0.36
    m6 = [get_f20_ms(k, "S6N3")[0] for k, _ in f20_order]
    sd6 = [get_f20_ms(k, "S6N3")[1] for k, _ in f20_order]
    m7 = [get_f20_ms(k, "S7N4")[0] for k, _ in f20_order]
    sd7 = [get_f20_ms(k, "S7N4")[1] for k, _ in f20_order]
    b6 = ax_b.bar(xb - wb / 2, m6, wb, yerr=sd6, capsize=2.0, color=f20_cols, alpha=0.92, edgecolor="#1F2937", linewidth=0.65, label="S6N3 (20% data)")
    ax_b.bar(xb + wb / 2, m7, wb, yerr=sd7, capsize=2.0, color=f20_cols, alpha=0.52, hatch="///", edgecolor="#1F2937", linewidth=0.65, label="S7N4 (20% data)")
    for b, m in zip(b6, m6):
        ax_b.text(b.get_x() + b.get_width() / 2, m + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold")
    ax_b.axhline(0.8144, color="#D97706", linestyle="--", linewidth=1.0, label="100% BC30 S6N3 (0.814)")
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([lbl for _, lbl in f20_order], fontsize=7.6, rotation=18)
    ax_b.set_ylim(0.50, 1.03)
    ax_b.set_ylabel("Closed-loop Success (20% Data)")
    ax_b.set_title("(b) Step-Matched 20% Low-Data Regime (900 Episodes)", fontweight="bold")
    ax_b.grid(axis="y", linestyle="--", alpha=0.35)
    ax_b.legend(fontsize=7.6, frameon=False, loc="lower left")

    # Panel (c): Representation Failures vs Color-Swap Sensitivity
    ax_c = fig.add_subplot(gs[1, 1])

    def get_probe(k: str) -> tuple[float, float]:
        src = pr["references"][k] if k in pr["references"] else pr["by_variant"][k]
        return src["color_swap_sensitivity"]["mean"], src["color_swap_sensitivity"]["sd"]

    def get_fail(k: str) -> tuple[float, float, float]:
        src = fl["references"]["aggregate"][k] if k in fl["references"]["aggregate"] else fl["aggregate"][k]
        return src["n_failure"]["mean"], src["representation"]["mean"], src["state_coverage"]["mean"]

    xc = np.arange(len(order))
    repr_f = [get_fail(k)[1] for k in order]
    cov_f = [get_fail(k)[2] for k in order]
    ax_c.bar(xc, repr_f, 0.52, color="#EF4444", alpha=0.85, edgecolor="#1F2937", linewidth=0.6, label="Representation failures (/300)")
    ax_c.bar(xc, cov_f, 0.52, bottom=repr_f, color="#F59E0B", alpha=0.75, edgecolor="#1F2937", linewidth=0.6, label="State-coverage / detour failures (/300)")
    for xi, (rf, cf) in enumerate(zip(repr_f, cov_f)):
        ax_c.text(xi, rf + cf + 1.0, f"{rf + cf:.1f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax_c_r = ax_c.twinx()
    ax_c_r.spines["top"].set_visible(False)
    cs_m = [get_probe(k)[0] for k in order]
    ax_c_r.plot(xc, cs_m, "o-", color="#1D4ED8", linewidth=1.6, markersize=5, label="Color-swap sensitivity")
    compact_labels = ["BC30", "FiLM", "v1:Gate", "v1:Softmax", "v1:MHA", "v2:Chaplot", "v2:NoPE", "v2:ResSS", "v2:MH-Res", "v2:MapInj", "v2:CoAttn", "v2:Rel"]
    ax_c.set_xticks(xc)
    ax_c.set_xticklabels(compact_labels, fontsize=7.6, rotation=32, ha="right")
    ax_c.set_ylim(0, 68)
    ax_c_r.set_ylim(0, 0.36)
    ax_c.set_ylabel("Mean Failures / 300 S6N3 Episodes")
    ax_c_r.set_ylabel("Color-swap Action Flip Rate", color="#1D4ED8")
    ax_c.set_title("(c) Failure Taxonomy Reduction & Counterfactual Sensitivity", fontweight="bold")
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)
    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_c_r.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, fontsize=7.2, frameon=False, loc="upper right")

    fig.savefig(ROOT / "reports/figures/b2_attn/b2_attn_v2_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_mechanisms_figure(ev: dict, diag: dict) -> None:
    print("== 6b. Generating 300-DPI b2_attn_v2_mechanisms.png ==")
    plt.rcParams.update({"font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    from eval_and_analyze import draw_egocentric_grid_with_map

    fig = plt.figure(figsize=(15.5, 8.6), dpi=300)
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.55)

    # Panel (a): Seed 4501 t=2 effective modulation multiplier comparison
    ax_a = fig.add_subplot(gs[0, 0:4])
    models_a = [
        ("v1_spatial_softmax", "v1:Spatial\nSoftmax"),
        ("v2_res_spatial_softmax", "v2:Res\nSoftmax"),
        ("v2_mh_res_spatial_softmax", "v2:MH-Res\nSoftmax"),
        ("v2_token_coattn_spatial", "v2:Token\nCoAttn"),
    ]
    cats = [
        ("target", "Target (blue key)", "#2563EB"),
        ("blocking_yellow_box", "Obstacle (yellow box)", "#D97706"),
        ("walls", "Walls", "#4B5563"),
    ]
    xa = np.arange(len(models_a))
    wa = 0.25
    mod_data = diag["seed4501_t2_effective_modulation"]
    for idx, (ck, clbl, col) in enumerate(cats):
        vals = [mod_data[mk][ck]["mean"] for mk, _ in models_a]
        sds = [mod_data[mk][ck]["sd"] for mk, _ in models_a]
        bars = ax_a.bar(xa + (idx - 1) * wa, vals, wa, yerr=sds, capsize=2.0, color=col, edgecolor="#1F2937", linewidth=0.65, label=clbl)
        for b, v, s in zip(bars, vals, sds):
            ax_a.text(b.get_x() + b.get_width() / 2, v + s + 0.28, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax_a.axhline(1.0, color="#DC2626", linestyle="--", linewidth=1.1, label="Identity floor (1.0×)")
    ax_a.set_xticks(xa)
    ax_a.set_xticklabels([lbl for _, lbl in models_a], fontsize=8.0)
    ax_a.set_ylabel("Effective Multiplier on h1 (Seed 4501, t=2)")
    ax_a.set_ylim(0, 18.2)
    ax_a.set_title("(a) Seed 4501 Obstacle Preservation Check", fontweight="bold")
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)
    ax_a.legend(fontsize=7.0, frameon=False, loc="upper right")

    # Panel (b): OOD scale curve (S6N3 -> S7N4 -> S8N6)
    ax_b = fig.add_subplot(gs[0, 4:8])
    curve_specs = [
        ("cross_attn_mha", True, "v1:Cross-MHA (abs pos_emb)", "#818CF8", "--", "s"),
        ("cross_mha_nopos", False, "v2:Cross-MHA (NoPE ablation)", "#6D28D9", "-", "D"),
        ("film_nwd", True, "FiLM-nwd (Channel)", "#059669", ":", "o"),
        ("spatial_softmax_attn", True, "v1:Spatial-Softmax", "#2563EB", "--", "^"),
        ("res_spatial_softmax", False, "v2:Res-Spatial-Softmax", "#DC2626", "-", "o"),
        ("mh_res_spatial_softmax", False, "v2:MH-Res-Spatial-Softmax", "#991B1B", "-", "*"),
        ("token_coattn_spatial", False, "v2:Token-CoAttn-Spatial", "#0D9488", "-", "P"),
    ]
    xs_scale = np.array([0, 1, 2])
    for key, is_ref, lbl, col, ls, mk in curve_specs:
        src = (ev["references"][key] if key in ("bc30", "film_nwd") else ev["references"][f"v1_{key}"]) if is_ref else ev["variants"][key]
        ys = [src[s]["closed_loop_success"]["mean"] for s in ("S6N3", "S7N4", "S8N6")]
        ax_b.plot(xs_scale, ys, linestyle=ls, marker=mk, color=col, linewidth=1.8, markersize=6, label=lbl)
    ax_b.set_xticks(xs_scale)
    ax_b.set_xticklabels(["S6N3\n(6×6 N=3)", "S7N4\n(7×7 N=4)", "S8N6\n(8×8 N=6)"], fontsize=8.2)
    ax_b.set_ylim(0.65, 1.00)
    ax_b.set_ylabel("Closed-loop Success")
    ax_b.set_title("(b) Cross-Scale Transfer & NoPE Effect", fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.35)
    ax_b.legend(fontsize=6.7, frameon=False, loc="lower left")

    # Panel (c): token_coattn_spatial 4-head word weights
    ax_c = fig.add_subplot(gs[0, 8:12])
    mat = np.array(diag["token_coattn_word_weights"]["heads_mean_4x5"])
    slots = diag["token_coattn_word_weights"]["slots"]
    im = ax_c.imshow(mat, cmap="Blues", vmin=0.0, vmax=max(0.6, float(mat.max())), aspect="auto")
    for r in range(4):
        for c in range(5):
            val = mat[r, c]
            ax_c.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=8.5, color="white" if val > mat.max() * 0.55 else "black", fontweight="bold")
    ax_c.set_xticks(np.arange(5))
    ax_c.set_xticklabels(slots, fontsize=8.2)
    ax_c.set_yticks(np.arange(4))
    ax_c.set_yticklabels([f"Head {i}" for i in range(4)], fontsize=8.2)
    ax_c.set_title(f"(c) Token-CoAttn Word Weights (mean over {diag['token_coattn_word_weights']['n_episodes_len5']} eps)", fontweight="bold")
    fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)

    # Bottom row: 4 heads 7x7 spatial attention maps (mh_res_spatial_softmax vs token_coattn_spatial - choose best performer among multi-head softmax models!)
    # Compare which of mh_res_spatial_softmax / token_coattn_spatial has higher S8N6 or show token_coattn_spatial's 4 heads on seed 4501 t=2
    best_mh = "mh_res_spatial_softmax"
    if ev["variants"]["token_coattn_spatial"]["S6N3"]["closed_loop_success"]["mean"] >= ev["variants"]["mh_res_spatial_softmax"]["S6N3"]["closed_loop_success"]["mean"]:
        best_mh = "token_coattn_spatial"
    m_best, _, _ = load_run_universal(f"runs/b2_attn_v2_{best_mh}_seed0")
    env = make_env(ENV_ID)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=4501)
        for a in (0, 2):
            obs, _, _, _, _ = env.step(a)
    env.close()
    img_b, tok_b, dir_b = encode_batch([obs])
    sp_b = m_best.extract_diagnostics(img_b, tok_b, dir_b)
    head_maps = sp_b["head_maps"]  # (4, 7, 7)

    for k in range(4):
        ax_k = fig.add_subplot(gs[1, k * 3 : (k + 1) * 3])
        draw_egocentric_grid_with_map(
            ax_k,
            obs["image"],
            head_maps[k],
            f"{SHORT_LABELS[best_mh].replace(chr(10), ' ')} · Head {k} (top: {['go','to','the','<col>','<type>'][int(np.argmax(sp_b['token_weights'][k, :5]))]}={float(sp_b['token_weights'][k, :5].max()):.2f})",
        )

    fig.savefig(ROOT / "reports/figures/b2_attn/b2_attn_v2_mechanisms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(ev: dict, f20: dict, pr: dict, fl: dict, diag: dict) -> None:
    print("== 7. Writing reports/b2_attn_v2_summary.md ==")
    order = [
        "bc30",
        "film_nwd",
        "spatial_attn_gate",
        "spatial_softmax_attn",
        "cross_attn_mha",
        "film_spatial_attn",
        "film_cross_mha",
        *ATTN_V2_VARIANTS,
    ]
    lines = [
        "# 支线 2 扩展消融 v2 · Attention 改进架构与深度分析汇总",
        "",
        "> 由 `branches/b2_attn/eval_v2_and_analyze.py` 自动汇总生成（`ddof=0` 总体标准差，n=3 初始化种子）。",
        "",
        "| 方法 | 参数量 | val 准确率 | **S6N3 Success** | **S7N4 迁移** | **S8N6 (10000:)** | **20% 数据 S6** | 颜色敏感度 | 表征+覆盖失败 (/300) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in order:
        if k in ("bc30", "film_nwd"):
            d6 = ev["references"][k]["S6N3"]
            d7 = ev["references"][k]["S7N4"]
            d8 = ev["references"][k]["S8N6"]
            f20_k = "bc_step_f020" if k == "bc30" else "film_step_f020"
            s20 = f"{f20['references'][f20_k]['S6N3']['success_mean']:.3f} ± {f20['references'][f20_k]['S6N3']['success_sd']:.3f}"
        elif f"v1_{k}" in ev["references"]:
            d6 = ev["references"][f"v1_{k}"]["S6N3"]
            d7 = ev["references"][f"v1_{k}"]["S7N4"]
            d8 = ev["references"][f"v1_{k}"]["S8N6"]
            if f"v1_{k}" in f20["variants"]:
                m20 = f20["variants"][f"v1_{k}"]["S6N3"]["closed_loop_success"]
                s20 = f"{m20['mean']:.3f} ± {m20['sd']:.3f}"
            elif f"v1_{k}" in f20["references"]:
                m20 = f20["references"][f"v1_{k}"]["S6N3"]["closed_loop_success"]
                s20 = f"{m20['mean']:.3f} ± {m20['sd']:.3f}"
            else:
                s20 = "—"
        else:
            d6 = ev["variants"][k]["S6N3"]
            d7 = ev["variants"][k]["S7N4"]
            d8 = ev["variants"][k]["S8N6"]
            if k in f20["variants"]:
                m20 = f20["variants"][k]["S6N3"]["closed_loop_success"]
                s20 = f"{m20['mean']:.3f} ± {m20['sd']:.3f}"
            else:
                s20 = "—"
        pr_src = pr["references"][k] if k in pr["references"] else pr["by_variant"][k]
        cs = pr_src["color_swap_sensitivity"]
        fa = fl["references"]["aggregate"][k] if k in fl["references"]["aggregate"] else fl["aggregate"][k]
        lines.append(
            f"| `{k}` | {d6['n_params']:,} | {d6['val_action_acc']['mean']:.3f} ± {d6['val_action_acc']['sd']:.3f} | "
            f"**{d6['closed_loop_success']['mean']:.3f} ± {d6['closed_loop_success']['sd']:.3f}** | "
            f"{d7['closed_loop_success']['mean']:.3f} ± {d7['closed_loop_success']['sd']:.3f} | "
            f"{d8['closed_loop_success']['mean']:.3f} ± {d8['closed_loop_success']['sd']:.3f} | "
            f"{s20} | {cs['mean']:.3f} ± {cs['sd']:.3f} | "
            f"{fa['n_failure']['mean']:.1f} ({fa['representation']['mean']:.1f}+{fa['state_coverage']['mean']:.1f}) |"
        )

    lines.extend(["", "## 个案修复对比（3 seeds 通过数：4501, 4516, 4560, 4568, 4502）", "", "| 方法 | 4501 (绕障) | 4516 (紫盒) | 4560 (绿球) | 4568 (盲区) | 4502 (拥挤紫匙) |", "|---|---|---|---|---|---|"])
    for k in order:
        row = [f"`{k}`"]
        c_src = fl["references"]["cases"][k] if k in fl["references"]["cases"] else fl["cases"][k]
        for cs in CASE_SEEDS:
            c = c_src[str(cs)]
            lens = [str(x["episode_len"]) if x["success"] else "T" for x in c["per_seed"]]
            row.append(f"{c['n_solved']}/3 ({'/'.join(lens)})")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Seed 4501 (t=2) 有效调制倍数对比（障碍物保留诊断）",
        "",
        "| 模型 | 目标 `blue_key` | 阻挡障碍 `yellow_box` | 墙壁 `wall` | 空白格 `empty` |",
        "|---|---|---|---|---|",
    ])
    for mk, mv in diag["seed4501_t2_effective_modulation"].items():
        def _fmt(key):
            return f"{mv[key]['mean']:.2f} ± {mv[key]['sd']:.2f}" if key in mv and mv[key]["n"] else "—"
        lines.append(f"| `{mk}` | {_fmt('target')} | {_fmt('blocking_yellow_box')} | {_fmt('walls')} | {_fmt('empty')} |")

    (ROOT / "reports/b2_attn_v2_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    if args.plots_only:
        ev = json.loads((ROOT / "reports/b2_attn_v2_eval.json").read_text())
        f20 = json.loads((ROOT / "reports/b2_attn_v2_f020_eval.json").read_text())
        pr = json.loads((ROOT / "reports/b2_attn_v2_probe.json").read_text())
        fl = json.loads((ROOT / "reports/b2_attn_v2_failures.json").read_text())
        diag = json.loads((ROOT / "reports/b2_attn_v2_diagnostics.json").read_text())
    else:
        ev = run_main_eval()
        f20 = run_f020_eval()
        pr = run_probe()
        fl = run_failures()
        diag = run_diagnostics()
    make_benchmark_figure(ev, f20, pr, fl)
    make_mechanisms_figure(ev, diag)
    write_summary_md(ev, f20, pr, fl, diag)
    print(f"ALL B2_ATTN_V2 EVALUATIONS & FIGURES COMPLETE IN {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
