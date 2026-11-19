"""支线 3 (v2) · 统一闭环评估与失败机理分解（100% 复用冻结的 eval_closed_loop.evaluate 与 classify_failure）。"""

from __future__ import annotations

import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import frozen_diff_guard, make_env  # noqa: E402
from common import load_run, predict_actions  # noqa: E402
from eval_closed_loop import evaluate, parse_seeds  # noqa: E402
from render_episodes import classify_failure, run_episode  # noqa: E402

MECHANISM = {
    "orientation_lock": "state_coverage",
    "turn_oscillation": "state_coverage",
    "approach_then_lost": "state_coverage",
    "forward_stall": "state_coverage",
    "wander_timeout": "state_coverage",
    "attribute_confusion": "representation",
    "distractor_priority": "representation",
}


def load_any(run_dir: str):
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    if cfg.get("branch") == "b2_film" or "variant" in cfg.get("arch", {}):
        from film_policy import load_film_run
        return load_film_run(run_dir)
    return load_run(run_dir)


def group_of(run_dir: str, cfg: dict) -> str:
    name = Path(run_dir).name
    if "_seed" in name:
        return name.rsplit("_seed", 1)[0]
    return cfg.get("variant", name)


def eval_one_run(run_dir: str, env_id: str, seeds: list[int], with_failures: bool) -> dict:
    torch.set_num_threads(1)
    model, cfg, _ = load_any(run_dir)
    t0 = time.time()
    res = evaluate(lambda obs: predict_actions(model, [obs])[0], env_id, seeds)
    res.pop("_trajectories", None)

    fail_info = {}
    if with_failures:
        env = make_env(env_id)
        types = collections.Counter()
        groups = collections.Counter()
        with contextlib.redirect_stdout(io.StringIO()):
            for s in seeds:
                tr = run_episode(env, model, s, render=False)
                if not tr["success"]:
                    ft, _ = classify_failure(tr)
                    types[ft] += 1
                    groups[MECHANISM.get(ft, "other")] += 1
        env.close()
        fail_info = {
            "failure_types": dict(types),
            "failure_groups": dict(groups),
        }

    data_cfg = cfg.get("data", {})
    res_cfg = cfg.get("result", {})
    res.update(
        {
            "run": run_dir,
            "group": group_of(run_dir, cfg),
            "variant": cfg.get("variant", "bc"),
            "arch": cfg.get("arch", {}).get("variant", "bc"),
            "train_seed": cfg.get("seed"),
            "val_action_acc": res_cfg.get("best_val_acc"),
            "best_epoch": res_cfg.get("best_epoch"),
            "onpolicy_action_acc": res_cfg.get("onpolicy_action_acc"),
            "onpolicy_recovery_acc": res_cfg.get("onpolicy_recovery_acc"),
            "n_params": cfg.get("arch", {}).get("n_params"),
            "intv_weight": cfg.get("optim", {}).get("intv_weight", 1.0),
            "n_expert_train": data_cfg.get("n_expert_train"),
            "n_extra_transitions_kept": data_cfg.get("n_extra_transitions_kept", data_cfg.get("n_train_transitions")),
            "n_extra_oracle_queries": data_cfg.get("n_extra_oracle_queries"),
            "total_oracle_labels": data_cfg.get("total_oracle_labels"),
            "n_train_transitions": data_cfg.get("n_train_transitions"),
            "env_id": env_id,
            "eval_seconds": round(time.time() - t0, 2),
            **fail_info,
        }
    )
    return res


def _eval_worker(job: tuple[str, str, list[int], bool]) -> dict:
    return eval_one_run(*job)


def summarize_group(rs: list[dict], ref_by_seed: dict[int, float] | None = None) -> dict:
    succs = [r["closed_loop_success"] for r in rs]
    accs = [r["val_action_acc"] for r in rs if r.get("val_action_acc") is not None]
    rets = [r["mean_return"] for r in rs]
    touts = [r["n_timeout"] for r in rs]
    onp = [r["onpolicy_action_acc"] for r in rs if r.get("onpolicy_action_acc") is not None]
    rec = [r["onpolicy_recovery_acc"] for r in rs if r.get("onpolicy_recovery_acc") is not None]
    q_extra = [r["n_extra_oracle_queries"] for r in rs if r.get("n_extra_oracle_queries") is not None]
    q_tot = [r["total_oracle_labels"] for r in rs if r.get("total_oracle_labels") is not None]

    out = {
        "n": len(rs),
        "success_mean": round(st.mean(succs), 6),
        "success_sd": round(st.pstdev(succs) if len(succs) > 1 else 0.0, 6),
        "success_values": [round(x, 6) for x in succs],
        "val_acc_mean": round(st.mean(accs), 6) if accs else None,
        "val_acc_sd": round(st.pstdev(accs) if len(accs) > 1 else 0.0, 6) if accs else None,
        "mean_return": round(st.mean(rets), 6),
        "timeouts_mean": round(st.mean(touts), 2),
        "onpolicy_acc_mean": round(st.mean(onp), 6) if onp else None,
        "recovery_acc_mean": round(st.mean(rec), 6) if rec else None,
        "extra_oracle_queries_mean": round(st.mean(q_extra), 1) if q_extra else None,
        "total_oracle_labels_mean": round(st.mean(q_tot), 1) if q_tot else None,
    }
    if any("failure_groups" in r for r in rs):
        keys = sorted({k for r in rs for k in r.get("failure_groups", {})})
        out["failure_groups_mean"] = {
            k: round(st.mean([r.get("failure_groups", {}).get(k, 0) for r in rs]), 2) for k in keys
        }
        tkeys = sorted({k for r in rs for k in r.get("failure_types", {})})
        out["failure_types_mean"] = {
            k: round(st.mean([r.get("failure_types", {}).get(k, 0) for r in rs]), 2) for k in tkeys
        }
    if ref_by_seed:
        paired = [
            r["closed_loop_success"] - ref_by_seed[r["train_seed"]]
            for r in rs
            if r.get("train_seed") in ref_by_seed
        ]
        if paired:
            out["paired_delta_vs_ref_mean"] = round(st.mean(paired), 6)
            out["paired_delta_vs_ref_sd"] = round(st.pstdev(paired) if len(paired) > 1 else 0.0, 6)
            out["paired_all_positive"] = all(d > 0 for d in paired)
            out["paired_deltas"] = [round(d, 6) for d in paired]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref-group", default="b3_bc30", help="配对差值参考组名")
    ap.add_argument("--env-id", default="BabyAI-GoToLocalS6N3-v0")
    ap.add_argument("--seeds", default="test")
    ap.add_argument("--with-failures", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    frozen_diff_guard()
    seeds = parse_seeds(args.seeds)
    t0 = time.time()

    jobs = [(r, args.env_id, seeds, args.with_failures) for r in args.runs]
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
            results = list(ex.map(_eval_worker, jobs))
    else:
        results = [_eval_worker(j) for j in jobs]

    for r, res in zip(args.runs, results):
        print(
            f"{r:34s} [{res['group']:22s}] succ={res['closed_loop_success']:.4f} "
            f"val_acc={res['val_action_acc'] or 0:.4f} tout={res['n_timeout']:3d}"
        )

    by_group: dict[str, list[dict]] = collections.defaultdict(list)
    for r in results:
        by_group[r["group"]].append(r)

    ref_by_seed = {}
    if args.ref_group in by_group:
        ref_by_seed = {r["train_seed"]: r["closed_loop_success"] for r in by_group[args.ref_group]}

    agg = {g: summarize_group(rs, ref_by_seed) for g, rs in by_group.items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_id": args.env_id,
        "seeds": f"{seeds[0]}:{seeds[-1] + 1}",
        "ref_group": args.ref_group,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_seconds": round(time.time() - t0, 2),
        "runs": results,
        "by_group": agg,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {out} ({payload['total_seconds']}s)")


if __name__ == "__main__":
    main()
