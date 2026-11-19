"""支线 3 (v2) · 交互式模仿学习统一采集器（支持 Unmemorized Seeds / Loop-Unsticking / Active Querying）。

改进依据（详见 branches/b3_dagger/LOG.md §12）：
  1. 消除训练 seed 记忆盲区（Chevalier-Boisvert et al., ICLR 2019; mila-iqia/babyai `train_intelligent_expert.py`）：
     支持显式非重叠训练种子池（`train_tail1500` / `fresh_r1` / `fresh_r2`），杜绝在已背熟的训练种子上空转。
  2. 死锁解锁采样（Loop-Unsticking，受 LazyDAgger [Hoque et al., ICRA 2021] 与 β-DAgger [Ross et al., 2011] 启发）：
     GoToLocal 中 100% 失败均为原地振荡/顶墙跑满 36 步。检测到同 episode 内 `(image, dir)` 重复时，
     记录专家纠偏标签并由专家执行 1 步脱离死锁，随后交还策略控制权，使单条轨迹可覆盖后续多处恢复状态。
  3. 主动查询门控（Active Querying：SafeDAgger [AAAI 2017] / EnsembleDAgger [IROS 2019] / ThriftyDAgger [CoRL 2021] / HG-DAgger [ICRA 2019]）：
     支持 `all`（全步查询）、`disagree`（仅保留 π(s)≠a* 的分歧状态）、`margin`（单模型置信度裕度门控，跳过高置信步的专家查询）、
     `ensemble`（3-seed 集成投票分歧或低裕度时才查询专家，精确统计 `n_oracle_queries`）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import assert_no_test_seed, make_env  # noqa: E402
from common import ENV_ID, TRAIN_SEEDS, VAL_SEEDS, encode_batch, load_run  # noqa: E402


def load_policy_model(run_dir: str):
    """兼容加载冻结 BCNet 或 b2 FiLMBCNet。"""
    cfg = json.loads((Path(run_dir) / "config.json").read_text())
    if cfg.get("branch") == "b2_film" or "variant" in cfg.get("arch", {}):
        from film_policy import load_film_run
        return load_film_run(run_dir)
    return load_run(run_dir)


@torch.no_grad()
def policy_probs(models: list[torch.nn.Module], obs: dict) -> tuple[int, float, bool]:
    """返回 (主模型贪心动作, 平均概率 top1-top2 裕度 margin, 集成成员间是否存在动作分歧)。"""
    img, toks, d = encode_batch([obs], getattr(models[0], "blank_language", False))
    probs_list = [torch.softmax(m(img, toks, d), dim=-1)[0] for m in models]
    greedy_actions = [int(p.argmax().item()) for p in probs_list]
    ens_disagree = len(set(greedy_actions)) > 1

    mean_p = torch.stack(probs_list, dim=0).mean(dim=0)
    top2 = torch.topk(mean_p, k=2).values
    margin = float((top2[0] - top2[1]).item())
    return greedy_actions[0], margin, ens_disagree


def resolve_seeds(pool_spec: str, allow_val_probe: bool = False) -> list[int]:
    """解析采集种子池，并强制校验与 VAL_SEEDS / TEST_SEEDS / TEST_SEEDS_S7 严格不相交。"""
    if pool_spec == "train_head1500":
        seeds = list(TRAIN_SEEDS[:1500])
    elif pool_spec == "train_tail1500":
        # 严格 4500 预算切分模式：前 3000 条训 π1，后 1500 条做未见房间交互采集
        seeds = list(TRAIN_SEEDS[3000:4500])
    elif pool_spec == "fresh_r1":
        seeds = list(range(5300, 6800))
    elif pool_spec == "fresh_r2":
        seeds = list(range(6800, 8300))
    elif pool_spec == "val_probe":
        if not allow_val_probe:
            raise ValueError("val_probe requires --allow-val-probe")
        seeds = list(VAL_SEEDS)
    elif ":" in pool_spec:
        a, b = pool_spec.split(":")
        seeds = list(range(int(a), int(b)))
    else:
        raise ValueError(f"未知 seed_pool: {pool_spec}")

    assert_no_test_seed(seeds)
    if not allow_val_probe:
        assert not (set(seeds) & set(VAL_SEEDS)), "❌ 采集种子池混入 VAL_SEEDS！"
    return seeds


def collect_episode(
    env,
    models: list[torch.nn.Module],
    bot_cls,
    del_err_cls,
    seed: int,
    exec_mode: str,
    query_gate: str,
    beta: float,
    margin_thresh: float,
) -> dict:
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    bot = bot_cls(u)
    rng = np.random.default_rng(seed * 10007 + 31)

    images, dirs, missions, expert_actions, policy_actions = [], [], [], [], []
    disagree_flags, unstick_flags = [], []

    visited_ep: set[tuple[bytes, int]] = set()
    n_env_steps = 0
    n_oracle_queries = 0
    n_unsticks = 0
    terminated = truncated = False
    had_unstick = False

    for _t in range(u.max_steps):
        state_ep_key = (obs["image"].tobytes(), int(obs["direction"]))
        is_loop = state_ep_key in visited_ep

        if exec_mode == "expert":
            try:
                ea = bot.replan(None)
            except (del_err_cls, AssertionError, IndexError):
                break
            if ea is None:
                break
            expert_a = int(ea)
            n_oracle_queries += 1
            images.append(obs["image"].copy())
            dirs.append(int(obs["direction"]))
            missions.append(obs["mission"])
            expert_actions.append(expert_a)
            policy_actions.append(expert_a)
            disagree_flags.append(False)
            unstick_flags.append(False)
            taken_a = expert_a
        else:
            policy_a, margin, ens_disagree = policy_probs(models, obs)

            # 判断主动查询门控是否触发专家查询
            if query_gate in ("all", "disagree"):
                should_query = True
            elif query_gate == "margin":
                should_query = (margin < margin_thresh) or is_loop
            elif query_gate == "ensemble":
                should_query = ens_disagree or (margin < margin_thresh) or is_loop
            else:
                raise ValueError(query_gate)

            # 若需要执行 beta 混合或死锁解锁，也必须向专家查询当前步动作
            want_beta_step = (exec_mode == "beta") and (rng.random() < beta)
            want_unstick_step = (exec_mode == "unstick") and is_loop
            if want_beta_step or want_unstick_step:
                should_query = True

            expert_a = None
            if should_query:
                try:
                    ea = bot.replan(None)
                except (del_err_cls, AssertionError, IndexError):
                    break
                if ea is None:
                    break
                expert_a = int(ea)
                n_oracle_queries += 1

                is_dis = (policy_a != expert_a)
                # disagree 门控：查询后仅保留策略与专家分歧的关键状态
                keep_sample = True if query_gate != "disagree" else is_dis
                if keep_sample:
                    images.append(obs["image"].copy())
                    dirs.append(int(obs["direction"]))
                    missions.append(obs["mission"])
                    expert_actions.append(expert_a)
                    policy_actions.append(policy_a)
                    disagree_flags.append(bool(is_dis))
                    unstick_flags.append(bool(want_unstick_step))
            else:
                # 跳过专家动作查询，仅同步视野掩码（零人工标注成本）
                bot._process_obs()
                bot._remember_current_state()

            if want_unstick_step and expert_a is not None:
                taken_a = expert_a
                n_unsticks += 1
                had_unstick = True
            elif want_beta_step and expert_a is not None:
                taken_a = expert_a
            else:
                taken_a = policy_a

        visited_ep.add(state_ep_key)
        n_env_steps += 1
        obs, _r, terminated, truncated, _ = env.step(taken_a)
        if terminated or truncated:
            break

    return {
        "seed": seed,
        "exec_mode": exec_mode,
        "query_gate": query_gate,
        "mission": missions[0] if missions else obs.get("mission"),
        "images": images,
        "directions": dirs,
        "actions": expert_actions,
        "policy_actions": policy_actions,
        "disagree": disagree_flags,
        "unstick": unstick_flags,
        "episode_len": len(images),
        "env_steps": n_env_steps,
        "oracle_queries": n_oracle_queries,
        "n_unsticks": n_unsticks,
        "rollout_terminated": bool(terminated),
        # 纯策略无协助成功标记（仅当未发生专家接管且成功到达时为 True）
        "autonomous_success": bool(terminated) and (not had_unstick) and (exec_mode not in ("expert", "beta")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/b3_bc30_seed0", help="主执行策略目录")
    ap.add_argument("--ensemble-runs", nargs="*", default=[], help="EnsembleDAgger 附加策略目录列表")
    ap.add_argument("--env-id", default=ENV_ID)
    ap.add_argument("--seed-pool", default="fresh_r1",
                    help="train_tail1500 | fresh_r1 | fresh_r2 | train_head1500 | A:B")
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--exec-mode", default="unstick", choices=["expert", "greedy", "beta", "unstick"])
    ap.add_argument("--query-gate", default="all", choices=["all", "disagree", "margin", "ensemble"])
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--margin-thresh", type=float, default=0.50)
    ap.add_argument("--allow-val-probe", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    seeds = resolve_seeds(args.seed_pool, args.allow_val_probe)[: args.episodes]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    models = []
    cfg_main = {}
    if args.exec_mode != "expert":
        m0, cfg_main, _ = load_policy_model(args.run)
        models.append(m0)
        for er in args.ensemble_runs:
            if Path(er).resolve() != Path(args.run).resolve():
                mi, _, _ = load_policy_model(er)
                models.append(mi)
        if args.query_gate == "ensemble" and len(models) < 2:
            raise SystemExit("query_gate=ensemble 至少需要 2 个模型（请传 --ensemble-runs）")

    from minigrid.utils.baby_ai_bot import BabyAIBot, DisappearedBoxError

    env = make_env(args.env_id)
    episodes = []
    t0 = time.time()

    with contextlib.redirect_stdout(io.StringIO()):
        for s in seeds:
            ep = collect_episode(
                env, models, BabyAIBot, DisappearedBoxError, s,
                exec_mode=args.exec_mode,
                query_gate=args.query_gate,
                beta=args.beta,
                margin_thresh=args.margin_thresh,
            )
            episodes.append(ep)
    env.close()

    kept_eps = [e for e in episodes if e["images"]]
    n_env_steps = sum(e["env_steps"] for e in episodes)
    n_queries = sum(e["oracle_queries"] for e in episodes)
    n_saved = sum(e["episode_len"] for e in kept_eps)
    n_disagree = sum(sum(e["disagree"]) for e in kept_eps)
    n_unsticks = sum(e["n_unsticks"] for e in episodes)

    meta = {
        "branch": "b3_dagger_v2",
        "exec_mode": args.exec_mode,
        "query_gate": args.query_gate,
        "beta": args.beta if args.exec_mode == "beta" else None,
        "margin_thresh": args.margin_thresh if args.query_gate in ("margin", "ensemble") else None,
        "policy_run": args.run if args.exec_mode != "expert" else None,
        "ensemble_runs": args.ensemble_runs if args.query_gate == "ensemble" else [],
        "policy_val_acc": cfg_main.get("result", {}).get("best_val_acc"),
        "env_id": args.env_id,
        "seed_pool": args.seed_pool,
        "seed_range": f"{seeds[0]}:{seeds[-1] + 1}",
        "n_episodes_rolled": len(episodes),
        "n_episodes_nonempty": len(kept_eps),
        "n_env_steps": n_env_steps,
        "n_oracle_queries": n_queries,
        "query_rate": round(n_queries / max(1, n_env_steps), 6),
        "n_transitions_saved": n_saved,
        "n_disagree_saved": n_disagree,
        "n_unsticks": n_unsticks,
        "rollout_terminated_rate": round(float(np.mean([e["rollout_terminated"] for e in episodes])), 6),
        "seeds": [e["seed"] for e in kept_eps],
        "elapsed_sec": round(time.time() - t0, 2),
    }

    with open(out, "wb") as f:
        pickle.dump({"meta": meta, "episodes": kept_eps}, f)

    print(
        f"[{args.exec_mode}/{args.query_gate}] pool={args.seed_pool} "
        f"env_steps={n_env_steps} queries={n_queries} ({meta["query_rate"]:.1%}) "
        f"saved={n_saved} (disagree={n_disagree}, unsticks={n_unsticks}) "
        f"term={meta["rollout_terminated_rate"]:.3f} ({meta["elapsed_sec"]}s) -> {out}"
    )


if __name__ == "__main__":
    main()
