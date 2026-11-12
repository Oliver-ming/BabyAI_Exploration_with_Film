"""P1-3 · 回放与可视化（§8）+ P2-3 失败案例采集（§9.3）。

用法:
    python src/render_episodes.py --run runs/baseline_seed0 \
        --n-success 2 --require-types orientation_lock,attribute_confusion,distractor_priority \
        --outdir videos --faildir reports/failures

产出：
    videos/success_seed{seed}.gif / videos/failure_seed{seed}.gif
    reports/failures/seed{seed}.json  （逐步动作日志 + 明确归因）
    reports/failures/seed{seed}.png   （首 / 中 / 末三帧拼图）

失败归因（`classify_failure`）
  §8.3 规定的三类：
    attribute_confusion : 最终盯着的是颜色/类型不对的物体（把 X 认成 Y）
    distractor_priority : 朝向了更近的同类干扰物，而不是指令指定的那个
    orientation_lock    : 已经站在目标旁边（距离 1）但朝向不对，原地转圈到超时
  本数据上额外出现、同样属于分布偏移的两类（作为补充类别报告）：
    turn_oscillation    : 原地 left/right 交替振荡，位置一格不动，从不 forward
    forward_stall       : 一直 forward 顶着墙/边界，从不转向
  兜底：
    wander_timeout      : 全程没靠近过目标
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import minigrid  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ENV_ID, load_run, predict_actions  # noqa: E402

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def _cell_desc(cell):
    if cell is None:
        return None
    return {"type": cell.type, "color": getattr(cell, "color", None)}


def _manhattan(a, b) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def run_episode(env, model, seed: int, render: bool = False) -> dict:
    """闭环跑一条 episode，记录逐步 (状态, 动作) 与末尾状态。"""
    obs, _ = env.reset(seed=seed)
    u = env.unwrapped
    mission = obs["mission"]

    desc = getattr(u.instrs, "desc", None)
    target_poss = [tuple(int(v) for v in p) for p in (desc.obj_poss if desc else [])]
    target_color = desc.color if desc is not None else None
    target_type = desc.type if desc is not None else None

    # 场景里所有物体（用来判断"盯错物体"和"被更近的干扰物吸引"）
    objects = []
    for i in range(u.grid.width):
        for j in range(u.grid.height):
            c = u.grid.get(i, j)
            if c is not None and c.type not in ("wall", "door"):
                objects.append(
                    {"pos": [i, j], "type": c.type, "color": getattr(c, "color", None)}
                )

    frames, steps = [], []
    terminated = truncated = False
    if render:
        frames.append(env.render())

    for t in range(u.max_steps):
        a = predict_actions(model, [obs])[0]
        agent_pos = tuple(int(v) for v in u.agent_pos)
        front = tuple(int(v) for v in u.front_pos)
        steps.append(
            {
                "t": t,
                "action": ACTION_NAMES[a],
                "agent_pos": list(agent_pos),
                "agent_dir": int(u.agent_dir),
                "front_cell": _cell_desc(u.grid.get(*front)),
                "facing_target": front in target_poss,
                "manhattan_to_target": min(_manhattan(agent_pos, p) for p in target_poss)
                if target_poss
                else None,
            }
        )
        obs, _r, terminated, truncated, _ = env.step(a)
        if render:
            frames.append(env.render())
        if terminated or truncated:
            break

    # 末尾状态（step 之后！成功判定就是基于这个状态）
    final_pos = tuple(int(v) for v in u.agent_pos)
    final_front = tuple(int(v) for v in u.front_pos)
    final_front_cell = _cell_desc(u.grid.get(*final_front))
    final_facing_target = final_front in target_poss
    final_dist = min(_manhattan(final_pos, p) for p in target_poss) if target_poss else None

    # 干扰物：同类不同色 / 同色不同类 / 完全同类的其它实例
    distractors = []
    for o in objects:
        if tuple(o["pos"]) in target_poss:
            continue
        if o["type"] == target_type or o["color"] == target_color:
            distractors.append(o)
    distractor_dists = [_manhattan(final_pos, d["pos"]) for d in distractors]

    return {
        "seed": seed,
        "mission": mission,
        "success": bool(terminated),
        "truncated": bool(truncated),
        "episode_len": len(steps),
        "steps": steps,
        "frames": frames,
        "target_poss": [list(p) for p in target_poss],
        "target": {"color": target_color, "type": target_type},
        "objects": objects,
        "distractors": distractors,
        "distractor_dists": distractor_dists,
        "final_agent_pos": list(final_pos),
        "final_agent_dir": int(u.agent_dir),
        "final_front_cell": final_front_cell,
        "final_facing_target": bool(final_facing_target),
        "final_distance_to_target": final_dist,
        "time_left": u.max_steps - len(steps),
    }


def classify_failure(traj: dict) -> tuple[str, str]:
    steps = traj["steps"]
    if not steps:
        return "wander_timeout", "episode 一步都没走"

    mission = traj["mission"]
    color, obj_type = mission.split()[-2], mission.split()[-1]
    tail = steps[-5:]
    tail_pos = tuple(tail[-1]["agent_pos"])
    same_pos = len({tuple(s["agent_pos"]) for s in tail}) == 1
    only_turns = all(s["action"] in ("left", "right") for s in tail)
    only_forward = all(s["action"] == "forward" for s in tail)
    best = min((s["manhattan_to_target"] for s in steps if s["manhattan_to_target"] is not None),
               default=None)
    final_d = traj["final_distance_to_target"]
    fc = traj["final_front_cell"]

    # ---- attribute_confusion：面前摆着一个「属性相近」的物体（同类不同色 / 同色不同类）----
    # 只有真正"可混淆"的属性错才归为这一类；面前是完全无关的物体不算属性认错
    if fc and (
        (fc["type"] == obj_type and fc["color"] != color)
        or (fc["color"] == color and fc["type"] != obj_type)
    ):
        wrong = f"{fc['color']} {fc['type']}"
        kind = "同类不同色" if fc["type"] == obj_type else "同色不同类"
        return "attribute_confusion", (
            f"最终停在 {traj['final_agent_pos']} 面向 {wrong}（{kind}），但指令要的是 "
            f"{color} {obj_type}——模型把目标属性认错了。场景里确实有 {color} {obj_type}"
            f"（坐标 {traj['target_poss']}），它被这个属性相近的物体抢走了注意力"
        )

    # ---- orientation_lock：已经在目标旁边（距离 1），但朝向不对 ----
    if final_d == 1 and not traj["final_facing_target"]:
        spin = same_pos and only_turns
        return "orientation_lock", (
            f"最终位置 {traj['final_agent_pos']} 距目标只有 {final_d} 格（目标 "
            f"{traj['target_poss']}），但朝向 {traj['final_agent_dir']} 没有面向它"
            + ("，而且最后几步一直在原地左右转" if spin else "")
            + "——GoToLocal 的成功条件是「站在旁边 + 面向它」，位置对了朝向错了照样失败"
        )

    # ---- distractor_priority：最后停在离干扰物比离目标更近的位置 ----
    if traj["distractor_dists"] and final_d is not None:
        nearest_dis = min(traj["distractor_dists"])
        if nearest_dis < final_d:
            return "distractor_priority", (
                f"最终站在 {traj['final_agent_pos']}：离最近的干扰物 {nearest_dis} 格，"
                f"离真正的目标 {final_d} 格——走去更近的同类物体了"
            )

    # ---- 补充类：原地振荡 / 顶墙 ----
    if same_pos and only_turns:
        return "turn_oscillation", (
            f"最后 5 步停在 {tail_pos} 原地 left/right 交替，位置一格没动，离目标还有 "
            f"{final_d} 格——模型对「该往哪边转」两侧 logit 交替占优，永远不敢往前走"
        )
    if same_pos and only_forward:
        return "forward_stall", (
            f"最后 5 步停在 {tail_pos} 重复 forward（顶着墙/边界），离目标还有 {final_d} 格"
            "——训练集里 forward 占 57%，一旦进入「前方不可走」的状态就成为死循环"
        )

    if best is not None and best <= 2:
        return "approach_then_lost", (
            f"曾经走到距目标只有 {best} 格的位置，但没面向它，之后又转开，最终停在 "
            f"{traj['final_agent_pos']}（距目标 {final_d} 格）——"
            "它其实已经找到了目标，只是没有完成「面向」这个动作就离开了"
        )
    return "wander_timeout", (
        f"全程离目标最近也有 {best} 格，{traj['episode_len']} 步耗尽——"
        "模型在专家从未访问过的状态里拿不到可用的方向信息"
    )


def save_gif(frames, path: Path, fps: int = 3, repeat_last: int = 6) -> None:
    """存 gif；末尾重复几帧，让"卡住"的结局在播放时能看清。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    imageio.mimsave(path, frames + [frames[-1]] * repeat_last, fps=fps, loop=0)


def save_strip(frames, path: Path) -> None:
    """首 / 中 / 末三帧横向拼图（§11 图 7 素材）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    idx = [0, len(frames) // 2, len(frames) - 1]
    imageio.imwrite(path, np.concatenate([frames[i] for i in idx], axis=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/baseline_seed0")
    ap.add_argument("--env-id", default=ENV_ID)
    ap.add_argument("--seeds", default="test")
    ap.add_argument("--n-success", type=int, default=2)
    ap.add_argument("--n-failure", type=int, default=3,
                    help="最少失败样本数（不足 --require-types 时以类型为准）")
    ap.add_argument("--require-types", default="orientation_lock,attribute_confusion,distractor_priority",
                    help="逐个凑齐这些失败类型（逗号分隔）；留空则只按数量采")
    ap.add_argument("--scan", type=int, default=300, help="最多扫描多少个 seed")
    ap.add_argument("--outdir", default="videos")
    ap.add_argument("--faildir", default="reports/failures")
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()

    from eval_closed_loop import parse_seeds

    seeds = parse_seeds(args.seeds)[: args.scan]
    model, cfg, _ = load_run(args.run)
    outdir, faildir = Path(args.outdir), Path(args.faildir)
    outdir.mkdir(parents=True, exist_ok=True)
    faildir.mkdir(parents=True, exist_ok=True)
    required = [t for t in args.require_types.split(",") if t]

    env = gym.make(args.env_id, render_mode="rgb_array")  # 服务器必须 rgb_array
    records: list[dict] = []
    used: set[int] = set()
    type_counts: dict[str, int] = collections.Counter()
    n_ok = 0

    def handle(traj: dict, kind: str) -> None:
        gif = outdir / f"{kind}_seed{traj['seed']}.gif"
        save_gif(traj["frames"], gif, args.fps)
        rec = {
            "seed": traj["seed"],
            "mission": traj["mission"],
            "outcome": kind,
            "gif": str(gif),
            "episode_len": traj["episode_len"],
            "target": traj["target"],
            "target_poss": traj["target_poss"],
            "final_agent_pos": traj["final_agent_pos"],
            "final_agent_dir": traj["final_agent_dir"],
            "final_front_cell": traj["final_front_cell"],
            "final_distance_to_target": traj["final_distance_to_target"],
            "final_facing_target": traj["final_facing_target"],
            "steps": traj["steps"],
        }
        if kind == "failure":
            ftype, note = classify_failure(traj)
            rec["failure_type"] = ftype
            rec["note"] = note
            rec["modification_would_help"] = (
                "DAgger 会在这个具体状态上让 BabyAIBot 标注正确的转向/前进动作并加入训练集，"
                "直接补上该状态下的正确行为。"
                if ftype
                in (
                    "turn_oscillation",
                    "forward_stall",
                    "orientation_lock",
                    "approach_then_lost",
                    "wander_timeout",
                )
                else "这一类（属性/干扰物混淆）本质上是表征能力问题，仅靠加同分布样本收益有限，"
                     "需要在报告里诚实说明 DAgger 未必能解决"
            )
            (faildir / f"seed{traj['seed']}.json").write_text(
                json.dumps(rec, indent=2, ensure_ascii=False)
            )
            save_strip(traj["frames"], faildir / f"seed{traj['seed']}.png")
            print(f"failure seed={traj['seed']} [{ftype}] {traj['mission']}")
        else:
            print(f"success seed={traj['seed']} {traj['mission']}")
        records.append(rec)

    with contextlib.redirect_stdout(io.StringIO()):
        # 阶段 1：采集成功样本
        for seed in seeds:
            if n_ok >= args.n_success:
                break
            traj = run_episode(env, model, seed, render=True)
            used.add(seed)
            if traj["success"]:
                n_ok += 1
                handle(traj, "success")

        # 阶段 2：逐个凑齐 required 失败类型
        for want in required:
            if type_counts.get(want, 0) > 0:
                continue
            for seed in seeds:
                if seed in used:
                    continue
                traj = run_episode(env, model, seed, render=True)
                used.add(seed)
                if traj["success"]:
                    continue
                ftype, _ = classify_failure(traj)
                if ftype == want:
                    type_counts[ftype] += 1
                    handle(traj, "failure")
                    break
                if type_counts[ftype] == 0:
                    # 其它类型也先留一个，便于报告多列几类（受 n_failure 上限约束）
                    if sum(type_counts.values()) < args.n_failure:
                        type_counts[ftype] += 1
                        handle(traj, "failure")

        # 阶段 3：失败样本数量补足
        for seed in seeds:
            if sum(type_counts.values()) >= args.n_failure:
                break
            if seed in used:
                continue
            traj = run_episode(env, model, seed, render=True)
            used.add(seed)
            if traj["success"]:
                continue
            ftype, _ = classify_failure(traj)
            type_counts[ftype] += 1
            handle(traj, "failure")

    env.close()

    n_fail = sum(1 for r in records if r["outcome"] == "failure")

    # 合并成一条演示视频（§8.1：30–60 秒）：成功 2 条 → 失败全部
    if records:
        seq = []
        for r in sorted(records, key=lambda x: (x["outcome"] != "success", x["seed"])):
            gif_path = Path(r["gif"])
            if not gif_path.exists():
                continue
            frames = list(imageio.mimread(gif_path))  # mimread 才能读出 gif 的全部帧
            for i, f in enumerate(frames):
                seq.append(f)
            # 段间加 4 帧定格，便于区分不同 episode
            seq += [frames[-1]] * 4
        demo = outdir / "demo_all.gif"
        # §8.1 要求合并后的演示视频 30–60 秒；用 4 fps 压到 ~50 秒
        demo_fps = max(args.fps, round(len(seq) / 55)) or 4
        imageio.mimsave(demo, seq, fps=demo_fps, loop=0)
        print(f"wrote {demo}  ({len(seq)} frames @ {demo_fps}fps，约 {len(seq) / demo_fps:.0f} 秒)")

    index = {
        "run": args.run,
        "env_id": args.env_id,
        "n_success": n_ok,
        "n_failure": n_fail,
        "failure_type_counts": dict(type_counts),
        "items": [
            {k: v for k, v in r.items() if k != "steps"} | {"n_steps": r["episode_len"]}
            for r in records
        ],
    }
    (outdir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"wrote {outdir}/index.json  ({n_ok} success, {n_fail} failure) types={dict(type_counts)}")
    print(f"合并演示视频：{outdir}/demo_all.gif")


if __name__ == "__main__":
    main()
