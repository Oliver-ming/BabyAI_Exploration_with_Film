"""支线共用诊断：数据标签一致性（DAgger 与专家数据都适用）。

回答一个问题：**同一个 (image, direction, mission) 是否对应多个不同的动作标签？**
- 若对应多个 → 该状态的标签存在内在歧义，模型无论多强都会在这些样本上出错，
  验证准确率会被人为压低（这是 DAgger 常见现象，也是 §4 的"红线"）
- 若几乎不重复 → 标签函数是确定的，掉准确率只能来自"状态覆盖变了"

用法:
    python branches/_shared/check_label_conflicts.py --data data/b3_dagger_400.pkl
    python branches/_shared/check_label_conflicts.py --data data/expert_5000.pkl --split train
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path


def load_episodes(path: str, split: str | None) -> list[dict]:
    obj = pickle.load(open(path, "rb"))
    if isinstance(obj, dict) and "episodes" in obj:  # 支线采集格式
        eps = obj["episodes"]
        if split:
            eps = [e for e in eps if e.get("split", split) == split]
        return eps
    eps = obj  # 官方专家格式
    if split:
        eps = [e for e in eps if e.get("split", "train") == split]
    return eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="", help="只统计某个 split（专家数据用 train/val）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    eps = load_episodes(args.data, args.split or None)
    labels: dict[tuple, set] = collections.defaultdict(set)
    counts: dict[tuple, int] = collections.defaultdict(int)

    for e in eps:
        mission = e["mission"]
        for img, d, a in zip(e["images"], e["directions"], e["actions"]):
            k = (img.tobytes(), int(d), mission)
            labels[k].add(int(a))
            counts[k] += 1

    total_states = len(labels)
    total_trans = sum(counts.values())
    ambiguous = [k for k, v in labels.items() if len(v) > 1]
    affected = sum(counts[k] for k in ambiguous)

    # 冲突状态的典型例子
    examples = []
    for k in ambiguous[:5]:
        examples.append(
            {
                "mission": k[2],
                "direction": int(k[1]),
                "n_labels": len(labels[k]),
                "labels": sorted(labels[k]),
                "count": counts[k],
            }
        )

    out = {
        "data": args.data,
        "split": args.split or None,
        "n_episodes": len(eps),
        "n_transitions": total_trans,
        "n_unique_states": total_states,
        "n_ambiguous_states": len(ambiguous),
        "ambiguous_state_rate": round(len(ambiguous) / max(1, total_states), 6),
        "affected_transition_rate": round(affected / max(1, total_trans), 6),
        "revisited_state_rate": round(
            sum(1 for k in counts if counts[k] > 1) / max(1, total_states), 6
        ),
        "examples": examples,
    }

    print(
        f"{args.data}: {len(eps)} eps / {total_trans} transitions / "
        f"{total_states} unique (img,dir,mission)"
    )
    print(
        f"  重复访问的状态: {out['revisited_state_rate']:.1%}　"
        f"多标签状态: {len(ambiguous)} ({out['ambiguous_state_rate']:.2%})　"
        f"受影响转移: {affected} ({out['affected_transition_rate']:.2%})"
    )
    for ex in examples:
        print(f"  e.g. {ex['mission']} dir={ex['direction']} -> labels {ex['labels']} x{ex['count']}")

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
