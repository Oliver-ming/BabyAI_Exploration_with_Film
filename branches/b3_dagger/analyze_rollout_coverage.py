"""支线 3 (v2) · 采集数据集状态覆盖与纠偏密度诊断（不涉及测试集）。

对比 v1 与 v2 各 1500-episode 采集数据集：
  - n_env_steps / n_oracle_queries / query_rate
  - unique_idm_states: 唯一 (image, dir, mission) 状态数
  - novel_idm_vs_expert_train: 相对 4500 条官方训练专家的新颖状态数
  - disagree_steps: 策略与专家动作不一致的步数
  - novel_disagree_idm_states: 唯一且新颖的分歧状态数
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import frozen_diff_guard  # noqa: E402


DATASETS = [
    ("v1_expert_extra_dup", "data/b3_expertextra_1500.pkl", "v1 额外专家(seed 0:1500 重叠)"),
    ("v2_fresh_expert", "data/b3_v2_fresh_expert1500.pkl", "v2 真实新专家(seed 5300:6800)"),
    ("v1_dagger_memorized", "data/b3_dagger1500_pol0.pkl", "v1 DAgger(seed 0:1500 记忆)"),
    ("v2_greedy_fresh", "data/b3_v2_greedy_s0.pkl", "v2 Greedy(seed 5300:6800)"),
    ("v2_beta025_fresh", "data/b3_v2_beta025_s0.pkl", "v2 β=0.25 混合(seed 5300:6800)"),
    ("v2_unstick_fresh", "data/b3_v2_unstick_s0.pkl", "v2 死锁解锁(seed 5300:6800)"),
    ("v2_qdis_fresh", "data/b3_v2_qdis_s0.pkl", "v2 仅留分歧(seed 5300:6800)"),
    ("v2_qmar_fresh", "data/b3_v2_qmar_s0.pkl", "v2 裕度门控(seed 5300:6800)"),
    ("v2_qens_fresh", "data/b3_v2_qens_s0.pkl", "v2 集成门控(seed 5300:6800)"),
]


def inspect_dataset(path: Path, seen_expert_idm: set) -> dict:
    obj = pickle.load(open(path, "rb"))
    meta = {k: v for k, v in obj["meta"].items() if k != "seeds"}
    eps = obj["episodes"]

    unique_idm = set()
    disagree_idm = set()
    disagree_steps = 0
    saved_steps = 0

    for ep in eps:
        m = ep["mission"]
        dis_list = ep.get("disagree")
        if dis_list is None:
            pa_list = ep.get("policy_actions", [])
            if pa_list and len(pa_list) == len(ep["actions"]):
                dis_list = [int(pa) != int(ea) for pa, ea in zip(pa_list, ep["actions"])]
            else:
                dis_list = [False] * len(ep["actions"])

        for img, d, dis in zip(ep["images"], ep["directions"], dis_list):
            k = (img.tobytes(), int(d), m)
            unique_idm.add(k)
            saved_steps += 1
            if dis:
                disagree_steps += 1
                disagree_idm.add(k)

    n_env_steps = meta.get("n_env_steps", saved_steps)
    n_oracle_queries = meta.get("n_oracle_queries", saved_steps)
    novel_idm = unique_idm - seen_expert_idm
    novel_dis = disagree_idm - seen_expert_idm

    return {
        "path": str(path),
        "n_episodes": len(eps),
        "n_env_steps": int(n_env_steps),
        "n_oracle_queries": int(n_oracle_queries),
        "query_rate": round(float(n_oracle_queries / max(1, n_env_steps)), 4),
        "n_saved_transitions": saved_steps,
        "unique_idm_states": len(unique_idm),
        "novel_idm_vs_expert_train": len(novel_idm),
        "novel_ratio": round(len(novel_idm) / max(1, len(unique_idm)), 4),
        "disagree_steps": disagree_steps,
        "unique_disagree_idm_states": len(disagree_idm),
        "novel_disagree_idm_states": len(novel_dis),
        "n_unsticks": int(meta.get("n_unsticks", 0)),
        "rollout_success": meta.get("rollout_terminated_rate", meta.get("dagger_policy_success_rate")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/b3_v2_rollout_coverage.json")
    args = ap.parse_args()

    frozen_diff_guard()
    expert_train = [e for e in pickle.load(open("data/expert_5000.pkl", "rb")) if e["split"] == "train"]
    seen_expert_idm = {
        (img.tobytes(), int(d), e["mission"])
        for e in expert_train
        for img, d in zip(e["images"], e["directions"])
    }

    rows = []
    for key, rel_path, desc in DATASETS:
        p = ROOT / rel_path
        if not p.exists():
            print(f"SKIP missing {rel_path}")
            continue
        info = inspect_dataset(p, seen_expert_idm)
        info["key"] = key
        info["label"] = desc
        rows.append(info)
        print(
            f"{key:22s} | queries={info["n_oracle_queries"]:5d} ({info["query_rate"]:5.1%}) | "
            f"uniq={info["unique_idm_states"]:5d} | novel={info["novel_idm_vs_expert_train"]:5d} "
            f"| dis={info["disagree_steps"]:4d} | novel_dis={info["novel_disagree_idm_states"]:4d}"
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"expert_train_unique_idm": len(seen_expert_idm), "datasets": rows}, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
