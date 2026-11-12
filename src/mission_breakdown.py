"""P2-1 · mission 类型分解（§9.1）-> reports/mission_breakdown.csv + 说明 md。

用法:
    python src/mission_breakdown.py \
        --eval reports/baseline_eval.json \
        --train-data data/expert_train.pkl \
        --out reports/mission_breakdown.csv

输出列（§9.1 规定）：
    mission,color,type,article,n_train_episodes,n_test_episodes,success_rate

额外列（便于报告引用）：
    success_rate_sd   三个 seed 之间的标准差
    low_coverage      训练样本 < 20 时为 1（成功率不可靠，报告须标注）
    verb              指令里的动词（恒为 go to，留作扩展）
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COLORS, TYPES, parse_mission  # noqa: E402

LOW = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="reports/baseline_eval.json")
    ap.add_argument("--data", default="data/expert_5000.pkl",
                    help="专家数据；只统计 split=train 的 episode 作为训练样本数")
    ap.add_argument("--train-data", default="", help="覆盖：直接指定训练集 pkl")
    ap.add_argument("--out", default="reports/mission_breakdown.csv")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="只统计这些 run（默认全部）")
    args = ap.parse_args()

    payload = json.loads(Path(args.eval).read_text())
    runs = [r for r in payload["runs"] if r.get("kind", "bc") == "bc"]  # 排除平凡策略
    if args.runs:
        runs = [r for r in runs if r["run"] in args.runs]

    if args.train_data:
        train_eps = pickle.load(open(args.train_data, "rb"))
        train_src = args.train_data
    else:
        all_eps = pickle.load(open(args.data, "rb"))
        train_eps = [e for e in all_eps if e.get("split") == "train"]
        train_src = args.data + " (split=train)"
    n_train = {}
    for e in train_eps:
        n_train[e["mission"]] = n_train.get(e["mission"], 0) + 1

    # 汇总所有 run 的 per_mission
    stats: dict[str, dict] = {}
    for run in runs:
        for m, s in run["per_mission"].items():
            d = stats.setdefault(m, {"n": 0, "success": []})
            d["n"] += s["n"]
            d["success"].append(s["success"])

    rows = []
    for mission in sorted(stats):
        info = parse_mission(mission)
        succ = stats[mission]["success"]
        mean = sum(succ) / len(succ)
        sd = statistics.pstdev(succ) if len(succ) > 1 else 0.0
        ntr = n_train.get(mission, 0)
        rows.append(
            {
                "mission": mission,
                "color": info["color"],
                "type": info["obj_type"],
                "article": info["article"],
                "n_train_episodes": ntr,
                "n_test_episodes": stats[mission]["n"],
                "success_rate": round(mean, 4),
                "success_rate_sd": round(sd, 4),
                "low_coverage": int(ntr < LOW),
                "verb": "go to",
            }
        )

    # 训练集里存在但测试集没抽到的 mission 也要出现（success_rate 留空）
    for mission, ntr in sorted(n_train.items()):
        if mission in stats:
            continue
        info = parse_mission(mission)
        rows.append(
            {
                "mission": mission,
                "color": info["color"],
                "type": info["obj_type"],
                "article": info["article"],
                "n_train_episodes": ntr,
                "n_test_episodes": 0,
                "success_rate": "",
                "success_rate_sd": "",
                "low_coverage": int(ntr < LOW),
                "verb": "go to",
            }
        )

    rows.sort(key=lambda r: (r["color"], r["type"], r["article"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------- 汇总 md ----------------
    tested = [r for r in rows if r["n_test_episodes"] > 0]
    overall = sum(r["success_rate"] * r["n_test_episodes"] for r in tested) / max(
        1, sum(r["n_test_episodes"] for r in tested)
    )
    by_article: dict[str, list] = {}
    for r in tested:
        by_article.setdefault(r["article"], []).append(r)
    by_type: dict[str, list] = {}
    for r in tested:
        by_type.setdefault(r["type"], []).append(r)
    low = [r for r in tested if r["low_coverage"]]

    def wavg(rs):
        tot = sum(x["n_test_episodes"] for x in rs)
        return sum(x["success_rate"] * x["n_test_episodes"] for x in rs) / max(1, tot), tot

    L = ["# P2-1 · mission 类型分解\n"]
    L.append(f"> 数据源：`{args.eval}`（{len(runs)} 个 run，测试 seed {payload['seeds']}）")
    L.append(f"> 训练样本数来自 `{train_src}`\n")
    L.append(f"**加权总成功率：{overall:.3f}**（覆盖 {len(tested)}/36 种 mission，"
             f"共 {sum(r['n_test_episodes'] for r in tested)} 个测试 episode）\n")
    L.append("## 按冠词分解\n")
    L.append("| 冠词 | 测试 episode | 成功率 | 说明 |")
    L.append("|---|---|---|---|")
    for art in ("the", "a"):
        if art in by_article:
            s, n = wavg(by_article[art])
            L.append(
                f"| `{art}` | {n} | **{s:.3f}** | "
                + ("场景只有 1 个匹配物体" if art == "the" else "场景有多个匹配物体，任一个都算成功")
                + " |"
            )
    L.append("")
    L.append("## 按物体类型分解\n")
    L.append("| 类型 | 测试 episode | 成功率 |")
    L.append("|---|---|---|")
    for t in TYPES:
        if t in by_type:
            s, n = wavg(by_type[t])
            L.append(f"| `{t}` | {n} | **{s:.3f}** |")
    L.append("")
    L.append("## 按颜色分解\n")
    by_color: dict[str, list] = {}
    for r in tested:
        by_color.setdefault(r["color"], []).append(r)
    L.append("| 颜色 | 测试 episode | 成功率 |")
    L.append("|---|---|---|")
    for c in COLORS:
        if c in by_color:
            s, n = wavg(by_color[c])
            L.append(f"| `{c}` | {n} | **{s:.3f}** |")
    L.append("")
    L.append(f"## 低覆盖 mission（训练样本 < {LOW}，成功率不可靠）\n")
    if low:
        L.append("| mission | 训练样本 | 测试 episode | 成功率 |")
        L.append("|---|---|---|---|")
        for r in sorted(low, key=lambda x: x["n_train_episodes"]):
            L.append(
                f"| `{r['mission']}` | {r['n_train_episodes']} | {r['n_test_episodes']} | "
                f"{r['success_rate']:.3f} |"
            )
        L.append("")
        L.append(
            "这些 mission 在测试集里只有个位数 episode，单个 episode 的成败就能让成功率跳动 "
            "±20 个百分点以上，**不能据此下任何结论**。报告中引用时应标注为「样本不足，仅供参考」。\n"
        )
    else:
        L.append(f"无（全部 36 种 mission 的训练样本都 ≥{LOW}）。\n")

    best = sorted(tested, key=lambda r: -r["success_rate"])[:5]
    worst = sorted(tested, key=lambda r: r["success_rate"])[:5]
    L.append("## 成功率最高 / 最低的 mission（仅列测试 episode ≥ 5 的）\n")
    L.append("| | mission | n_train | n_test | success |")
    L.append("|---|---|---|---|---|")
    for tag, group in (("最高", best), ("最低", worst)):
        for r in group:
            if r["n_test_episodes"] < 5:
                continue
            L.append(
                f"| {tag} | `{r['mission']}` | {r['n_train_episodes']} | "
                f"{r['n_test_episodes']} | {r['success_rate']:.3f} |"
            )
    L.append("")

    md = out.with_suffix(".md")
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out} and {md}")
    print(f"missions tested: {len(tested)}, weighted success: {overall:.3f}, low_coverage: {len(low)}")


if __name__ == "__main__":
    main()
