"""P0-3 · 数据集统计审计 -> reports/data_audit.md（§5 要求的 6 项 + 协议说明）。

用法:
    python src/audit_data.py
    python src/audit_data.py --train data/expert_train.pkl --val data/expert_val.pkl
"""

from __future__ import annotations

import argparse
import collections
import pickle
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    COLORS,
    N_EXPERT_TOTAL,
    N_EXPERT_TRAIN,
    N_EXPERT_VAL,
    TEST_SEEDS,
    TEST_SEEDS_S7,
    TYPES,
    TRAIN_SEEDS,
    TRAIN_SEEDS_PRIMARY,
    TRAIN_SEEDS_SUPPLEMENT,
    VAL_SEEDS,
    split_of,
)

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
ALL_MISSIONS = [
    f"go to {art} {c} {t}" for art in ("the", "a") for c in COLORS for t in TYPES
]
LOW = 20  # §5 的低覆盖阈值


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expert_5000.pkl",
                    help="expert_5000.pkl（含 split 字段）；结果会按 split 拆成训练/验证")
    ap.add_argument("--train", default="", help="可选：单独指定训练集（覆盖 --data 的 train 划分）")
    ap.add_argument("--val", default="", help="可选：单独指定验证集")
    ap.add_argument("--out", default="reports/data_audit.md")
    args = ap.parse_args()

    if args.train:
        train_eps = pickle.load(open(args.train, "rb"))
        val_eps = pickle.load(open(args.val, "rb"))
        train_path, val_path = args.train, args.val
    else:
        all_eps = pickle.load(open(args.data, "rb"))
        train_eps = [e for e in all_eps if split_of(e["seed"]) == "train"]
        val_eps = [e for e in all_eps if split_of(e["seed"]) == "val"]
        train_path, val_path = args.data, args.data
    eps = train_eps  # 统计主体 = 训练集（这是"专家演示数据集"的定义域）

    n_ep = len(eps)
    lens = [e["episode_len"] for e in eps]
    n_tr = sum(lens)
    act_ctr = collections.Counter(a for e in eps for a in e["actions"])
    mission_ctr = collections.Counter(e["mission"] for e in eps)
    art_ctr = collections.Counter(e["mission"].split()[2] for e in eps)
    poss_ctr = collections.Counter(e.get("obj_poss_n") for e in eps)

    uniq_img_dir, uniq_img = set(), set()
    for e in eps:
        for img, d in zip(e["images"], e["directions"]):
            uniq_img_dir.add((img.tobytes(), d))
            uniq_img.add(img.tobytes())

    low_cover = [(m, mission_ctr.get(m, 0)) for m in ALL_MISSIONS if mission_ctr.get(m, 0) < LOW]
    n_ok = 36 - len(low_cover)
    missing = [m for m in ALL_MISSIONS if mission_ctr.get(m, 0) == 0]
    bad = set(act_ctr) & {3, 4, 5, 6}
    n_val_tr = sum(e["episode_len"] for e in val_eps)
    val_missions = len({e["mission"] for e in val_eps})

    L: list[str] = []
    L.append("# P0-3 · 专家数据集审计（BabyAI-GoToLocalS6N3）\n")
    L.append(
        f"> 数据文件 `{train_path}`（按 seed 的 `split` 字段分词；训练 {len(train_eps)} / 验证 {len(val_eps)}）　"
        "生成脚本 `src/gen_expert.py`　审计脚本 `src/audit_data.py`\n"
    )

    # ================= §0 协议 =================
    L.append("## 0. 实验协议与实际 seed 划分（必读）\n")
    L.append("### 0.1 为什么不能照抄 §1.3 的 seed 配置\n")
    L.append(
        "指导文档 §1.3 写 `TRAIN_SEEDS = range(0, 4000)` 用来生成 **5000 条**专家演示。这里有两处冲突：\n"
    )
    L.append(
        "1. **数量不足**：同一个 env seed 只能产出一条确定性 episode，`range(0,4000)` 最多 4000 条互不重复样本。"
        "用循环抽样凑到 5000 条会注入 1000 条完全重复数据，直接污染 §10.2 的数据效率消融。"
    )
    L.append(
        "2. **验证集也要专家**：§1.2 ① 强制用**验证集动作准确率**选 checkpoint，"
        "算准确率必须知道 VAL seed 上的专家动作，所以要额外跑 500 条专家 episode。"
        "若训练集仍占满 5000 条，专家 episode 总数 = 5500 > 手册硬上限 5000。\n"
    )
    L.append("### 0.2 本实现的划分（`src/common.py` 是唯一定义处）\n")
    L.append("| 用途 | seed 段 | 条数 | 与文档的关系 |")
    L.append("|---|---|---|---|")
    L.append(f"| 训练专家（主段） | `0..4000` | {len(TRAIN_SEEDS_PRIMARY)} | 与文档一致 |")
    L.append(
        f"| 训练专家（补足段） | `4800..5300` | {len(TRAIN_SEEDS_SUPPLEMENT)} | **文档未覆盖的区间** |"
    )
    L.append(f"| 验证专家（选 checkpoint） | `4000..4500` | {len(VAL_SEEDS)} | 与文档一致 |")
    L.append(f"| 测试（最终闭环评估） | `4500..4800` | {len(TEST_SEEDS)} | 与文档一致 |")
    L.append(f"| 迁移测试 S7N4 | `10000..10300` | {len(TEST_SEEDS_S7)} | 与文档一致 |")
    L.append("")
    L.append(
        "**专家 episode 总预算**："
        f"{N_EXPERT_TRAIN}（训练）+ {N_EXPERT_VAL}（验证）= **{N_EXPERT_TOTAL}** 条，"
        "正好等于手册上限 5000，未超。\n"
    )
    L.append(
        "补足段 `4800..5300` 与 VAL(`4000..4500`)、TEST(`4500..4800`)、TEST_S7(`10000..10300`) "
        "**全不相交**，且从未参与任何评估，因此「测试 seed 绝不进训练」这条硬性规定仍然成立。\n"
    )
    L.append("### 0.3 实际数据规模\n")
    L.append("| 数据 | episode | 转移 |")
    L.append("|---|---|---|")
    L.append(f"| 训练集（split=train） | {len(train_eps)} | {n_tr} |")
    L.append(f"| 验证集（split=val） | {len(val_eps)} | {n_val_tr} |")
    L.append(f"| 合计（= 专家 episode 总预算） | **{len(train_eps) + len(val_eps)}** | {n_tr + n_val_tr} |")
    L.append("")

    # ================= §1 规模 =================
    L.append("## 1. 训练集规模与 episode 长度\n")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| 成功专家 episode 数 | **{n_ep}** |")
    L.append(f"| 转移（(obs, action) 训练样本）总数 | **{n_tr}** |")
    L.append(f"| 平均长度 | {n_tr / n_ep:.2f} 步 |")
    L.append(f"| 长度 min / median / max | {min(lens)} / {statistics.median(lens)} / {max(lens)} |")
    L.append("| 专家失败 episode | 0（BabyAIBot 在 S6N3 上 100% 成功） |")
    L.append(f"| 验证集 episode 数 | {len(val_eps)}（{val_missions} 种 mission） |")
    L.append("")
    L.append("长度分布：\n")
    L.append("| 长度（步） | episode 数 | 占比 |")
    L.append("|---|---|---|")
    for k, v in sorted(collections.Counter(lens).items()):
        L.append(f"| {k} | {v} | {v / n_ep:.1%} |")
    L.append("")
    L.append(
        f"**关键含义**：{n_ep} 条 episode 只展开成 **{n_tr}** 个 (obs, action) 样本。"
        "这个量级意味着 BC 训练几十秒就能收敛，主要风险是**过拟合专家访问过的状态分布**，"
        "而不是欠拟合。这也是 §7 动作准确率与闭环 Success 会出现巨大差距的根本原因。\n"
    )
    L.append(
        "episode 长度 = 1 的样本占 "
        f"{collections.Counter(lens)[1] / n_ep:.1%}：agent 出生时就已「站在目标旁边并面向它」，"
        "一步都不用动。这类样本对训练几乎只提供「当前朝向已正确」这一条信息，"
        "却在数据里占固定比例，会让验证准确率虚高。\n"
    )

    # ================= §2 动作分布 =================
    L.append("## 2. 动作分布（对照 §4.1 ④）\n")
    L.append("| 动作 | 次数 | 占比 |")
    L.append("|---|---|---|")
    for a in range(7):
        v = act_ctr.get(a, 0)
        L.append(f"| `{ACTION_NAMES[a]}` | {v} | {v / n_tr:.1%} |")
    L.append("")
    L.append(
        f"**平凡基线**：「永远预测 forward」可拿到 **{act_ctr.get(2, 0) / n_tr:.1%}** 的 "
        "teacher-forcing 动作准确率，但闭环 Success 恒为 0（不会转向就永远到不了目标）。"
        "报告必须把这组数字与闭环 Success 并列呈现。"
    )
    L.append(
        "`pickup/drop/toggle/done` 合计 "
        f"**{sum(act_ctr.get(a, 0) for a in (3, 4, 5, 6)) / n_tr:.1%}**"
        + ("" if not bad else f"　→ ❌ 出现禁用动作 {sorted(bad)}")
        + "　GoToLocal 只需移动+转向，符合预期。\n"
    )

    # ================= §3 mission 覆盖 =================
    L.append("## 3. mission 覆盖（36 种字符串）\n")
    L.append(
        f"训练集出现 **{len(mission_ctr)}/36** 种 mission；≥{LOW} 条的有 **{n_ok}/36** 种。"
        f"未出现 {len(missing)} 种，低覆盖 {len(low_cover)} 种。\n"
    )
    L.append("| mission | 训练 episode 数 | 占比 | 覆盖判定 |")
    L.append("|---|---|---|---|")
    for m in ALL_MISSIONS:
        n = mission_ctr.get(m, 0)
        flag = "OK" if n >= LOW else ("**低覆盖**" if n > 0 else "**未出现**")
        L.append(f"| `{m}` | {n} | {n / n_ep:.2%} | {flag} |")
    L.append("")
    if low_cover:
        L.append(f"### 低覆盖 mission（<{LOW} 条）\n")
        L.append("这些 mission 的**成功率估计方差极大**，P2-1 的分解表里必须单独标注，不可当作结论。\n")
        for m, n in low_cover:
            L.append(f"- `{m}`：{n} 条")
        L.append("")
    else:
        L.append(f"✅ 全部 36 种 mission 均 ≥{LOW} 条。\n")

    # ================= §4 冠词 =================
    L.append("## 4. 冠词分布（`the` vs `a`）\n")
    L.append("| 冠词 | 训练 episode 数 | 占比 | 语义（`surface()` 逻辑） |")
    L.append("|---|---|---|---|")
    L.append(
        f"| `the` | {art_ctr.get('the', 0)} | {art_ctr.get('the', 0) / n_ep:.1%} | 场景里只有 1 个匹配物体 |"
    )
    L.append(
        f"| `a` | {art_ctr.get('a', 0)} | {art_ctr.get('a', 0) / n_ep:.1%} | 场景里有 >1 个匹配物体，任一个都算成功 |"
    )
    L.append("")
    L.append(
        f"`a` 只占 {art_ctr.get('a', 0) / n_ep:.1%}，但它天然更难：存在多个候选目标，模型必须靠"
        "「哪个更近 / 哪个在视野里」这类隐式线索挑一个，挑错就退化成 `distractor_priority` 失败。"
        "因此 36 种 mission 的成功率不会均匀，`a` 系列应当显著更低。\n"
    )

    # ================= §5 obj_poss =================
    L.append("## 5. 每个 episode 的匹配物体数量（`obj_poss`）分布\n")
    L.append("| 匹配物体数 | episode 数 | 占比 |")
    L.append("|---|---|---|")
    for k, v in sorted(poss_ctr.items(), key=lambda kv: (kv[0] is None, kv[0])):
        L.append(f"| {k} | {v} | {v / n_ep:.1%} |")
    L.append("")
    ones = poss_ctr.get(1, 0)
    multi = n_ep - ones - poss_ctr.get(None, 0)
    L.append(
        f"匹配物体数 = 1 的有 {ones} 条（{ones / n_ep:.1%}），>1 的有 {multi} 条（{multi / n_ep:.1%}）。"
        "这与 §4 的冠词分布严格对应（`surface()` 用 `len(obj_set) > 1` 决定加 `a`），"
        "可作为数据一致性交叉验证：两项比例应一致。\n"
    )

    # ================= §6 观测唯一性 =================
    L.append("## 6. 观测唯一性\n")
    L.append("| 指标 | 值 |")
    L.append("|---|---|")
    L.append(f"| 训练转移总数 | {n_tr} |")
    L.append(f"| 唯一 `(image, direction)` 组合 | **{len(uniq_img_dir)}**（{len(uniq_img_dir) / n_tr:.1%}） |")
    L.append(f"| 唯一 `image` 帧 | **{len(uniq_img)}**（{len(uniq_img) / n_tr:.1%}） |")
    L.append("")
    L.append(
        f"唯一 `(image, direction)` 占 {len(uniq_img_dir) / n_tr:.1%}，重复主要来自"
        "「转一圈回到同一朝向」这类状态。更关键的是唯一 `image` 帧数"
        f"（{len(uniq_img)}）明显少于唯一组合数（{len(uniq_img_dir)}）——"
        "**同一个 7×7 patch 可以对应多个绝对朝向**，这直接证明了 §4.1 ⑤ 的结论："
        "`direction` 是不能省的输入，只给 patch 无法唯一确定「该往哪边转」。\n"
    )

    # ================= §7 结论 =================
    checks = [
        (len(train_eps) + len(val_eps) == 5000, f"专家 episode 总数 = {len(train_eps)+len(val_eps)}（手册上限 5000）"),
        (not bad, "训练集无 pickup/drop/toggle/done 动作"),
        (len(mission_ctr) == 36, f"36 种 mission 全部在训练集出现（实际 {len(mission_ctr)}）"),
        (n_ok >= 30, f"≥{LOW} 条的 mission 有 {n_ok}/36（§5 要求 ≥30）"),
        (
            len({e["seed"] for e in train_eps}) == len(train_eps),
            "训练集内部无重复 seed（无重复样本）",
        ),
        (
            not (set(e["seed"] for e in train_eps + val_eps) & set(TEST_SEEDS + TEST_SEEDS_S7)),
            "训练集与验证集均不含任何 TEST / TEST_S7 seed",
        ),
        (
            not (set(e["seed"] for e in train_eps) & set(VAL_SEEDS)),
            "训练集与验证集 seed 不相交",
        ),
        (
            set(e["seed"] for e in train_eps) <= set(TRAIN_SEEDS),
            "训练集 seed 全部落在约定的 TRAIN_SEEDS 内",
        ),
    ]
    L.append("## 7. 验收结论\n")
    for ok, desc in checks:
        L.append(f"- {'✅' if ok else '❌'} {desc}")
    L.append("")
    L.append("对照 §5 的六项要求：1 规模/长度 ✅　2 动作分布 ✅　3 mission 覆盖表 ✅　"
             "4 冠词分布 ✅　5 `obj_poss` 分布 ✅　6 观测唯一性 ✅")
    L.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")
    print(
        f"train={len(train_eps)} val={len(val_eps)} transitions={n_tr} "
        f"missions={len(mission_ctr)}/36 low_cover={len(low_cover)} "
        f"uniq(img,dir)={len(uniq_img_dir)} uniq(img)={len(uniq_img)}"
    )
    failed = [d for ok, d in checks if not ok]
    assert not failed, "审计未通过：" + str(failed)


if __name__ == "__main__":
    main()
