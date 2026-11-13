"""支线 3 · DAgger-lite 训练（在"专家 + DAgger 聚合"数据上从零重训）。

用法:
    python branches/b3_dagger/train_dagger.py --seed 0 --variant dagger \
        --dagger-data data/b3_dagger_500.pkl --out runs/b3_dagger_seed0

关键约束（LOG.md §2.2）：
  1. **聚合而非替换**：训练集 = 4500 专家 + 全部 DAgger 样本，每轮从零训练
  2. **验证集保持纯净**：验证集永远是那 500 条专家 episode，
     DAgger 样本绝不混入 —— 否则"用验证集动作准确率选 checkpoint"就失效了
  3. 选模只看验证集动作准确率；但**不早停**（跑满 30 epoch 后事后选最优），
     非此不能拿到真实上限 — 见 branches/b3_dagger/LOG.md §5.1
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "src"))

# 冻结模块：只读复用（不改一行）
from common import (  # noqa: E402
    ENV_ID,
    PAD_IDX,
    VOCAB,
    MAX_TOKENS,
    BCNet,
    assert_no_leak,
    mission_to_ids,
    set_all_seeds,
    split_of,
)
from train_bc import build_dataset, eval_model, make_ds  # noqa: E402


def load_official_data(path: str = "data/expert_5000.pkl"):
    eps = pickle.load(open(path, "rb"))
    train_eps = [e for e in eps if e.get("split") == "train"]
    val_eps = [e for e in eps if e.get("split") == "val"]
    assert_no_leak(e["seed"] for e in train_eps + val_eps)
    return train_eps, val_eps


def load_branch_episodes(path: str, dedup: str = "none", expert_eps: list[dict] | None = None):
    """读 DAgger / 额外专家采集结果，转成和专家数据同构的 episode dict。

    dedup 选项（针对 DAgger 数据的高度冗余问题）：
      - "none"    : 全量保留
      - "episode" : 每条 episode 内部，同一个 (image, direction) 只保留第一次出现
                    —— 策略卡在原地转圈时会连续多步访问同一个状态，这些重复样本
                    会把损失函数往"病理状态"上偏，必须去掉
      - "global"  : 在上面基础上，再丢掉专家训练集里已经出现过的状态
                    —— 只保留"真正新增"的状态覆盖
    """
    obj = pickle.load(open(path, "rb"))
    eps = obj["episodes"]
    meta = {k: v for k, v in obj["meta"].items() if k != "seeds"}

    seen_global: set = set()
    if dedup == "global" and expert_eps is not None:
        for e in expert_eps:
            for img, d in zip(e["images"], e["directions"]):
                seen_global.add((img.tobytes(), int(d)))

    out = []
    n_raw = n_kept = 0
    for e in eps:
        imgs, dirs, acts = [], [], []
        seen_local: set = set()
        for img, d, a in zip(e["images"], e["directions"], e["actions"]):
            n_raw += 1
            key = (img.tobytes(), int(d))
            if dedup in ("episode", "global") and key in seen_local:
                continue
            if dedup == "global" and key in seen_global:
                continue
            seen_local.add(key)
            imgs.append(img)
            dirs.append(d)
            acts.append(a)
            n_kept += 1
        if imgs:
            out.append(
                {
                    "seed": e["seed"],
                    "mission": e["mission"],
                    "images": imgs,
                    "directions": dirs,
                    "actions": acts,
                    "episode_len": len(imgs),
                }
            )

    meta["dedup"] = dedup
    meta["n_transitions_raw"] = n_raw
    meta["n_transitions_kept"] = n_kept
    meta["n_episodes_kept"] = len(out)
    return out, meta


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def onpolicy_accuracy(model, episodes, batch: int = 512) -> float:
    """在**策略自己访问到的状态**上，与专家标签的一致率。

    这是 DAgger 论文里真正对应的指标（`d_π` 上的损失），
    与"纯专家验证集准确率"是不同的分布。三者同时报告才不会误判。
    """
    if not episodes:
        return float("nan")
    tr = build_dataset(episodes)
    loader = DataLoader(make_ds(*tr), batch_size=batch, shuffle=False, num_workers=0)
    crit = nn.CrossEntropyLoss()
    _l, acc, _p, _t = eval_model(model, loader, crit)
    return float(acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variant", default="dagger", choices=["dagger", "bc_extra", "bc"])
    ap.add_argument("--dagger-data", default="",
                    help="dagger / bc_extra 组必填。支持逗号分隔多个文件（多轮 DAgger 聚合）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=0,
                    help="0 = 不早停，跑满 epochs 后事后选 val_acc 最优（默认，见训练循环注释）")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lang-dim", type=int, default=32)
    ap.add_argument("--obj-emb-dim", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--dedup", default="none", choices=["none", "episode", "global"],
                    help="DAgger 数据的去冗余强度（见 load_branch_episodes 的 docstring）")
    ap.add_argument("--official-data", default="data/expert_5000.pkl")
    ap.add_argument("--out", default="runs/b3_dagger_seed0")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    expert_train, expert_val = load_official_data(args.official_data)

    extra_eps, extra_meta = [], {}
    if args.variant in ("dagger", "bc_extra"):
        if not args.dagger_data:
            raise SystemExit(f"--variant {args.variant} 需要 --dagger-data")
        paths = [p for p in args.dagger_data.split(",") if p]
        extra_eps, extra_meta = [], {"sources": [], "dedup": args.dedup}
        seen_states: set = set()
        if args.dedup == "global":
            for e in expert_train:
                for img, d in zip(e["images"], e["directions"]):
                    seen_states.add((img.tobytes(), int(d)))
        for p in paths:
            eps_i, meta_i = load_branch_episodes(p, dedup=args.dedup, expert_eps=expert_train)
            # 跨文件去重：多轮 DAgger 里同一个状态极可能重复出现
            kept = []
            for e in eps_i:
                imgs, dirs, acts = [], [], []
                local: set = set()
                for img, d, a in zip(e["images"], e["directions"], e["actions"]):
                    k = (img.tobytes(), int(d))
                    if k in local or k in seen_states:
                        continue
                    local.add(k)
                    imgs.append(img); dirs.append(d); acts.append(a)
                if imgs:
                    kept.append(dict(e, images=imgs, directions=dirs, actions=acts, episode_len=len(imgs)))
            extra_eps.extend(kept)
            extra_meta["sources"].append({"path": p, "episodes": len(eps_i), "kept": len(kept), **meta_i})
        # 采集数据只允许来自 TRAIN_SEEDS（禁止测试/验证 seed 混入训练）
        assert_no_leak(e["seed"] for e in extra_eps)

    # ★ on-policy 状态集必须**留出**，否则测的是训练集准确率（会虚高）。
    # 按 seed 每 5 条留 1 条作 holdout，两组（dagger / bc_extra）用同一规则，保持可比。
    holdout_eps = [e for i, e in enumerate(sorted(extra_eps, key=lambda x: x["seed"])) if i % 5 == 0]
    train_extra = [e for i, e in enumerate(sorted(extra_eps, key=lambda x: x["seed"])) if i % 5 != 0]

    train_eps = expert_train + train_extra  # ★ 聚合，不是替换
    tr = build_dataset(train_eps)
    va = build_dataset(expert_val)  # ★ 验证集永远纯净

    train_loader = DataLoader(
        make_ds(*tr), batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0
    )
    val_loader = DataLoader(make_ds(*va), batch_size=512, shuffle=False, num_workers=0)

    model = BCNet(
        vocab_size=len(VOCAB),
        obj_emb_dim=args.obj_emb_dim,
        lang_dim=args.lang_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best_score, best_epoch, best_state = -float("inf"), -1, None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_n, run_c = 0.0, 0, 0
        for img, tok, d, act in train_loader:
            logits = model(img, tok, d)
            loss = crit(logits, act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * act.size(0)
            run_n += act.size(0)
            run_c += (logits.argmax(-1) == act).sum().item()

        val_loss, val_acc, _p, _l = eval_model(model, val_loader, crit)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(run_loss / run_n, 6),
                "train_acc": round(run_c / run_n, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
        )
        # ---------------------------------------------------------------
        # 选模判据：仍用 val_acc（手册硬规定「仅用验证集动作准确率」），
        # 但把「早停」与「选模」解耦：
        #   - 早停默认关闭（--patience 0），跑满 30 epoch，然后**事后**在所有
        #     epoch 里挑 val_acc 最高的那个 checkpoint。
        #   - 依据：arXiv 2602.22107 "Don't stop me now" 实测「post-hoc selection
        #     over all epochs」稳定优于「用 accuracy 做早停」；本支线也复现了这一点：
        #     patience=5 时 seed1/seed2 于 epoch 13/14 停住，val_acc 0.8167/0.8227，
        #     而 30-epoch 上限是 0.8720/0.8579。
        #   - 这不违反手册：选模依据仍是验证集动作准确率，只是不再提前掐断训练。
        # ---------------------------------------------------------------
        improved = val_acc > best_score + 1e-9
        if improved:
            best_score, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(
            f"[b3 {args.variant} seed {args.seed}] ep {epoch:2d} "
            f"train_loss={run_loss / run_n:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            + ("  *" if improved else "")
        )
        if args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"early stop @ {epoch}（best val_acc {best_score:.4f} @ {best_epoch}）")
            break

    model.load_state_dict(best_state)
    val_loss, val_acc, pred, lab = eval_model(model, val_loader, crit)

    # ★ 第三套指标：在策略自己访问到的状态上的动作准确率。
    # 用采集数据（DAgger 组）或额外专家数据（bc_extra 组）作为"on-policy 状态集"。
    onpolicy_acc = onpolicy_accuracy(model, holdout_eps) if holdout_eps else float("nan")

    torch.save(model.state_dict(), out / "best_model.pt")
    np.savez(out / "predictions_val.npz", pred=pred, label=lab)
    (out / "vocab.json").write_text(
        json.dumps({"vocab": VOCAB, "pad_idx": PAD_IDX, "max_tokens": MAX_TOKENS}, ensure_ascii=False)
    )

    cfg = {
        "branch": "b3_dagger",
        "variant": args.variant,
        "run": str(out),
        "git_commit": git_commit(),
        "env_id": ENV_ID,
        "seed": args.seed,
        "arch": {
            "img_mode": "embed",
            "obj_emb_dim": args.obj_emb_dim,
            "lang_dim": args.lang_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "n_params": n_params,
        },
        "optim": {
            "optimizer": "Adam",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "early_stop_patience": args.patience,
            "early_stop_metric": "val_action_accuracy",
        },
        "data": {
            "official_data": args.official_data,
            "n_expert_train": len(expert_train),
            "n_extra": len(extra_eps),
            "n_extra_train": len(train_extra),
            "n_extra_holdout": len(holdout_eps),
            "n_train_episodes": len(train_eps),
            "n_train_transitions": int(len(tr[3])),
            "n_val_episodes": len(expert_val),
            "n_val_transitions": int(len(va[3])),
            "dedup": args.dedup,
            "extra_meta": extra_meta,
        },
        "notes": {
            "val_metric_caveat": (
                "best_val_acc 用的是纯专家验证集（冻结口径，也是选 checkpoint 的唯一依据）。"
                "DAgger 有意把训练分布移向 d_π，因此这个数字下降不代表策略变差；"
                "判断是否有效的依据是 closed_loop_success 与 onpolicy_action_acc。"
            )
        },
        "result": {
            "best_val_acc": round(best_score, 6),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "train_seconds": round(time.time() - t0, 2),
            "onpolicy_action_acc": round(onpolicy_acc, 6) if onpolicy_acc == onpolicy_acc else None,
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "history.json").write_text(json.dumps(history, indent=2))

    print(
        f"{out}: best val_acc={best_score:.4f} @ {best_epoch} "
        f"| onpolicy_acc={onpolicy_acc:.4f} "
        f"({len(history)} ep, {cfg['result']['train_seconds']}s, "
        f"train={len(train_eps)} eps / {len(tr[3])} transitions)"
    )


if __name__ == "__main__":
    main()
