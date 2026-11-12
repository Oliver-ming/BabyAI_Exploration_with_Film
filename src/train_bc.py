"""P1-1 · BC 基线训练（§6）。

用法:
    python src/train_bc.py --seed 0 --epochs 30 --out runs/baseline_seed0
    python src/train_bc.py --seed 0 --data-frac 0.2 --out runs/defrac0.2_seed0

要点（§6.2）：
  - 图像 (7,7,3) uint8，显式 permute(2,0,1)；默认用 nn.Embedding 编码对象 id（不是 /255 的连续值）
  - mission：固定词表 14 词 + 32 维平均词嵌入，只对非 pad 词求均值
  - direction：one-hot(4)，全部必需
  - 早停**只看验证集动作准确率**（patience 5，最大 30 epoch），绝不看闭环 Success
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ENV_ID,
    MAX_TOKENS,
    PAD_IDX,
    VOCAB,
    BCNet,
    mission_to_ids,
    assert_no_leak,
    set_all_seeds,
    split_of,
)

ACTION_NAMES = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]


def build_dataset(episodes, blank_language: bool = False):
    """返回 (imgs, toks, dirs, acts) 四个 numpy 数组。"""
    imgs, toks, dirs, acts = [], [], [], []
    pad_ids = [PAD_IDX] * MAX_TOKENS
    for ep in episodes:
        if blank_language:
            ids = pad_ids  # 消融：抹掉 mission，模型只能看patch+朝向
        else:
            ids = mission_to_ids(ep["mission"])
            ids = (ids + [PAD_IDX] * MAX_TOKENS)[:MAX_TOKENS]
        for img, d, a in zip(ep["images"], ep["directions"], ep["actions"]):
            imgs.append(img)
            toks.append(ids)
            dirs.append(d)
            acts.append(a)
    return (
        np.asarray(imgs, dtype=np.uint8),
        np.asarray(toks, dtype=np.int64),
        np.asarray(dirs, dtype=np.int64),
        np.asarray(acts, dtype=np.int64),
    )


def make_ds(*arrays):
    return TensorDataset(*[torch.from_numpy(a) for a in arrays])


@torch.no_grad()
def eval_model(model, loader, crit) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_pred, all_lab = [], []
    for img, tok, d, act in loader:
        logits = model(img, tok, d)
        total_loss += crit(logits, act).item() * act.size(0)
        pred = logits.argmax(-1)
        correct += (pred == act).sum().item()
        n += act.size(0)
        all_pred.append(pred.numpy())
        all_lab.append(act.numpy())
    return total_loss / max(1, n), correct / max(1, n), np.concatenate(all_pred), np.concatenate(all_lab)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lang-dim", type=int, default=32)
    ap.add_argument("--obj-emb-dim", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--img-mode", choices=["embed", "div255"], default="embed")
    ap.add_argument("--no-language", action="store_true",
                    help="消融：把 mission 全部置为 <pad>（模型只能靠视觉+朝向决策），"
                         "用于量化语言接地的贡献")
    ap.add_argument("--data-frac", type=float, default=1.0, help="使用多少比例的专家 episode")
    ap.add_argument("--frac-scheme", choices=["prefix", "random"], default="random",
                    help="取子集方式：prefix=按 seed 顺序前 N 条；random=按 seed 随机抽 N 条")
    ap.add_argument("--data", default="data/expert_5000.pkl",
                    help="专家数据（按 episode 的 split 字段分成 train/val）")
    ap.add_argument("--train-data", default="", help="覆盖：直接指定训练集 pkl")
    ap.add_argument("--val-data", default="", help="覆盖：直接指定验证集 pkl")
    ap.add_argument("--out", default="runs/baseline_seed0")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch CPU 线程数。默认 1 是为了**可复现**：多线程会改变浮点归约顺序，"
                         "使同一 seed 的 val_acc 出现 ~0.01 量级的抖动（实测 0.8807 / 0.8888 / 0.8899）。"
                         "本任务数据量极小，单线程 30 epoch 约需 55 秒，多进程并行跑 seed 更划算")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)  # 双保险：set_num_threads 之后再设一次

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    if args.train_data:
        train_eps = pickle.load(open(args.train_data, "rb"))
        val_eps = pickle.load(open(args.val_data, "rb"))
        data_desc = {"train_path": args.train_data, "val_path": args.val_data}
    else:
        all_eps = pickle.load(open(args.data, "rb"))
        train_eps = [e for e in all_eps if e.get("split", split_of(e["seed"])) == "train"]
        val_eps = [e for e in all_eps if e.get("split", split_of(e["seed"])) == "val"]
        data_desc = {"train_path": args.data, "val_path": args.data + " (split=val)"}

    # 防泄漏硬断言：任何训练/验证数据都不得含测试 seed
    assert_no_leak(e["seed"] for e in train_eps)
    assert_no_leak(e["seed"] for e in val_eps)

    # 数据效率消融：按 episode 取子集（不是按转移取，避免破坏 episode 结构）
    if args.data_frac < 1.0:
        n_keep = max(1, int(round(len(train_eps) * args.data_frac)))
        if args.frac_scheme == "prefix":
            train_eps = train_eps[:n_keep]
        else:
            rng = random.Random(args.seed)
            train_eps = rng.sample(train_eps, n_keep)

    tr = build_dataset(train_eps, args.no_language)
    va = build_dataset(val_eps, args.no_language)
    train_loader = DataLoader(
        make_ds(*tr), batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0
    )
    val_loader = DataLoader(
        make_ds(*va), batch_size=512, shuffle=False, num_workers=0
    )

    model = BCNet(
        vocab_size=len(VOCAB),
        img_mode=args.img_mode,
        obj_emb_dim=args.obj_emb_dim,
        lang_dim=args.lang_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 平凡基线：val 上永远猜 forward / 永远猜 left / 永远猜训练集多数类
    va_labels = va[3]
    tr_labels = tr[3]
    trivial = {
        "always_forward": float((va_labels == 2).mean()),
        "always_left": float((va_labels == 0).mean()),
        "train_majority_class": float((va_labels == int(np.bincount(tr_labels).argmax())).mean()),
    }

    history: list[dict] = []
    best_acc, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_n, run_correct = 0.0, 0, 0
        for img, tok, d, act in train_loader:
            logits = model(img, tok, d)
            loss = crit(logits, act)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * act.size(0)
            run_n += act.size(0)
            run_correct += (logits.argmax(-1) == act).sum().item()

        train_loss = run_loss / run_n
        train_acc = run_correct / run_n
        val_loss, val_acc, pred, lab = eval_model(model, val_loader, crit)

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
        }
        history.append(rec)
        improved = val_acc > best_acc + 1e-9
        if improved:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(
            f"[seed {args.seed}] epoch {epoch:2d} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            + ("  *" if improved else "")
        )
        if epoch - best_epoch >= args.patience:
            print(f"early stop: 验证准确率 {args.patience} 个 epoch 未提升（best={best_acc:.4f} @ {best_epoch}）")
            break

    elapsed = time.time() - t0

    # 用最佳 checkpoint 重算验证集逐样本预测（predictions_val.npz 需要它）
    model.load_state_dict(best_state)
    val_loss, val_acc, pred, lab = eval_model(model, val_loader, crit)
    torch.save(model.state_dict(), out / "best_model.pt")

    np.savez(
        out / "predictions_val.npz",
        pred=pred,
        label=lab,
        image=va[0],
        tokens=va[1],
        direction=va[2],
    )

    cfg = {
        "run": str(out),
        "git_commit": git_commit(),
        "env_id": ENV_ID,
        "seed": args.seed,
        "arch": {
            "img_mode": args.img_mode,
            "obj_emb_dim": args.obj_emb_dim,
            "lang_dim": args.lang_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "n_params": n_params,
            "no_language": args.no_language,
        },
        "optim": {
            "optimizer": "Adam",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "early_stop_patience": args.patience,
            "early_stop_metric": "val_action_accuracy",
        },
        "data": {
            **data_desc,
            "data_frac": args.data_frac,
            "frac_scheme": args.frac_scheme,
            "n_train_episodes": len(train_eps),
            "n_train_transitions": int(len(tr[3])),
            "n_val_episodes": len(val_eps),
            "n_val_transitions": int(len(va[3])),
            "train_seeds": {
                "n": len(train_eps),
                "segments": "0..4000 ∪ 4800..5300（见 src/common.py）",
                "first": int(train_eps[0]["seed"]),
                "last": int(train_eps[-1]["seed"]),
            },
        },
        "result": {
            "best_val_acc": round(best_acc, 6),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "train_seconds": round(elapsed, 2),
            "trivial_baselines": {k: round(v, 6) for k, v in trivial.items()},
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "vocab.json").write_text(
        json.dumps({"vocab": VOCAB, "pad_idx": PAD_IDX, "max_tokens": MAX_TOKENS},
                   indent=2, ensure_ascii=False)
    )
    (out / "history.json").write_text(json.dumps(history, indent=2))

    print(
        f"{out}: best val_acc={best_acc:.4f} @ epoch {best_epoch} "
        f"({len(history)} epochs, {elapsed:.1f}s, {n_params} params)"
    )
    print(f"  平凡基线（val acc）: {trivial}")


if __name__ == "__main__":
    main()
