"""支线 1 · 训练（复用冻结的数据管线，只替换语言编码器，含 FiLM+GRU 组合对照）。

用法:
    OMP_NUM_THREADS=1 python branches/b1_langenc/train_langenc.py --variant dan --seed 0 \
        --out runs/b1_dan_seed0
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
sys.path.insert(0, str(ROOT / "branches" / "b1_langenc"))
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    ENV_ID,
    MAX_TOKENS,
    PAD_IDX,
    VOCAB,
    assert_no_leak,
    set_all_seeds,
)
from lang_encoders import (  # noqa: E402
    FILM_VARIANTS,
    VARIANTS,
    build_lang_model,
    lang_params,
    n_params,
)
from train_bc import build_dataset, make_ds  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_data(path: str):
    eps = pickle.load(open(path, "rb"))
    train_eps = [e for e in eps if e.get("split") == "train"]
    val_eps = [e for e in eps if e.get("split") == "val"]
    assert_no_leak(e["seed"] for e in train_eps + val_eps)
    return train_eps, val_eps


@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    total, correct, n = 0.0, 0, 0
    preds, labels = [], []
    for img, tok, d, act in loader:
        logits = model(img, tok, d)
        total += crit(logits, act).item() * act.size(0)
        pred = logits.argmax(-1)
        correct += (pred == act).sum().item()
        n += act.size(0)
        preds.append(pred.numpy())
        labels.append(act.numpy())
    return total / max(1, n), correct / max(1, n), np.concatenate(preds), np.concatenate(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="mean", choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="默认 0（支线 2 实测 wd=1e-4 让基线掉 0.129）")
    ap.add_argument("--film-l2", type=float, default=1e-3,
                    help="仅对 film_* 变体生效：FiLM (Δγ, Δβ) 偏离正则权重")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lang-dim", type=int, default=32)
    ap.add_argument("--obj-emb-dim", type=int, default=8)
    ap.add_argument("--data", default="data/expert_5000.pkl")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if not args.out:
        args.out = f"runs/b1_{args.variant}_seed{args.seed}"

    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    train_eps, val_eps = load_data(args.data)
    tr = build_dataset(train_eps)
    va = build_dataset(val_eps)
    train_loader = DataLoader(make_ds(*tr), batch_size=args.batch_size,
                              shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(make_ds(*va), batch_size=512, shuffle=False, num_workers=0)

    model = build_lang_model(
        variant=args.variant,
        vocab_size=len(VOCAB),
        obj_emb_dim=args.obj_emb_dim,
        lang_dim=args.lang_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    )
    use_film = args.variant in FILM_VARIANTS
    film_l2 = args.film_l2 if use_film else 0.0
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict] = []
    best_acc, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_n, run_c = 0.0, 0, 0
        for img, tok, d, act in train_loader:
            logits = model(img, tok, d)
            loss = crit(logits, act)
            if use_film and film_l2 > 0.0 and hasattr(model, "film_penalty"):
                loss = loss + film_l2 * model.film_penalty(tok)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * act.size(0)
            run_n += act.size(0)
            run_c += (logits.argmax(-1) == act).sum().item()

        val_loss, val_acc, _p, _l = evaluate(model, val_loader, crit)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(run_loss / run_n, 6),
                "train_acc": round(run_c / run_n, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
        )
        improved = val_acc > best_acc + 1e-9
        if improved:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(
            f"[b1 {args.variant} s{args.seed}] ep {epoch:2d} "
            f"loss={run_loss / run_n:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            + ("  *" if improved else "")
        )

    model.load_state_dict(best_state)
    val_loss, val_acc, pred, lab = evaluate(model, val_loader, crit)
    torch.save(model.state_dict(), out / "best_model.pt")
    np.savez(out / "predictions_val.npz", pred=pred, label=lab,
             image=va[0], tokens=va[1], direction=va[2])
    (out / "vocab.json").write_text(
        json.dumps({"vocab": VOCAB, "pad_idx": PAD_IDX, "max_tokens": MAX_TOKENS}, ensure_ascii=False)
    )

    result_block: dict = {
        "best_val_acc": round(best_acc, 6),
        "best_epoch": best_epoch,
        "train_seconds": round(time.time() - t0, 2),
    }
    if hasattr(model, "film_stats"):
        result_block["film_stats_at_best"] = model.film_stats(torch.from_numpy(va[1]).long())

    cfg = {
        "branch": "b1_langenc",
        "variant": args.variant,
        "run": str(out),
        "git_commit": git_commit(),
        "env_id": ENV_ID,
        "seed": args.seed,
        "arch": {
            "variant": args.variant,
            "obj_emb_dim": args.obj_emb_dim,
            "lang_dim": args.lang_dim,
            "hidden": model.hidden,
            "dropout": args.dropout,
            "n_params": n_params(model),
            "n_lang_params": lang_params(model),
            "use_film": use_film,
            "fusion": (
                "film+concat（conv1 后零初始化线性 FiLM + 拼接）"
                if use_film
                else "concat（与基线相同，唯一变量是语言编码器）"
            ),
        },
        "optim": {
            "optimizer": "Adam",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "film_l2": film_l2,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stop": "none（跑满 30 epoch，事后选 val_acc 最优）",
        },
        "data": {
            "path": args.data,
            "n_train_episodes": len(train_eps),
            "n_train_transitions": int(len(tr[3])),
            "n_val_episodes": len(val_eps),
        },
        "result": result_block,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "history.json").write_text(json.dumps(history, indent=2))

    print(
        f"{out}: best val_acc={best_acc:.4f} @ ep{best_epoch} "
        f"({n_params(model)} params, lang={lang_params(model)}, "
        f"{cfg['result']['train_seconds']}s)"
    )


if __name__ == "__main__":
    main()
