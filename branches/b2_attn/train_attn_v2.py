"""支线 2 扩展消融 v2 · 训练脚本（与 `train_attn.py`、`train_film.py` 及 `b5` 步数对齐完全同口径）。

用法:
    OMP_NUM_THREADS=1 env_embodied/bin/python branches/b2_attn/train_attn_v2.py \
        --variant res_spatial_softmax --seed 0
    OMP_NUM_THREADS=1 env_embodied/bin/python branches/b2_attn/train_attn_v2.py \
        --variant res_spatial_softmax --frac 0.2 --match-steps --seed 0
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b2_attn"))
sys.path.insert(0, str(ROOT / "branches" / "b5_data_eff"))
sys.path.insert(0, str(ROOT / "src"))

from attn_v2_net import ATTN_V2_VARIANTS, build, n_params  # noqa: E402
from common import (  # noqa: E402
    ENV_ID,
    MAX_TOKENS,
    PAD_IDX,
    VOCAB,
    assert_no_leak,
    set_all_seeds,
)
from data_eff_utils import N_REF_TRANSITIONS, frac_tag, load_subset_episodes  # noqa: E402
from train_bc import build_dataset, make_ds  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


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
    ap.add_argument("--variant", required=True, choices=list(ATTN_V2_VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--obj-emb-dim", type=int, default=8)
    ap.add_argument("--lang-dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--match-steps", action="store_true")
    ap.add_argument("--data", default="data/expert_5000.pkl")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ftag = frac_tag(args.frac)
    if not args.out:
        suffix = f"_{ftag}" if args.frac < 1.0 else ""
        args.out = f"runs/b2_attn_v2_{args.variant}{suffix}_seed{args.seed}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    train_eps, val_eps = load_subset_episodes(args.data, args.frac, args.seed)
    assert_no_leak(e["seed"] for e in train_eps + val_eps)
    tr = build_dataset(train_eps)
    va = build_dataset(val_eps)
    n_tr = int(len(tr[3]))

    train_ds = make_ds(*tr)
    if args.match_steps and n_tr != N_REF_TRANSITIONS:
        sampler = RandomSampler(
            train_ds,
            replacement=(n_tr < N_REF_TRANSITIONS),
            num_samples=N_REF_TRANSITIONS,
            generator=g,
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0
        )
    val_loader = DataLoader(make_ds(*va), batch_size=512, shuffle=False, num_workers=0)

    model = build(
        args.variant,
        vocab_size=len(VOCAB),
        obj_emb_dim=args.obj_emb_dim,
        lang_dim=args.lang_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    )
    n_par = n_params(model)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict] = []
    best_acc, best_epoch, best_state, best_stats = -1.0, -1, None, {}
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

        val_loss, val_acc, _p, _l = evaluate(model, val_loader, crit)
        stats = model.scalar_stats()
        rec = {
            "epoch": epoch,
            "train_loss": round(run_loss / run_n, 6),
            "train_acc": round(run_c / run_n, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            **stats,
        }
        history.append(rec)
        if val_acc > best_acc + 1e-9:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_stats = stats
        print(
            f"[b2_attn_v2 {args.variant} {ftag} s{args.seed}] ep {epoch:2d} "
            f"loss={run_loss / run_n:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    assert best_state is not None
    model.load_state_dict(best_state)
    val_loss, val_acc, pred, lab = evaluate(model, val_loader, crit)
    torch.save(model.state_dict(), out / "best_model.pt")
    np.savez(out / "predictions_val.npz", pred=pred, label=lab, image=va[0], tokens=va[1], direction=va[2])
    (out / "vocab.json").write_text(
        json.dumps({"vocab": VOCAB, "pad_idx": PAD_IDX, "max_tokens": MAX_TOKENS}, ensure_ascii=False)
    )

    cfg = {
        "branch": "b2_attn_v2",
        "variant": args.variant,
        "run": str(out),
        "git_commit": git_commit(),
        "env_id": ENV_ID,
        "seed": args.seed,
        "frac": args.frac,
        "match_steps": bool(args.match_steps),
        "arch": {
            "variant": args.variant,
            "obj_emb_dim": args.obj_emb_dim,
            "lang_dim": args.lang_dim,
            "hidden": model.hidden,
            "dropout": args.dropout,
            "n_params": n_par,
        },
        "optim": {
            "optimizer": "Adam",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stop": "none（跑满 30 epoch 后事后选 val_acc 最优）",
        },
        "data": {
            "path": args.data,
            "n_train_episodes": len(train_eps),
            "n_train_transitions": n_tr,
            "n_val_episodes": len(val_eps),
        },
        "result": {
            "best_val_acc": round(best_acc, 6),
            "best_epoch": best_epoch,
            "train_seconds": round(time.time() - t0, 2),
            "mod_stats_at_best": best_stats,
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"{out}: best val_acc={best_acc:.4f} @ ep{best_epoch} ({n_par} params, {cfg['result']['train_seconds']}s)")


if __name__ == "__main__":
    main()
