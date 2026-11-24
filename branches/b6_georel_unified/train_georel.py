"""Train the retained single-frame AB / ABC models on frozen expert demonstrations.

Uses the inherited BC30 protocol: 30 full epochs, max validation action accuracy checkpoint.
Disclosed deviation: early stopping is disabled for this controlled comparison.
Low-data runs optionally match sampled transitions / optimizer updates to full-data runs.
Usage: python branches/b6_georel_unified/train_georel.py --variant s3_map1h_relbias --seed 0
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import platform
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
sys.path.insert(0, str(ROOT / "branches" / "b6_georel_unified"))
sys.path.insert(0, str(ROOT / "branches" / "b5_data_eff"))
sys.path.insert(0, str(ROOT / "src"))

from common import (  # noqa: E402
    ENV_ID,
    PAD_IDX,
    VOCAB,
    assert_no_leak,
    set_all_seeds,
)
from data_eff_utils import N_REF_TRANSITIONS, frac_tag, load_subset_episodes  # noqa: E402
from georel_net import ALL_GEOREL_VARIANTS, build, n_params  # noqa: E402
from train_bc import build_dataset, make_ds  # noqa: E402



def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def evaluate_single_frame(model, loader, crit):
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
    ap.add_argument("--variant", required=True, choices=list(ALL_GEOREL_VARIANTS))
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
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not 1 <= args.epochs <= 30:
        ap.error("--epochs must be between 1 and 30")
    torch.set_num_threads(args.threads)
    ftag = frac_tag(args.frac)
    if not args.out:
        suffix = f"_{ftag}" if args.frac < 1.0 else ""
        args.out = f"runs/b6_georel_{args.variant}{suffix}_seed{args.seed}"
    out = Path(args.out)
    if (out / "best_model.pt").exists() and not args.overwrite:
        ap.error(f"Run already exists: {out}; use a fresh --out or explicit --overwrite")
    out.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    train_eps, val_eps = load_subset_episodes(args.data, args.frac, args.seed)
    assert_no_leak(e["seed"] for e in train_eps + val_eps)

    model = build(
        args.variant,
        vocab_size=len(VOCAB),
        obj_emb_dim=args.obj_emb_dim,
        lang_dim=args.lang_dim,
        hidden=args.hidden,
        dropout=args.dropout,
    )
    n_par = n_params(model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    crit = nn.CrossEntropyLoss()
    tr = build_dataset(train_eps)
    va = build_dataset(val_eps)
    n_tr_trans = int(len(tr[3]))
    n_va_trans = int(len(va[3]))
    train_ds = make_ds(*tr)
    if args.match_steps and n_tr_trans != N_REF_TRANSITIONS:
        sampler = RandomSampler(
            train_ds,
            replacement=(n_tr_trans < N_REF_TRANSITIONS),
            num_samples=N_REF_TRANSITIONS,
            generator=g,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0
        )
    val_loader = DataLoader(make_ds(*va), batch_size=512, shuffle=False, num_workers=0)

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
            run_c += (logits.argmax(-1) == act).sum().item()
            run_n += act.size(0)
        val_loss, val_acc, _, _ = evaluate_single_frame(model, val_loader, crit)

        tr_loss = run_loss / max(1, run_n)
        tr_acc = run_c / max(1, run_n)
        sstats = model.scalar_stats()
        rec = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 5),
            "train_acc": round(tr_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "elapsed_s": round(time.time() - t0, 2),
            "scalars": sstats,
        }
        history.append(rec)
        if epoch == 1 or epoch % 5 == 0:
            print(f"[epoch {epoch}/{args.epochs}] train_loss={tr_loss:.4f} val_acc={val_acc:.4f}", flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_stats = sstats

    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save(best_state, out / "best_model.pt")
    _, _, preds, labels = evaluate_single_frame(model, val_loader, crit)
    np.savez_compressed(out / "predictions_val.npz", preds=preds, labels=labels)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "vocab.json").write_text(json.dumps({"vocab": VOCAB, "pad_idx": PAD_IDX}, indent=2))

    source_files = sorted(list((ROOT / "src").rglob("*.py")) + list((ROOT / "branches").rglob("*.py")))
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    cfg = {
        "branch": "b6_georel_unified",
        "env_id": ENV_ID,
        "git_commit": git_commit(),
        "source_hashes": source_hashes,
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "runtime": {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "threads": args.threads},
        "seed": args.seed,
        "frac": args.frac,
        "match_steps": bool(args.match_steps),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "data": args.data,
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_train_transitions": n_tr_trans,
        "n_val_transitions": n_va_trans,
        "arch": {
            "variant": args.variant,
            "is_recurrent": False,
            "obj_emb_dim": args.obj_emb_dim,
            "lang_dim": args.lang_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "n_params": n_par,
        },
        "result": {
            "best_val_acc": round(best_acc, 6),
            "best_epoch": best_epoch,
            "train_seconds": round(time.time() - t0, 2),
            "best_scalars": best_stats,
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(
        f"[b6_georel] {args.variant:18s} frac={args.frac:4.2f} seed={args.seed} "
        f"params={n_par:7d} best_val_acc={best_acc:.4f} (ep {best_epoch:2d}) "
        f"time={cfg['result']['train_seconds']:.1f}s scalars={best_stats}",
        flush=True,
    )


if __name__ == "__main__":
    main()
