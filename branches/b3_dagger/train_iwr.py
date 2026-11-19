"""支线 3 (v2) · 聚合与干预加权回归训练器（IWR + 语言条件状态去重 + 零数据浪费 + 统一 on-policy 探针）。

核心修复与改进（详见 branches/b3_dagger/LOG.md §12）：
  1. 语言条件状态去重：去重键从 `(image, dir)` 修正为 `(image, dir, mission)`，避免跨指令状态误删；
     同时修复多轮聚合时跨文件未更新全局已见集的问题。
  2. 取消 20% 采集样本丢弃：100% 采集样本用于训练；`onpolicy_action_acc` 改为在独立的验证种子
     on-policy 探针集（只读、绝不参训、绝不参与选模）上统一评测，实现跨方法严格同分布可比。
  3. 干预加权回归（Intervention-Weighted Regression, Mandlekar et al., CoRL 2020; Sirius, RSS 2023）：
     对 `π_rollout(s) != a*(s)` 的关键纠偏样本赋予权重 `w_intv >= 1.0`，防止海量常规专家样本淹没稀疏纠偏梯度。
  4. 支持严格等预算切分（`--expert-limit 3000`）：3000 专家 + 1500 交互采集 = 4500 训练种子（零额外种子预算）。
  5. 选模准则严格不变：**仅使用纯净 500 条 `expert_val` 上的动作准确率 `val_acc`**（手册硬性规定 B12）。
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "branches" / "_shared"))
sys.path.insert(0, str(ROOT / "branches" / "b2_film"))
sys.path.insert(0, str(ROOT / "src"))

from branch_utils import frozen_diff_guard  # noqa: E402
from common import (  # noqa: E402
    ENV_ID,
    MAX_TOKENS,
    PAD_IDX,
    TRAIN_SEEDS,
    VAL_SET,
    VOCAB,
    BCNet,
    assert_no_leak,
    mission_to_ids,
    set_all_seeds,
)
from train_bc import build_dataset, eval_model, make_ds  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_official_data(path: str = "data/expert_5000.pkl", expert_limit: int = 0):
    eps = pickle.load(open(path, "rb"))
    train_eps = [e for e in eps if e.get("split") == "train"]
    val_eps = [e for e in eps if e.get("split") == "val"]
    if expert_limit > 0:
        allowed = set(TRAIN_SEEDS[:expert_limit])
        train_eps = [e for e in train_eps if e["seed"] in allowed]
    assert_no_leak(e["seed"] for e in train_eps + val_eps)
    return train_eps, val_eps


def state_key(img: np.ndarray, d: int, mission: str, key_mode: str):
    if key_mode == "id_only":
        return (img.tobytes(), int(d))
    return (img.tobytes(), int(d), mission)


def load_extra_files(
    paths: list[str],
    dedup: str,
    key_mode: str,
    expert_eps: list[dict],
) -> tuple[list[dict], dict]:
    seen_cross: set = set()
    if dedup == "global":
        for e in expert_eps:
            m = e["mission"]
            for img, d in zip(e["images"], e["directions"]):
                seen_cross.add(state_key(img, d, m, key_mode))

    merged_eps: list[dict] = []
    sources = []
    total_queries = 0
    total_raw = 0
    total_kept = 0
    total_disagree_kept = 0

    for p in paths:
        obj = pickle.load(open(p, "rb"))
        eps = obj["episodes"]
        meta = {k: v for k, v in obj["meta"].items() if k != "seeds"}
        assert_no_leak(e["seed"] for e in eps)
        bad_val = {e["seed"] for e in eps} & set(VAL_SET)
        assert not bad_val, f"❌ 训练采集数据混入 VAL_SEEDS: {sorted(bad_val)[:5]}"

        n_raw = n_kept = n_dis = 0
        for e in eps:
            m = e["mission"]
            dis_list = e.get("disagree")
            if dis_list is None:
                pa_list = e.get("policy_actions", [])
                if pa_list and len(pa_list) == len(e["actions"]):
                    dis_list = [int(pa) != int(ea) for pa, ea in zip(pa_list, e["actions"])]
                else:
                    dis_list = [False] * len(e["actions"])

            imgs, dirs, acts, dis_kept = [], [], [], []
            seen_ep: set = set()
            for img, d, a, is_dis in zip(e["images"], e["directions"], e["actions"], dis_list):
                n_raw += 1
                k = state_key(img, d, m, key_mode)
                if dedup in ("episode", "cross", "global") and k in seen_ep:
                    continue
                if dedup in ("cross", "global") and k in seen_cross:
                    continue
                seen_ep.add(k)
                if dedup in ("cross", "global"):
                    seen_cross.add(k)
                imgs.append(img)
                dirs.append(int(d))
                acts.append(int(a))
                dis_kept.append(bool(is_dis))
                n_kept += 1
                if is_dis:
                    n_dis += 1

            if imgs:
                merged_eps.append(
                    {
                        "seed": e["seed"],
                        "mission": m,
                        "images": imgs,
                        "directions": dirs,
                        "actions": acts,
                        "disagree": dis_kept,
                        "episode_len": len(imgs),
                    }
                )

        q = meta.get("n_oracle_queries", meta.get("n_transitions", n_raw))
        total_queries += int(q)
        total_raw += n_raw
        total_kept += n_kept
        total_disagree_kept += n_dis
        sources.append(
            {
                "path": p,
                "n_raw": n_raw,
                "n_kept": n_kept,
                "n_disagree_kept": n_dis,
                "n_oracle_queries": int(q),
                **meta,
            }
        )

    summary_meta = {
        "dedup": dedup,
        "key_mode": key_mode,
        "n_raw_total": total_raw,
        "n_kept_total": total_kept,
        "n_disagree_kept_total": total_disagree_kept,
        "n_oracle_queries_total": total_queries,
        "sources": sources,
    }
    return merged_eps, summary_meta


def build_weighted_tensors(episodes: list[dict], intv_weight: float):
    images, tokens, dirs, acts, weights = [], [], [], [], []
    for ep in episodes:
        ids = mission_to_ids(ep["mission"])
        ids = (ids + [PAD_IDX] * MAX_TOKENS)[:MAX_TOKENS]
        dis_list = ep.get("disagree", [False] * len(ep["actions"]))
        for img, d, a, is_dis in zip(ep["images"], ep["directions"], ep["actions"], dis_list):
            images.append(img)
            tokens.append(ids)
            dirs.append(int(d))
            acts.append(int(a))
            weights.append(float(intv_weight) if is_dis else 1.0)
    return (
        torch.from_numpy(np.asarray(images, dtype=np.uint8)),
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(dirs, dtype=torch.long),
        torch.tensor(acts, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )


@torch.no_grad()
def eval_probe(model: nn.Module, probe_path: str) -> dict:
    p = Path(probe_path)
    if not p.exists():
        return {"all_acc": None, "disagree_acc": None}
    obj = pickle.load(open(p, "rb"))
    eps = obj["episodes"]
    all_tr = build_dataset(eps)
    crit = nn.CrossEntropyLoss()
    _, all_acc, _, _ = eval_model(model, DataLoader(make_ds(*all_tr), batch_size=512), crit)

    # 在 π1 犯错状态子集上的纠错准确率（真正的 recovery accuracy）
    dis_eps = []
    for e in eps:
        imgs, dirs, acts = [], [], []
        for img, d, a, dis in zip(e["images"], e["directions"], e["actions"], e.get("disagree", [])):
            if dis:
                imgs.append(img)
                dirs.append(d)
                acts.append(a)
        if imgs:
            dis_eps.append({"mission": e["mission"], "images": imgs, "directions": dirs, "actions": acts})
    dis_acc = None
    if dis_eps:
        dis_tr = build_dataset(dis_eps)
        _, dis_acc, _, _ = eval_model(model, DataLoader(make_ds(*dis_tr), batch_size=512), crit)
    return {
        "all_acc": round(float(all_acc), 6),
        "disagree_acc": round(float(dis_acc), 6) if dis_acc is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v2_unstick_iwr", help="实验组标签（写入 config）")
    ap.add_argument("--arch", default="bc", choices=["bc", "film"])
    ap.add_argument("--extra-data", default="", help="逗号分隔的交互/额外采集 pkl 路径")
    ap.add_argument("--expert-limit", type=int, default=0, help="0=全量 4500 专家；>0 取前 K 条（如 3000）")
    ap.add_argument("--dedup", default="cross", choices=["none", "episode", "cross", "global"])
    ap.add_argument("--dedup-key", default="idm", choices=["idm", "id_only"])
    ap.add_argument("--intv-weight", type=float, default=1.0, help="IWR 干预样本权重 w_intv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--film-l2", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--official-data", default="data/expert_5000.pkl")
    ap.add_argument("--probe-data", default="data/b3_v2_val_probe.pkl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frozen_diff_guard()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    set_all_seeds(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    expert_train, expert_val = load_official_data(args.official_data, args.expert_limit)
    n_expert_transitions = sum(len(e["actions"]) for e in expert_train)

    extra_eps, extra_meta = [], {"n_oracle_queries_total": 0, "n_kept_total": 0, "n_disagree_kept_total": 0}
    if args.extra_data:
        paths = [p.strip() for p in args.extra_data.split(",") if p.strip()]
        extra_eps, extra_meta = load_extra_files(paths, args.dedup, args.dedup_key, expert_train)

    train_eps = expert_train + extra_eps
    tr_tensors = build_weighted_tensors(train_eps, args.intv_weight)
    va_tensors = build_dataset(expert_val)

    train_ds = TensorDataset(*tr_tensors)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(make_ds(*va_tensors), batch_size=512, shuffle=False, num_workers=0)

    if args.arch == "bc":
        model = BCNet(vocab_size=len(VOCAB), obj_emb_dim=8, lang_dim=32, hidden=128, dropout=0.0)
    else:
        from film_net import build as build_film
        model = build_film("film", vocab_size=len(VOCAB), obj_emb_dim=8, lang_dim=32, hidden=128, dropout=0.0)

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    val_crit = nn.CrossEntropyLoss()

    history: list[dict] = []
    best_score, best_epoch, best_state = -float("inf"), -1, None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_w, run_n, run_c = 0.0, 0.0, 0, 0
        for img, tok, d, act, w in train_loader:
            logits = model(img, tok, d)
            per_loss = F.cross_entropy(logits, act, reduction="none")
            loss = (per_loss * w).sum() / w.sum()
            if args.arch == "film" and args.film_l2 > 0:
                loss = loss + args.film_l2 * model.film_penalty(tok)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            run_loss += (per_loss * w).sum().item()
            run_w += w.sum().item()
            run_n += act.size(0)
            run_c += (logits.argmax(-1) == act).sum().item()

        val_loss, val_acc, _, _ = eval_model(model, val_loader, val_crit)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(run_loss / max(1e-9, run_w), 6),
                "train_acc": round(run_c / max(1, run_n), 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
        )
        # 严格仅按纯净验证集动作准确率选模（手册硬规定）
        if val_acc > best_score + 1e-9:
            best_score, best_epoch = val_acc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_loss, val_acc, pred, lab = eval_model(model, val_loader, val_crit)
    probe_res = eval_probe(model, args.probe_data)

    torch.save(model.state_dict(), out / "best_model.pt")
    np.savez(out / "predictions_val.npz", pred=pred, label=lab)
    (out / "vocab.json").write_text(
        json.dumps({"vocab": VOCAB, "pad_idx": PAD_IDX, "max_tokens": MAX_TOKENS}, ensure_ascii=False)
    )

    arch_cfg = {
        "img_mode": "embed",
        "obj_emb_dim": 8,
        "lang_dim": 32,
        "hidden": 128,
        "dropout": 0.0,
        "n_params": n_params,
    }
    if args.arch == "film":
        arch_cfg["variant"] = "film"

    total_oracle_labels = n_expert_transitions + int(extra_meta.get("n_oracle_queries_total", 0))
    cfg = {
        "branch": "b3_dagger",
        "variant": args.tag,
        "run": str(out),
        "git_commit": git_commit(),
        "env_id": ENV_ID,
        "seed": args.seed,
        "arch": arch_cfg,
        "optim": {
            "optimizer": "Adam",
            "lr": args.lr,
            "weight_decay": 0.0,
            "film_l2": args.film_l2 if args.arch == "film" else 0.0,
            "intv_weight": args.intv_weight,
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "early_stop_patience": 0,
            "early_stop_metric": "val_action_accuracy",
        },
        "data": {
            "official_data": args.official_data,
            "n_expert_train": len(expert_train),
            "n_expert_transitions": n_expert_transitions,
            "n_extra_episodes": len(extra_eps),
            "n_extra_transitions_kept": int(extra_meta.get("n_kept_total", 0)),
            "n_disagree_transitions_kept": int(extra_meta.get("n_disagree_kept_total", 0)),
            "n_extra_oracle_queries": int(extra_meta.get("n_oracle_queries_total", 0)),
            "total_oracle_labels": total_oracle_labels,
            "n_train_episodes": len(train_eps),
            "n_train_transitions": int(len(tr_tensors[3])),
            "n_val_episodes": len(expert_val),
            "n_val_transitions": int(len(va_tensors[3])),
            "dedup": args.dedup,
            "dedup_key": args.dedup_key,
            "extra_meta": extra_meta,
        },
        "result": {
            "best_val_acc": round(best_score, 6),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "train_seconds": round(time.time() - t0, 2),
            "onpolicy_action_acc": probe_res["all_acc"],
            "onpolicy_recovery_acc": probe_res["disagree_acc"],
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (out / "history.json").write_text(json.dumps(history, indent=2))

    print(
        f"[{args.tag} s{args.seed}] val_acc={best_score:.4f}@ep{best_epoch} "
        f"onpol_acc={probe_res["all_acc"]} recov_acc={probe_res["disagree_acc"]} "
        f"oracle_labels={total_oracle_labels} train_trans={len(tr_tensors[3])} ({cfg["result"]["train_seconds"]}s)"
    )


if __name__ == "__main__":
    main()
