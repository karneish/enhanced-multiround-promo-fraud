#!/usr/bin/env python
"""Does training the reference embedder close the round-0 gap?

The from-scratch TPNE port scores 57.25 on round-0 where the paper's
``XGB-SP`` reaches 66.47. This trains the author's *actual*
``TemporalEmbedder`` through its own differentiable loss and evaluates the
resulting embedding on the paper's round-0 split, reporting the trend so a
single run shows whether the gap is a training-budget problem.

Round 0 uses only the real dataset, so this is the clean like-for-like:
identical data, identical split, identical classifier family.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "vendor" / "author-tpne" / "src"))

from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from backend.config import DEFAULT_CONFIG  # noqa: E402
from backend.detector import ThreeModelEnsemble  # noqa: E402
from backend.embedder import AuthorTemporalEmbedder  # noqa: E402

from compare_tpne import load_tolokers  # noqa: E402
from upgraded_tpne import author_split  # noqa: E402


def evaluate(E, y, i_tr, i_va, i_te, seed, thr=None):
    cfg = dict(DEFAULT_CONFIG)
    cfg["seed"] = seed
    cfg["buffer_max_labels"] = 60000
    cfg["min_fit_labels"] = 100
    cfg["retrain_interval_events"] = 10 ** 9
    cfg["retrain_interval_seconds"] = 10 ** 9
    cfg["class_weight"] = None
    cfg["scale_pos_weight"] = 1.0
    ens = ThreeModelEnsemble("adver", [f"c{i}" for i in range(E.shape[1])], cfg)
    for row, lab in zip(E[i_tr], y[i_tr]):
        ens.add_labeled(row, int(lab))
    ens.fit()
    pv = ens.predict_proba_batch(E[i_va])
    if thr is None:
        best, bf = 0.5, -1.0
        for k in range(1, 20):
            t = round(0.05 * k, 2)
            f = f1_score(y[i_va], (np.asarray(pv) >= t).astype(int),
                         zero_division=0)
            if f > bf:
                best, bf = t, f
        thr = best
    pt = np.asarray(ens.predict_proba_batch(E[i_te]))
    pred = (pt >= thr).astype(int)
    return {
        "f1": float(f1_score(y[i_te], pred, zero_division=0)),
        "auc": float(roc_auc_score(y[i_te], pt)) if len(np.unique(y[i_te])) > 1 else None,
        "thr": thr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--checkpoint", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--h-feats", type=int, default=64)
    ap.add_argument("--round-window", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/embedder_training_curve.json")
    a = ap.parse_args()

    g = load_tolokers()
    y = np.asarray(g["label"]).astype(int)
    graph = {"features": g["features"], "edge_index": g["edge_index"],
             "age": np.zeros(len(y))}
    i_tr, i_va, i_te = author_split(g, seed=a.seed)
    print(f"split: train={len(i_tr)} val={len(i_va)} test={len(i_te)} "
          f"(overlap {len(set(i_tr) & set(i_te))})", flush=True)
    print("paper's round-0 test F1 to beat: 0.6647", flush=True)

    emb = AuthorTemporalEmbedder(h_feats=a.h_feats, num_layers=2,
                                 round_window=a.round_window)
    curve = []
    E0, _ = emb.embed(graph)
    base = evaluate(E0, y, i_tr, i_va, i_te, a.seed)
    curve.append({"epochs": 0, **base,
                  "loss": {k: round(v, 4) for k, v in emb.last_loss.items()}})
    print(f"epoch   0: F1 {base['f1']:.4f}  AUC {base['auc']:.4f}  "
          f"thr {base['thr']}", flush=True)

    done = 0
    while done < a.epochs:
        chunk = min(a.checkpoint, a.epochs - done)
        t0 = time.time()
        loss = emb.pretrain(graph, epochs=chunk, lr=a.lr, warm=(done > 0))
        done += chunk
        E, _ = emb.embed(graph)
        r = evaluate(E, y, i_tr, i_va, i_te, a.seed)
        curve.append({"epochs": done, **r,
                      "loss": {k: round(v, 4) for k, v in loss.items()},
                      "seconds": round(time.time() - t0, 1)})
        print(f"epoch {done:>3}: F1 {r['f1']:.4f}  AUC {r['auc']:.4f}  "
              f"thr {r['thr']}  loss {r and loss.get('final_loss', 0):.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(curve, indent=2, default=float))
    print("\nwrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())