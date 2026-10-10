"""Clean offline evaluation for the adaptive 3-model ensemble.

Why this script exists
----------------------
The numbers the dashboard shows are a *streaming* holdout: the deployed models
are trained on the whole rolling window and then scored on a slice of that same
window, and because XGBoost/HistGB are warm-started from earlier rounds they
have already seen those rows.  They are useful for watching the system adapt
live, and they are optimistic - they must not be quoted as research results.

This script produces the reportable numbers.  The split is strictly temporal and
one-way:

    rows  [0 ............ n_train)   -> training window, fitted
    rows  [n_train ...... n_total)   -> clean holdout, never fitted on

The holdout is built *after* the last training event is absorbed, so its profile
features reflect real history and its labels were never shown to any model.  No
warm-start continuation, no re-fitting, no resampling across the boundary.

Usage
-----
    python scripts/evaluate.py
    python scripts/evaluate.py --limit 60000 --holdout 20000 --json out.json
    python scripts/evaluate.py --source ieee
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import load_config                      # noqa: E402
from backend.detector import ThreeModelEnsemble             # noqa: E402
from backend.features import ORACLE_COLUMNS, columns_for    # noqa: E402
from backend.sources.replay import IeeeReplaySource            # noqa: E402


def build_rows(config, source, limit, use_v=False):
    """Materialise `limit` canonical events in time order, with the feature
    vector of each event built from the profile state that precedes it."""
    cfg = dict(config)
    cfg["stream_slice"] = int(limit)
    if use_v:
        cfg.setdefault("ieee", {})["use_v_features"] = True
    src = IeeeReplaySource(cfg)
    status = src.load()
    if not status.get("loaded"):
        raise SystemExit("dataset unavailable: %s (%s)" % (status.get("error"),
                                                            status.get("hint")))
    from backend.features import FeatureBuilder, ieee_columns, IEEE_NATIVE_COLUMNS as IEEE_BASE_COLS

    extra = None
    if use_v and source == "ieee":
        extra = {"ieee": ieee_columns(True)[len(IEEE_BASE_COLS):]}
        cols = FeatureBuilder(extra).columns_for_source(source)
    else:
        cols = columns_for(source)
    builder = FeatureBuilder(extra)
    rows, labels, amounts, times = [], [], [], []
    for ev in src.stream(_NeverStop()):
        if ev is None:
            # the source yields None as its end-of-stream sentinel; requesting
            # more rows than the (sliced) dataset holds must not crash the run
            break
        vec = builder.build(ev)
        rows.append([float(vec[c]) for c in cols])
        labels.append(int(ev.get("label") if ev.get("label") is not None else 0))
        amounts.append(float(ev.get("amount") or 0.0))
        times.append(float(ev["ts"]))
        builder.absorb(ev)
        if len(rows) >= limit:
            break
    return status, cols, np.asarray(rows, dtype=float), np.asarray(labels, dtype=int), \
        np.asarray(amounts, dtype=float), np.asarray(times, dtype=float)


class _NeverStop:
    def is_set(self):
        return False

    def wait(self, _t=None):
        return False


def _metrics(y_true, probs, thresholds=(0.3, 0.5, 0.7)):
    out = {}
    for thr in thresholds:
        pred = (probs >= thr).astype(int)
        tp = int(np.count_nonzero((pred == 1) & (y_true == 1)))
        fp = int(np.count_nonzero((pred == 1) & (y_true == 0)))
        fn = int(np.count_nonzero((pred == 0) & (y_true == 1)))
        tn = int(np.count_nonzero((pred == 0) & (y_true == 0)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out["thr_%.2f" % thr] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "accuracy": round((tp + tn) / max(1, len(y_true)), 4),
        }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        out["pr_auc"] = round(float(average_precision_score(y_true, probs)), 4)
        out["roc_auc"] = round(float(roc_auc_score(y_true, probs)), 4)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="ieee", choices=["ieee", "chain", "webhook"])
    ap.add_argument("--limit", type=int, default=40000,
                    help="total events to build (train + holdout)")
    ap.add_argument("--holdout", type=int, default=10000,
                    help="rows reserved as the clean temporal holdout")
    ap.add_argument("--balance", type=float, default=2.0,
                    help="max negative:positive ratio in the training window")
    ap.add_argument("--json", default=None, help="also write the report as JSON")
    ap.add_argument("--v-features", action="store_true",
                    help="also model the anonymised V1..V339 columns "
                         "(widens the vector from 26 to ~353 features)")
    args = ap.parse_args()

    config = load_config()
    if args.limit < 500:
        raise SystemExit("--limit must be at least 500 to fit anything")
    if args.holdout >= args.limit - 120:
        raise SystemExit("--holdout leaves too few training rows (need >= 120 labels)")

    print("loading %s ..." % args.source)
    t0 = time.time()
    status, cols, X, y, amounts, times = build_rows(config, args.source, args.limit,
                                                 args.v_features)
    load_s = time.time() - t0
    print("  %d events in %.1fs  (fraud rate %.4f, path %s)"
          % (len(y), load_s, y.mean(), status.get("path")))

    n_hold = args.holdout
    n_train = len(y) - n_hold
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]

    # The slice is contiguous in time, so a single boundary is enough; verify it
    # rather than assume it (a source with out-of-order timestamps would
    # invalidate the whole claim).
    if times[n_train] < times[0]:
        raise SystemExit("events are not in time order - cannot claim a temporal split")
    span = (times[n_train - 1] - times[0]) / 3600.0
    hold_span = (times[-1] - times[n_train]) / 3600.0
    print("  train window  rows[0:%d]  %d fraud (%.4f)  spanning %.1f h"
          % (n_train, int(ytr.sum()), ytr.mean(), span))
    print("  clean holdout rows[%d:%d] %d fraud (%.4f)  spanning %.1f h"
          % (n_train, len(y), int(yte.sum()), yte.mean(), hold_span))
    if ytr.sum() == 0 or yte.sum() == 0:
        raise SystemExit("need fraud on both sides of the split; raise --limit")

    # down-sample the majority class of the *training* window only
    cfg = dict(config)
    cfg["min_fit_labels"] = 1
    ens = ThreeModelEnsemble(args.source, cols, cfg)
    pos = np.where(ytr == 1)[0]
    neg = np.where(ytr == 0)[0]
    keep_neg = min(len(neg), max(2 * len(pos), int(args.balance * len(pos))))
    rng = np.random.default_rng(cfg.get("seed", 42))
    if keep_neg < len(neg):
        neg = np.sort(rng.choice(neg, size=keep_neg, replace=False))
    train_idx = np.sort(np.concatenate([pos, neg]))
    print("  fitting on %d rows (%d fraud, %.1f%% negative cap)"
          % (len(train_idx), int(ytr[train_idx].sum()), 100.0 * args.balance))

    for i in train_idx:
        ens.add_labeled(Xtr[i], int(ytr[i]))
    t0 = time.time()
    snap = ens.fit()
    fit_s = time.time() - t0
    print("  fit %.1fs  breakdown %s" % (fit_s, snap.get("last_fit_breakdown")))

    t0 = time.time()
    probs = np.asarray(ens.predict_proba_batch(Xte), dtype=float)
    score_s = time.time() - t0
    print("  scored %d holdout rows in %.2fs (%.0f rows/s)\n"
          % (len(probs), score_s, len(probs) / max(score_s, 1e-9)))

    weights = snap.get("weights") or {}
    report = {
        "generated_by": "scripts/evaluate.py",
        "source": args.source,
        "events": int(len(y)),
        "train_rows": int(len(train_idx)),
        "holdout_rows": int(n_hold),
        "fraud_rate_train": round(float(ytr.mean()), 5),
        "fraud_rate_holdout": round(float(yte.mean()), 5),
        "train_span_hours": round(span, 2),
        "holdout_span_hours": round(hold_span, 2),
        "fit_seconds": round(fit_s, 2),
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "oracle_columns_excluded": sorted(ORACLE_COLUMNS),
        "feature_count": len(cols),
        "holdout_protocol": "strictly temporal, trailing, disjoint from every fit",
        "ensemble_holdout": _metrics(yte, probs),
    }
    per_model = {}
    models = ens.deployed_models()
    for name, m in models.items():
        try:
            if name == "XGBoost":
                import xgboost as xgb
                p = np.asarray(m.predict(xgb.DMatrix(Xte)), dtype=float)
            else:
                p = np.asarray(m.predict_proba(Xte)[:, 1], dtype=float)
        except Exception as exc:
            per_model[name] = {"error": str(exc)}
            continue
        per_model[name] = _metrics(yte, p)
    report["models_holdout"] = per_model

    def line(tag, d):
        h = d.get("thr_0.50", {})
        return ("  %-22s F1 %.4f  P %.4f  R %.4f  acc %.4f  PR-AUC %s  ROC-AUC %s"
                % (tag, h.get("f1", 0), h.get("precision", 0), h.get("recall", 0),
                   h.get("accuracy", 0), d.get("pr_auc", "-"), d.get("roc_auc", "-")))

    print("clean temporal holdout @0.50  (reportable):")
    for name, d in per_model.items():
        print(line(name, d))
    print(line("ENSEMBLE (adaptive)", report["ensemble_holdout"]))
    print("\nweights: %s" % {k: round(v, 4) for k, v in weights.items()})
    print("streaming dashboard numbers for comparison (optimistic, do not cite):")
    print("  per-model holdout F1: %s"
          % {k: (v or {}).get("f1") for k, v in (snap.get("models") or {}).items()})
    print("  internal holdout mode: %s%s"
          % (snap.get("holdout_mode"),
             " (degenerate -> stratified fallback)" if snap.get("holdout_degenerate") else ""))
    print("\n%s" % ("-" * 78))
    print("The dashboard's own metrics are labelled 'streaming holdout (optimistic)':")
    print("they are computed on a slice of the window the models were fitted on, and")
    print("the XGBoost/HistGB boosters resume earlier rounds of that same window.")
    print("Cite the clean holdout above in the paper.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
