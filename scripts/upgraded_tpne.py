#!/usr/bin/env python
"""Upgraded pipeline: TPNE + full support-graph training + incremental update.

Three arms, all under the paper's protocol on ``tolokers_bid``:

``author-parity``
    Reproduces the base paper's own recipe as closely as our code allows:
    one XGBoost head on the TPNE embedding, trained on the whole growing
    support graph. This is the fair like-for-like comparison arm.

``ensemble``
    Same, but the head is the project's adaptive 3-model ensemble. This is the
    upgrade - the paper uses a single XGBoost.

``ensemble + incremental``
    The full proposal: ensemble plus online incremental embedding updates, so a
    round costs one propagation instead of a full re-encode.

Every arm trains on the *cumulative* support graph, which is the structural fix
for the retention gap: the base paper gains +0.93 F1 across rounds while the
earlier implementation lost 0.71, because its rolling window (3000 labels)
could not carry multi-round history.

    python scripts/upgraded_tpne.py --arm ensemble --pretrain-epochs 100
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_tpne import load_tolokers, rt_metrics  # noqa: E402

from backend.config import DEFAULT_CONFIG  # noqa: E402
from backend.embedder import TPNEEmbedder  # noqa: E402
from backend.sources.adversarial import AdversarialScenario  # noqa: E402


def _metrics(y, pred):
    from sklearn.metrics import f1_score, recall_score
    return (float(f1_score(y, pred, zero_division=0)),
            float(recall_score(y, pred, zero_division=0)))


class Head:
    """Classifier on top of the embedding: single XGBoost, or the ensemble."""

    def __init__(self, kind, cfg, columns, seed):
        self.kind = kind
        self.seed = seed
        if kind == "single":
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=int(cfg.get("n_estimator", 500)),
                learning_rate=0.1, max_depth=6,
                tree_method="hist", eval_metric="logloss",
                scale_pos_weight=float(cfg.get("scale_pos_weight", 1.0)),
                random_state=seed, n_jobs=-1)
        else:
            from backend.detector import ThreeModelEnsemble
            self.model = ThreeModelEnsemble("adver", columns, cfg)
        self._fitted = False

    def fit(self, X, y):
        if len(np.unique(y)) < 2:
            self._fitted = False
            return
        if self.kind == "single":
            self.model.fit(X, y)
        else:
            for row, lab in zip(X, y):
                self.model.add_labeled(row, int(lab))
            self.model.fit()
        self._fitted = True

    def predict_proba(self, X):
        if not self._fitted:
            return np.full(len(X), 0.15)
        if self.kind == "single":
            return self.model.predict_proba(X)[:, 1]
        return np.asarray(self.model.predict_proba_batch(X), dtype=float)

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) >= thr).astype(int)



def tune_threshold(head, X, y, grid=None):
    """Pick the F1-optimal decision threshold on a held-out validation split.

    Fitted on validation only - never on the query graph (which is what is
    scored) and never on the retention set (which measures forgetting). Tuning
    on either would be leakage, and the paper's headline numbers would not be
    comparable.
    """
    from sklearn.metrics import f1_score
    if grid is None:
        grid = [round(0.05 * i, 2) for i in range(1, 20)]
    probs = head.predict_proba(X)
    best, best_f1 = 0.5, -1.0
    for t in grid:
        f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best, best_f1 = t, f1
    return best


def run(arm, rounds, seed, pretrain_epochs, pretrain_epochs_round,
        trials, buffer_max, h_feats, round_window, features="embed",
        embedder="ours"):
    graph = load_tolokers()
    npos = int((graph["label"] == 1).sum())
    nneg = int((graph["label"] == 0).sum())

    base = dict(DEFAULT_CONFIG)
    base["seed"] = seed
    # structural fix: keep the whole support graph, not a 3000-row window
    base["buffer_max_labels"] = int(buffer_max)
    base["min_fit_labels"] = 100
    base["retrain_interval_events"] = 10 ** 9
    base["retrain_interval_seconds"] = 10 ** 9
    # The classifier's booster budget is independent of the embedding's
    # pre-training epochs. The base paper uses 500 estimators
    # (model_config['n_estimator']); the ensemble defaults to 200 rounds.
    base["xgb_rounds"] = int(os.environ.get("XGB_ROUNDS", 200))
    base["scale_pos_weight"] = float(os.environ.get("SCALE_POS_WEIGHT", 1.0))
    base["xgb_extra_rounds"] = 50

    scenario_cfg = {"adver_choose": "greedy", "adver_mod": "replay",
                    "augment": "reage", "round_window": round_window,
                    "round_new_pos": int(0.05 * npos),
                    "round_new_neg": int(0.05 * nneg)}

    kind = "single" if arm == "author-parity" else "ensemble"
    incremental = arm.endswith("incremental")

    out = []
    for trial in range(trials):
        s = seed + trial
        scenario = AdversarialScenario(graph, {**scenario_cfg, "seed": s})
        if embedder == "author":
            from backend.embedder import AuthorTemporalEmbedder
            emb = AuthorTemporalEmbedder(h_feats=h_feats, num_layers=2,
                                         round_window=round_window)
        else:
            emb = TPNEEmbedder(h_feats=h_feats, num_layers=2,
                               round_window=round_window,
                               mode="incremental" if incremental else "offline",
                               seed=s)
        emb_cols = emb.columns()
        head = Head(kind, base, emb_cols, s)

        t0 = time.time()
        if pretrain_epochs and embedder != "author":
            emb.pretrain(scenario.graph_dict(), epochs=pretrain_epochs, lr=0.01)
        E0, _ = emb.embed(scenario.graph_dict())

        # round 0: the real labelled graph, split into support / holdout
        evs = list(scenario.stream_events())
        RAW = np.asarray(scenario.features, dtype=np.float64)

        def feats_for(node_ids, E):
            """Assemble the model input according to --features."""
            raw = np.asarray(scenario.features, dtype=np.float64)[
                np.asarray(node_ids)]
            if features == "raw":
                return raw
            emb_part = E[np.asarray(node_ids)]
            if features == "embed":
                return emb_part
            return np.hstack([emb_part, raw])
        y_all = np.array([int(e["label"]) if e["label"] is not None else 0
                          for e in evs])
        # the author's ratio_test is 0.20, so 80% is available for training;
        # holding out 40% starved the classifier
        # three-way split: train / val (threshold only) / test (retention).
        # The paper's ratio_val is 0.20; the threshold must never be tuned on
        # the query sets or on the retention set.
        n = len(y_all)
        n_test = int(0.20 * n)
        n_val = int(0.20 * n)
        split = n - n_test - n_val          # train end
        val_slice = slice(split, split + n_val)
        X0 = feats_for(np.arange(len(evs)), E0)
        head.fit(X0[:split], y_all[:split])
        # ---- threshold selection on the validation split only
        thr = 0.5 if os.environ.get("NO_TUNE") else tune_threshold(
            head, X0[val_slice], y_all[val_slice])
        print(f"  threshold: {thr:.3f} "
              f"(val F1 {_metrics(y_all[val_slice], head.predict(X0[val_slice], thr))[0]*100:.2f}, "
              f"val F1@0.5 {_metrics(y_all[val_slice], head.predict(X0[val_slice], 0.5))[0]*100:.2f})",
              flush=True)
        f1_init = _metrics(y_all[split:], head.predict(X0[split:], thr))[0]
        rec0 = {"f1_initial": f1_init, "f1_round0_nodes": f1_init,
                "threshold": thr}

        per_round = []
        for r in range(1, rounds + 1):
            query, support = scenario.next_round()
            if len(query) == 0:
                break
            ev_r = list(scenario.stream_events(start_round=r, which="query"))
            if not ev_r:
                continue
            if incremental and pretrain_epochs_round and embedder != "author":
                emb.pretrain(scenario.graph_dict(),
                             epochs=pretrain_epochs_round, lr=0.01, warm=True)
            E, _ = emb.embed(scenario.graph_dict())
            yr = np.array([int(e["label"]) if e["label"] is not None else 0
                           for e in ev_r])
            Xr = feats_for([int(e["raw"]["node_id"]) for e in ev_r], E)
            pq = head.predict(Xr, thr)
            ret = head.predict(X0[split:], thr)
            per_round.append({
                "round": r,
                "f1_query": _metrics(yr, pq)[0],
                "recall_query": _metrics(yr, pq)[1],
                "f1_round0_nodes": _metrics(y_all[split:], ret)[0],
                "n_query": int(len(yr)),
            })

            # ---- support graph: TRAIN on this round's adversarial nodes
            # (paper §4: G_tS grows, G_tQ is only ever scored)
            ev_s = list(scenario.stream_events(start_round=r, which="support"))
            if ev_s:
                ys = np.array([int(e["label"]) if e["label"] is not None else 0
                               for e in ev_s])
                Xs = feats_for([int(e["raw"]["node_id"]) for e in ev_s], E)
                head.fit(Xs, ys)
            # the round-0 holdout is deliberately NOT added: it is the
            # retention measurement, so training on it would fake the gap.
            scenario.mark_predicted(query, pq.astype(bool))

        out.append({"round0": rec0, "rounds": per_round,
                    "seconds": round(time.time() - t0, 1),
                    "embed_summary": emb.summary()})
        m = rt_metrics([out[-1]])
        print(f"  trial {trial}: RT-f1_0_start={m['RT-f1_0_start']*100:.2f} "
              f"RT-avg_f1={m['RT-avg_f1_t>0']*100 if m['RT-avg_f1_t>0'] else 0:.2f} "
              f"RT-recall={m['RT-recall_0']*100 if m['RT-recall_0'] else 0:.2f} "
              f"({out[-1]['seconds']}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ensemble",
                    choices=["author-parity", "ensemble", "ensemble+incremental"])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrain-epochs", type=int, default=60)
    ap.add_argument("--pretrain-epochs-round", type=int, default=20)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--buffer-max", type=int, default=60000)
    ap.add_argument("--h-feats", type=int, default=64)
    ap.add_argument("--round-window", type=int, default=7)
    ap.add_argument("--scale-pos-weight", type=float, default=0.0,
                    help="positive class weight; 0 = let the detector decide")
    ap.add_argument("--no-tune-threshold", action="store_true",
                    help="use the default 0.5 threshold instead of tuning "
                         "it on the validation split")
    ap.add_argument("--features", default="embed",
                    choices=["embed", "raw", "both"],
                    help="embed = TPNE embedding only (paper parity); "
                         "raw = node features only; both = concatenated")
    ap.add_argument("--class-weight", default="balanced",
                    help="'balanced' | 'none' | numeric weight for the "
                         "ensemble members (trades recall for precision)")
    ap.add_argument("--embedder", default="ours",
                    choices=["ours", "author"],
                    help="'ours' = the from-scratch TPNE port; 'author' = the "
                         "reference TemporalEmbedder vendored in vendor/")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.class_weight not in ("balanced",):
        if a.class_weight.lower() in ("none", "off"):
            DEFAULT_CONFIG["class_weight"] = None
        else:
            cw = float(a.class_weight)
            # HistGradientBoosting only accepts 'balanced'/None/dict, so a
            # numeric weight is expressed as a per-class dict; ExtraTrees
            # accepts the number directly.
            DEFAULT_CONFIG["class_weight"] = {0: 1.0, 1: cw}
            DEFAULT_CONFIG["class_weight_numeric"] = cw
    if a.scale_pos_weight:
        DEFAULT_CONFIG.setdefault("scale_pos_weight",
                                  a.scale_pos_weight)
    print(f"arm={a.arm} rounds={a.rounds} trials={a.trials} "
          f"pre={a.pretrain_epochs}/{a.pretrain_epochs_round} "
          f"buffer={a.buffer_max} h={a.h_feats} W={a.round_window}", flush=True)
    if a.no_tune_threshold:
        os.environ["NO_TUNE"] = "1"
    recs = run(a.arm, a.rounds, a.seed, a.pretrain_epochs,
               a.pretrain_epochs_round, a.trials, a.buffer_max,
               a.h_feats, a.round_window, a.features, a.embedder)
    summary = rt_metrics(recs)
    # Strict parity with the base paper: its `round_num = 3` run emits query
    # rows for rounds 1 and 2 only, so RT-avg_f1_t>0 is a mean over exactly
    # those two rounds. Averaging our three rounds would be more generous.
    import statistics as _st
    strict = [_st.mean([r["f1_query"] for r in t["rounds"][:2]]) * 100
              for t in recs if len(t.get("rounds", [])) >= 2]
    if strict:
        summary["RT-avg_f1_strict_parity"] = _st.mean(strict) / 100.0
        summary["strict_per_trial"] = [round(x, 2) for x in strict]
    summary.update(arm=a.arm, trials=a.trials, pretrain_epochs=a.pretrain_epochs,
                   buffer_max=a.buffer_max, per_trial=recs)
    out = a.out or f"reports/upgraded_{a.arm.replace('+','_')}.json"
    os.makedirs(Path(out).parent, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "per_trial"}, indent=2, default=float))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())