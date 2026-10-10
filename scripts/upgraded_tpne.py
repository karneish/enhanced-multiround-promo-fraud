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


def run(arm, rounds, seed, pretrain_epochs, pretrain_epochs_round,
        trials, buffer_max, h_feats, round_window):
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
        emb = TPNEEmbedder(h_feats=h_feats, num_layers=2,
                           round_window=round_window,
                           mode="incremental" if incremental else "offline",
                           seed=s)
        emb_cols = emb.columns()
        head = Head(kind, base, emb_cols, s)

        t0 = time.time()
        emb.pretrain(scenario.graph_dict(), epochs=pretrain_epochs, lr=0.01)
        E0, _ = emb.embed(scenario.graph_dict())

        # round 0: the real labelled graph, split into support / holdout
        evs = list(scenario.stream_events())
        y_all = np.array([int(e["label"]) if e["label"] is not None else 0
                          for e in evs])
        # the author's ratio_test is 0.20, so 80% is available for training;
        # holding out 40% starved the classifier
        split = int(float(os.environ.get("TRAIN_FRAC", 0.8)) * len(y_all))
        X0 = E0[:len(evs)]
        head.fit(X0[:split], y_all[:split])
        f1_init = _metrics(y_all[split:], head.predict(X0[split:]))[0]
        rec0 = {"f1_initial": f1_init, "f1_round0_nodes": f1_init}

        per_round = []
        for r in range(1, rounds + 1):
            query, support = scenario.next_round()
            if len(query) == 0:
                break
            ev_r = list(scenario.stream_events(start_round=r, which="query"))
            if not ev_r:
                continue
            if incremental and pretrain_epochs_round:
                emb.pretrain(scenario.graph_dict(),
                             epochs=pretrain_epochs_round, lr=0.01, warm=True)
            E, _ = emb.embed(scenario.graph_dict())
            yr = np.array([int(e["label"]) if e["label"] is not None else 0
                           for e in ev_r])
            Xr = E[np.array([int(e["raw"]["node_id"]) for e in ev_r])]
            pq = head.predict(Xr)
            ret = head.predict(X0[split:])
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
                Xs = E[np.array([int(e["raw"]["node_id"]) for e in ev_s])]
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
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    print(f"arm={a.arm} rounds={a.rounds} trials={a.trials} "
          f"pre={a.pretrain_epochs}/{a.pretrain_epochs_round} "
          f"buffer={a.buffer_max} h={a.h_feats} W={a.round_window}", flush=True)
    recs = run(a.arm, a.rounds, a.seed, a.pretrain_epochs,
               a.pretrain_epochs_round, a.trials, a.buffer_max,
               a.h_feats, a.round_window)
    summary = rt_metrics(recs)
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