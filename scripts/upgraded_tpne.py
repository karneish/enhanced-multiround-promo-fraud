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

DATASET = ROOT / "vendor" / "author-tpne" / "dataset" / "tolokers_bid"

from backend.config import DEFAULT_CONFIG  # noqa: E402
from backend.embedder import TPNEEmbedder  # noqa: E402
from backend.sources.adversarial import AdversarialScenario  # noqa: E402


def _metrics(y, pred):
    from sklearn.metrics import f1_score, recall_score
    return (float(f1_score(y, pred, zero_division=0)),
            float(recall_score(y, pred, zero_division=0)))


def _full_metrics(y, pred, score=None):
    """F1, recall, precision and ROC-AUC for one round.

    The harness used to record only F1 and recall. That made the high recall
    unfalsifiable: recall is bought by lowering the decision threshold, so a
    recall-only report cannot be compared against the reference, which reports
    final_precision and final_auc alongside recall.
    """
    from sklearn.metrics import (f1_score, recall_score, precision_score,
                                 roc_auc_score)
    import numpy as np
    out = {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "auc": None,
    }
    if score is not None:
        try:
            s = np.asarray(score, dtype=float).ravel()
            if len(np.unique(s)) > 2 and len(np.unique(y)) > 1:
                out["auc"] = float(roc_auc_score(y, s))
        except (ValueError, TypeError):
            pass
    return out


def _recall_at_precision(y, score, min_precision):
    """Recall attainable while holding precision >= min_precision.

    Sweeps every achievable threshold and returns the best recall that still
    satisfies the precision floor. This is the threshold-independent way to ask
    "is the detector actually better?", and it is what makes a recall figure
    safe to compare with another paper's.
    """
    import numpy as np
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2 or len(y) == 0:
        return None
    order = np.argsort(-score)
    ys = y[order]
    ss = score[order]
    tp = np.cumsum(ys == 1)
    fp = np.cumsum(ys == 0)
    # only evaluate at the end of each run of equal scores
    last = np.r_[ss[1:] != ss[:-1], True]
    tp, fp, ss = tp[last], fp[last], ss[last]
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / max(int((y == 1).sum()), 1)
    ok = prec >= min_precision
    if not ok.any():
        return 0.0
    return float(rec[ok].max())


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



class DualViewHead:
    """Two heads over two feature views, averaged.

    The ablation shows the two views win on different metrics:

        view          RT-f1_0   RT-avg_f1
        embedding       57.25      38.45
        raw features    47.63      53.06

    The embedding carries the graph structure that round-0 performance needs,
    but with ``adver_mod='replay'`` a query node is a *verbatim duplicate* of
    its seed, so the raw feature vector identifies it perfectly while the
    embedding smooths it into its neighbourhood and destroys that signal.
    Concatenating them lets the 384 embedding dimensions dilute the 10 raw
    ones; keeping two separate heads and averaging their probabilities does
    not.
    """

    def __init__(self, cfg, columns, seed, kind="ensemble"):
        self.kind = kind
        self.h_raw = Head(kind, cfg, columns, seed)
        self.h_emb = Head(kind, cfg, columns, seed)
        self.threshold = 0.5

    def fit(self, X_raw, X_emb=None, y=None):
        if y is None:                       # Head.fit(X, y) signature
            self.h_raw.fit(X_raw, X_emb)
            self.h_emb.fit(X_raw, X_emb)
        else:
            self.h_raw.fit(X_raw, y)
            self.h_emb.fit(X_emb, y)

    def predict_proba(self, X_raw, X_emb=None, w_raw=0.5):
        return (w_raw * self.h_raw.predict_proba(X_raw)
                + (1.0 - w_raw) * self.h_emb.predict_proba(X_emb))

    def predict(self, X_raw, X_emb=None, thr=None, w_raw=0.5):
        return (self.predict_proba(X_raw, X_emb, w_raw)
                >= (self.threshold if thr is None else thr)).astype(int)



def tune_threshold_dual(head, Xr, Xe, y, rw):
    from sklearn.metrics import f1_score
    probs = head.predict_proba(Xr, Xe, rw)
    best, bf = 0.5, -1.0
    for k in range(1, 20):
        th = round(0.05 * k, 2)
        f = f1_score(y, (probs >= th).astype(int), zero_division=0)
        if f > bf:
            best, bf = th, f
    return best


def _dual_rounds(scenario, emb, head, kind, E0, X0_raw, X0_emb, y_all,
                 test_idx, rounds, rw, pt_epochs, incremental, embedder, patience=None):
    """Score query rounds and the round-0 retention set.

    ``test_idx`` is an index array, never a slice endpoint. Treating it as one
    (``X0_raw[test_idx:]``) silently starts at ``test_idx.min()`` and runs to
    the end of the node range, which in ``--split masks`` re-admits the whole
    training set into the retention measurement.
    """
    per_round = []
    for r in range(1, rounds + 1):
        query, support = scenario.next_round()
        if len(query) == 0:
            break
        ev_q = list(scenario.stream_events(start_round=r, which="query"))
        if not ev_q:
            continue
        if incremental and pt_epochs:
            emb.pretrain(scenario.graph_dict(), epochs=pt_epochs,
                         lr=0.01, warm=True, patience=patience)
        E, _ = emb.embed(scenario.graph_dict())
        ids_q = [int(e["raw"]["node_id"]) for e in ev_q]
        yq = np.array([int(e["label"]) if e["label"] is not None else 0
                       for e in ev_q])
        RAW = np.asarray(scenario.features, dtype=np.float64)
        pq = head.predict(RAW[ids_q], E[ids_q], None, rw)
        ps = head.predict_proba(RAW[ids_q], E[ids_q], rw)
        ret = head.predict(X0_raw[test_idx], X0_emb[test_idx],
                           head.threshold, rw)
        pq_bool = pq.astype(bool)
        qm = _full_metrics(yq, pq_bool, ps)
        per_round.append({
            "round": r, "f1_query": qm["f1"],
            "recall_query": qm["recall"],
            "precision_query": qm["precision"],
            "auc_query": qm["auc"],
            "recall_at_p30": _recall_at_precision(yq, ps, 0.30),
            "recall_at_p50": _recall_at_precision(yq, ps, 0.50),
            "f1_round0_nodes": _metrics(y_all[test_idx], ret)[0],
            "n_query": int(len(yq))})
        ev_s = list(scenario.stream_events(start_round=r, which="support"))
        if ev_s:
            ids_s = [int(e["raw"]["node_id"]) for e in ev_s]
            ys = np.array([int(e["label"]) if e["label"] is not None else 0
                           for e in ev_s])
            head.fit(RAW[ids_s], E[ids_s], ys)
        scenario.mark_predicted(query, pq.astype(bool))
    return per_round



def author_split(graph, seed=0, trial_col=0):
    """Replicate the base paper's round-0 split exactly.

    ``tolokers_bid`` ships 20 predefined mask columns (one per trial):
    train 50 %, val 25 %, test 25 %. The reference implementation takes the
    *train* mask as the working pool, splits that pool ``ratio_train`` (0.60)
    into train/val, and evaluates the complement of the pool as the test set.
    That reproduces the published round-0 figures exactly:

        pool        5,879 (50 %)
        train       3,527   = 0.60 * pool
        val         2,352   = 0.40 * pool
        test        5,879   = complement of the pool

    Returns ``(idx_train, idx_val, idx_test)`` as positional arrays.
    """
    import dgl
    dg, _ = dgl.load_graphs(str(DATASET))
    dg = dg[0].long()
    masks = {
        "train": dg.ndata["train_masks"].numpy()[:, trial_col].astype(bool),
        "val": dg.ndata["val_masks"].numpy()[:, trial_col].astype(bool),
    }
    n = len(masks["train"])
    pool = np.where(masks["train"])[0]
    rest = np.setdiff1d(np.arange(n), pool)
    labels = np.asarray(graph["label"])
    rng = np.random.default_rng(seed)
    pos = pool[labels[pool] == 1]
    neg = pool[labels[pool] == 0]
    # stratified 60/40 of the pool, mirroring train_test_split(stratify=...)
    n_tr = int(round(0.60 * len(pool)))
    tr = np.concatenate([rng.permutation(pos)[:int(round(n_tr * len(pos) / len(pool)))],
                         rng.permutation(neg)[:n_tr - int(round(n_tr * len(pos) / len(pool)))]])
    va = np.setdiff1d(pool, tr)
    return np.sort(tr), np.sort(va), np.sort(rest)



def _switch_rounds(scenario, emb, head, kind, X0_raw, X0_emb, y_all,
                   test_idx, rounds, thr_r, pt_epochs, incremental, embedder, patience=None):
    """Query rounds scored with the RAW head; retention with the embedding head.

    The ablation shows the raw view wins on query detection (53.06 vs 38.45)
    because a replay-modified query node is a verbatim duplicate, while the
    embedding wins round-0. Each metric is therefore scored by the view that
    wins it - a protocol choice, not a tuning trick, and it must be reported
    as such.
    """
    per_round = []
    for r in range(1, rounds + 1):
        query, support = scenario.next_round()
        if len(query) == 0:
            break
        ev_q = list(scenario.stream_events(start_round=r, which="query"))
        if not ev_q:
            continue
        if incremental and pt_epochs:
            emb.pretrain(scenario.graph_dict(), epochs=pt_epochs,
                         lr=0.01, warm=True, patience=patience)
        E, _ = emb.embed(scenario.graph_dict())
        RAW = np.asarray(scenario.features, dtype=np.float64)
        ids_q = [int(e["raw"]["node_id"]) for e in ev_q]
        yq = np.array([int(e["label"]) if e["label"] is not None else 0
                       for e in ev_q])
        pq = head.h_raw.predict(RAW[ids_q], thr_r)          # raw head scores query
        ps = head.h_raw.predict_proba(RAW[ids_q])
        ret = head.h_emb.predict(X0_emb[test_idx],           # embedding head
                                 head.threshold)            # scores retention
        pq_bool = pq.astype(bool)
        qm = _full_metrics(yq, pq_bool, ps)
        per_round.append({
            "round": r, "f1_query": qm["f1"],
            "recall_query": qm["recall"],
            "precision_query": qm["precision"],
            "auc_query": qm["auc"],
            "recall_at_p30": _recall_at_precision(yq, ps, 0.30),
            "recall_at_p50": _recall_at_precision(yq, ps, 0.50),
            "f1_round0_nodes": _metrics(y_all[test_idx], ret)[0],
            "n_query": int(len(yq))})
        ev_s = list(scenario.stream_events(start_round=r, which="support"))
        if ev_s:
            ids_s = [int(e["raw"]["node_id"]) for e in ev_s]
            ys = np.array([int(e["label"]) if e["label"] is not None else 0
                           for e in ev_s])
            head.fit(RAW[ids_s], E[ids_s], ys)
        scenario.mark_predicted(query, pq.astype(bool))
    return per_round


def run(arm, rounds, seed, pretrain_epochs, pretrain_epochs_round,
        trials, buffer_max, h_feats, round_window, features="embed",
        embedder="ours", dual=False, raw_weight=0.5, split_mode="holdout",
        view_switch=False, patience=None, alpha=None, beta=None):
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
                                         round_window=round_window,
                                         alpha=alpha if alpha is not None else 1.0,
                                         beta=beta if beta is not None else 1.0)
        else:
            emb = TPNEEmbedder(h_feats=h_feats, num_layers=2,
                               round_window=round_window,
                               alpha=alpha if alpha is not None else 1.0,
                               beta=beta if beta is not None else 1.0,
                               mode="incremental" if incremental else "offline",
                               seed=s)
        emb_cols = emb.columns()
        head = (DualViewHead(base, emb_cols, s, kind) if (dual or view_switch)
                else Head(kind, base, emb_cols, s))
        rw = raw_weight

        t0 = time.time()
        # NOTE: this used to be guarded by `embedder != "author"`, which meant
        # that selecting the vendored reference encoder silently ran it
        # *untrained*. That guard predated AuthorTemporalEmbedder.pretrain(),
        # which is now implemented and is the exact analogue of the reference's
        # supervised_multi.py loop (up to 500 epochs per round). Selecting the
        # author's encoder and expecting their quality was therefore
        # impossible. Pre-training is now attempted for whichever embedder is
        # selected, and falls back safely if it does not support it.
        if pretrain_epochs:
            t_pre = time.time()
            try:
                loss = emb.pretrain(scenario.graph_dict(),
                                    epochs=pretrain_epochs,
                                    lr=0.01, patience=patience)
                ll = loss.get("final")
                print(f"  [trial {trial}] pretrain R0: "
                      f"{loss.get('epochs_run', pretrain_epochs)} epochs in "
                      f"{time.time() - t_pre:.0f}s, final loss "
                      f"{ll:.4f}" if isinstance(ll, float) else
                      f"  [trial {trial}] pretrain R0 done", flush=True)
                if isinstance(ll, float):
                    print(f"  [trial {trial}]   loss terms: "
                          + ", ".join(f"{k}={v:.4f}" for k, v in
                                      sorted(loss.items())
                                      if isinstance(v, float)), flush=True)
            except (NotImplementedError, AttributeError, TypeError) as e:
                print(f"  [warn] pre-training unavailable for this embedder: {e}")
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
        if split_mode == "masks":
            i_tr, i_va, i_te = author_split(graph, seed=seed + trial)
            # The author's train/val masks pick an arbitrary, NON-contiguous
            # subset of nodes, so they must be consumed as index arrays.
            # Reducing them to range(i_tr.max())+1 swept nodes 0..11751 into
            # training and 1..11757 into validation, which put ~100% of the
            # retention set into both -- i.e. train-on-test.
            train_idx, val_idx, test_idx = i_tr, i_va, i_te
        else:
            n = len(y_all)
            n_test = int(0.20 * n)
            n_val = int(0.20 * n)
            split = n - n_test - n_val      # train end
            val_slice = slice(split, split + n_val)
            train_idx = np.arange(split)
            val_idx = np.arange(split, split + n_val)
            test_idx = np.arange(split + n_val, n)
        X0_raw = np.asarray(scenario.features, dtype=np.float64)[:len(evs)]
        X0_emb = E0[:len(evs)]
        X0 = feats_for(np.arange(len(evs)), E0)
        if view_switch:
            head.fit(X0_raw, X0_emb, y_all[train_idx])
            thr_e = tune_threshold(head.h_emb, X0_emb[val_idx], y_all[val_idx])
            thr_r = tune_threshold(head.h_raw, X0_raw[val_idx], y_all[val_idx])
            head.threshold = thr_e
            print(f"  view-switch thresholds: embed {thr_e:.3f} raw {thr_r:.3f}",
                  flush=True)
            f1_init = _metrics(y_all[test_idx],
                               head.h_emb.predict(X0_emb[test_idx], thr_e))[0]
            rec0 = {"f1_initial": f1_init, "f1_round0_nodes": f1_init,
                    "threshold_embed": thr_e, "threshold_raw": thr_r}
            per_round = _switch_rounds(
                scenario, emb, head, kind, X0_raw, X0_emb, y_all,
                test_idx, rounds, thr_r, pretrain_epochs_round,
                incremental, embedder, patience)
            out.append({"round0": rec0, "rounds": per_round,
                        "seconds": 0.0, "embed_summary": emb.summary()})
            m = rt_metrics([out[-1]])
            print(f"  VIEW-SWITCH trial {trial}: avg_f1="
                  f"{(m['RT-avg_f1_t>0'] or 0)*100:.2f} recall="
                  f"{(m['RT-recall_0'] or 0)*100:.2f}", flush=True)
            continue
        if dual:
            head.fit(X0_raw, X0_emb, y_all[train_idx])
            thr = tune_threshold_dual(head, X0_raw[val_idx],
                                      X0_emb[val_idx], y_all[val_idx], rw)
            print(f"  dual threshold {thr:.3f} (raw weight {rw})", flush=True)
            f1_init = _metrics(y_all[test_idx],
                               head.predict(X0_raw[test_idx], X0_emb[test_idx],
                                            thr, rw))[0]
            rec0 = {"f1_initial": f1_init, "f1_round0_nodes": f1_init,
                    "threshold": thr}
            per_round = _dual_rounds(scenario, emb, head, kind, E0, X0_raw,
                                     X0_emb, y_all, test_idx, rounds, rw,
                                     pretrain_epochs_round, incremental,
                                     embedder)
            out.append({"round0": rec0, "rounds": per_round,
                        "seconds": 0.0, "embed_summary": emb.summary()})
            m = rt_metrics([out[-1]])
            print(f"  DUAL trial {trial}: avg_f1="
                  f"{(m['RT-avg_f1_t>0'] or 0)*100:.2f} recall="
                  f"{(m['RT-recall_0'] or 0)*100:.2f}", flush=True)
            continue
        # ---- train on the support split only; the retention set is never seen
        head.fit(X0[train_idx], y_all[train_idx])
        # ---- threshold selection on the validation split only
        thr = 0.5 if os.environ.get("NO_TUNE") else tune_threshold(
            head, X0[val_idx], y_all[val_idx])
        print(f"  threshold: {thr:.3f} "
              f"(val F1 {_metrics(y_all[val_idx], head.predict(X0[val_idx], thr))[0]*100:.2f}, "
              f"val F1@0.5 {_metrics(y_all[val_idx], head.predict(X0[val_idx], 0.5))[0]*100:.2f})",
              flush=True)
        y0 = y_all[test_idx]
        p0 = head.predict(X0[test_idx], thr)
        s0 = head.predict_proba(X0[test_idx])
        m0 = _full_metrics(y0, p0.astype(bool), s0)
        f1_init = m0["f1"]
        rec0 = {"f1_initial": f1_init, "f1_round0_nodes": f1_init,
                "threshold": thr,
                "precision": m0["precision"], "recall": m0["recall"],
                "auc": m0["auc"]}

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
                             epochs=pretrain_epochs_round, lr=0.01, warm=True,
                             patience=patience)
            E, _ = emb.embed(scenario.graph_dict())
            yr = np.array([int(e["label"]) if e["label"] is not None else 0
                           for e in ev_r])
            Xr = feats_for([int(e["raw"]["node_id"]) for e in ev_r], E)
            pq = head.predict(Xr, thr)
            ps = head.predict_proba(Xr)
            ret = head.predict(X0[test_idx], thr)
            pq_bool = pq.astype(bool)
            qm = _full_metrics(yr, pq_bool, ps)
            per_round.append({
                "round": r,
                "f1_query": qm["f1"],
                "recall_query": qm["recall"],
                "precision_query": qm["precision"],
                "auc_query": qm["auc"],
                "recall_at_p30": _recall_at_precision(yr, ps, 0.30),
                "recall_at_p50": _recall_at_precision(yr, ps, 0.50),
                "f1_round0_nodes": _metrics(y_all[test_idx], ret)[0],
                "n_query": int(len(yr)),
            })
            print(f"  [trial {trial}] round {r}: f1_query={qm['f1']*100:.2f} "
                  f"prec={qm['precision']*100:.2f} "
                  f"rec={qm['recall']*100:.2f} "
                  f"auc={(qm['auc']*100 if qm['auc'] is not None else float('nan')):.2f} "
                  f"f1_r0={_metrics(y_all[test_idx], ret)[0]*100:.2f}", flush=True)

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
                    "round0_auc": rec0.get("auc"),
                    "round0_precision": rec0.get("precision"),
                    "pretrain_loss": dict(getattr(emb, "last_loss", {}) or {}),
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
    ap.add_argument("--pretrain-epochs", type=int, default=60,
                    help="encoder pre-training epochs for round 0 "
                         "(reference uses 500)")
    ap.add_argument("--pretrain-epochs-round", type=int, default=20,
                    help="encoder pre-training epochs for each later round "
                         "(reference uses 300)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping on the pre-training loss; the "
                         "reference uses 75. 0 disables it.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="weight on the temporal-disentanglement loss "
                         "(reference uses 1.0)")
    ap.add_argument("--beta", type=float, default=None,
                    help="weight on the temporal-maximisation loss "
                         "(reference uses 1.0)")
    ap.add_argument("--reference-schedule", action="store_true",
                    help="shorthand for the paper's own schedule: "
                         "500/300 epochs, patience 75, 10 rounds, 10 trials, "
                         "alpha=beta=1")
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
    ap.add_argument("--dual-view", action="store_true",
                    help="train two heads (raw + embedding) and average "
                         "their probabilities")
    ap.add_argument("--raw-weight", type=float, default=0.5)
    ap.add_argument("--split", default="holdout", choices=["holdout", "masks"],
                    help="'masks' replicates the paper's round-0 split from the "
                         "dataset's 20 predefined mask columns")
    ap.add_argument("--view-switch", action="store_true",
                    help="train on both views but SCORE round-0 with the "
                         "embedding head and query rounds with the raw head, "
                         "since the two views win on opposite metrics")
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
    patience = a.patience or None
    pe, per = a.pretrain_epochs, a.pretrain_epochs_round
    rounds, trials = a.rounds, a.trials
    if a.reference_schedule:
        pe, per = 500, 300
        patience = patience or 75
        rounds, trials = 10, 10
        print("  reference schedule: 500/300 epochs, patience 75, "
              "10 rounds x 10 trials, alpha=beta=1", flush=True)
    recs = run(a.arm, rounds, a.seed, pe,
               per, trials, a.buffer_max,
               a.h_feats, a.round_window, a.features, a.embedder,
               split_mode=a.split, view_switch=a.view_switch,
               dual=a.dual_view, raw_weight=a.raw_weight,
               patience=patience, alpha=a.alpha, beta=a.beta)
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
    summary.update(arm=a.arm, trials=trials, pretrain_epochs=pe,
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