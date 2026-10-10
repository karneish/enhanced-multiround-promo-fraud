#!/usr/bin/env python
"""Honest head-to-head: our restored pipeline vs the author's published code.

Both sides run the *same* adversarial multi-round scenario on *tolokers_bid*
under the paper's protocol, so the numbers are comparable.

    our side      backend.embedder.TPNEEmbedder  (offline | incremental)
                   + backend.sources.adversarial.AdversarialScenario
    author side   vendor/author-tpne (their own main.py, run separately)

Usage
    # run our side only (writes a JSON the comparator consumes)
    python scripts/compare_tpne.py --mode ours --mode-flag offline --rounds 3
    python scripts/compare_tpne.py --mode ours --mode-flag incremental --rounds 3
    # compare against the author run directory
    python scripts/compare_tpne.py --mode compare \\
        --ours reports/ours_offline.json \\
        --author vendor/author-tpne/result/config_author_control
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "vendor" / "author-tpne" / "dataset" / "tolokers_bid"


def load_tolokers():
    """Load tolokers_bid through DGL and return plain numpy/scipy arrays."""
    import dgl  # imported lazily: only needed in the TPNE environment
    import scipy.sparse as sp
    graphs, _ = dgl.load_graphs(str(DATASET))
    g = graphs[0].long()
    feats = g.ndata["feature"].float().numpy().astype(np.float64)
    label = g.ndata["label"].numpy().astype(int)
    src, dst = g.edges()
    edge_index = np.vstack([src.numpy(), dst.numpy()])
    return {"features": feats, "edge_index": edge_index, "label": label}


def rt_metrics(trials):
    """The paper's three metrics from our own per-round records.

    Mirrors vendor/author-tpne/scripts/extract_metrics.py so the two sides are
    computed identically. ``trials`` is a list of
    ``{"round0": {...}, "rounds": [...]}``.
    """
    f1_0_start, f1_0_end, query_f1, final_recall = [], [], [], []
    for t in trials:
        r0 = t.get("round0", {})
        if "f1_initial" in r0:
            f1_0_start.append(r0["f1_initial"])
        if "f1_round0_nodes" in r0:
            f1_0_end.append(r0["f1_round0_nodes"])
        rounds = t.get("rounds", [])
        query_f1.extend(r["f1_query"] for r in rounds if "f1_query" in r)
        if rounds:
            final_recall.append(rounds[-1]["recall_query"])
            # the final retention value is the last round's round-0 score
            f1_0_end.append(rounds[-1]["f1_round0_nodes"])

    def mean(xs):
        return float(np.mean(xs)) if xs else None

    start, end = mean(f1_0_start), mean(f1_0_end)
    return {
        "RT-f1_0_start": start,
        "RT-f1_0_end": end,
        "RT-f1_0_gap": (end - start) if (start is not None
                                         and end is not None) else None,
        "RT-avg_f1_t>0": mean(query_f1),
        "RT-recall_0": (final_recall[-1] if final_recall else None),
        "n_rounds": max([len(t.get("rounds", [])) for t in trials], default=0),
    }


def run_ours(rounds, mode_flag, seed, trials, pretrain_epochs=0):
    """Run our restored pipeline under the adversarial scenario."""
    from backend.embedder import TPNEEmbedder
    from backend.sources.adversarial import AdversarialScenario
    from backend.features import columns_for, FeatureBuilder
    from backend.detector import ThreeModelEnsemble
    from backend.config import DEFAULT_CONFIG
    from sklearn.metrics import f1_score, recall_score, precision_score

    graph = load_tolokers()
    cfg = dict(DEFAULT_CONFIG)
    cfg["seed"] = seed
    # match the author's budgets: 5% of pos/neg per round
    npos = int((graph["label"] == 1).sum())
    nneg = int((graph["label"] == 0).sum())
    cfg.update({
        "round_new_pos": int(0.05 * npos),
        "round_new_neg": int(0.05 * nneg),
        "min_fit_labels": 40,
        "retrain_interval_events": 200,
    })
    scenario_cfg = {
        "seed": seed, "adver_choose": "greedy", "adver_mod": "relative",
        "augment": "reage", "round_window": 7,
        "round_new_pos": cfg["round_new_pos"],
        "round_new_neg": cfg["round_new_neg"],
    }

    results = []
    for trial in range(trials):
        scenario = AdversarialScenario(graph, {**scenario_cfg,
                                               "seed": seed + trial})
        embedder = TPNEEmbedder(mode=mode_flag, seed=seed + trial)
        cols = columns_for("adver")
        # the embedding columns are appended to the behavioural columns, so the
        # ensemble actually sees TPNE output (this is the whole point)
        emb_cols = embedder.columns()
        model_cols = cols + emb_cols
        ens = ThreeModelEnsemble("adver", model_cols, cfg)
        builder = FeatureBuilder()
        per_round = []

        # round 0: real labelled graph, embedded once up front
        if pretrain_epochs:
            embedder.pretrain(scenario.graph_dict(), epochs=pretrain_epochs,
                              lr=0.01)
        emb0, loss0 = embedder.embed(scenario.graph_dict())
        events = list(scenario.stream_events())
        X, y = [], []
        for ev in events:
            vec = builder.build(ev)
            builder.absorb(ev)
            node = int(ev["raw"]["node_id"])
            row = [vec[c] for c in cols] + \
                [float(emb0[node][j]) if node < len(emb0) else 0.0
                 for j in range(len(emb_cols))]
            X.append(row)
            y.append(ev.get("label") if ev.get("label") is not None else 0)
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        split = int(0.6 * len(y))
        Xtr, ytr = X[:split], y[:split]
        yte = y[split:]
        for row, lab in zip(Xtr, ytr):
            ens.add_labeled(row, int(lab))
        ens.fit()

        pred = ens.predict_proba_batch(X[split:])
        pred_bin = (np.asarray(pred) >= 0.5).astype(int)
        rec = {
            "round": 0,
            "f1_initial": float(f1_score(yte, pred_bin, zero_division=0)),
            "f1_round0_nodes": float(f1_score(y[split:], pred_bin,
                                             zero_division=0)),
        }

        for r in range(1, rounds + 1):
            query = scenario.next_round()
            if len(query) == 0:
                break
            ev_list = list(scenario.stream_events(start_round=r))
            # re-embed on the current graph; this is where offline vs
            # incremental differs
            if pretrain_epochs:
                embedder.pretrain(scenario.graph_dict(),
                                  epochs=pretrain_epochs, lr=0.01,
                                  warm=(mode_flag == "incremental"))
            emb, loss = embedder.embed(scenario.graph_dict())
            builder2 = FeatureBuilder()
            Xq, yq = [], []
            for ev in ev_list:
                vec = builder2.build(ev)
                builder2.absorb(ev)
                node = int(ev["raw"]["node_id"])
                row = [vec[c] for c in cols] + \
                    [float(emb[node][j]) if node < len(emb) else 0.0
                     for j in range(len(emb_cols))]
                Xq.append(row)
                yq.append(ev.get("label") if ev.get("label") is not None else 0)
            if not Xq:
                continue
            Xq, yq = np.asarray(Xq, dtype=float), np.asarray(yq, dtype=int)
            pq = (np.asarray(ens.predict_proba_batch(Xq)) >= 0.5).astype(int)
            # retention of the round-0 nodes is re-scored every round
            ret_pred = (np.asarray(ens.predict_proba_batch(X[split:]))
                        >= 0.5).astype(int)
            per_round.append({
                "round": r,
                "f1_query": float(f1_score(yq, pq, zero_division=0)),
                "recall_query": float(recall_score(yq, pq, zero_division=0)),
                "f1_round0_nodes": float(
                    f1_score(y[split:], ret_pred, zero_division=0)),
                "embed_loss": {k: float(v) for k, v in loss.items()},
            })
            # feed decisions back so greedy targets misses
            scenario.mark_predicted(query, pq.astype(bool))
            for row, lab in zip(Xq, yq):
                ens.add_labeled(row, int(lab))
            ens.fit()

        results.append({"round0": rec, "rounds": per_round,
                        "embed_summary": embedder.summary()})

    # aggregate
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["ours", "compare"], default="ours")
    ap.add_argument("--mode-flag", choices=["offline", "incremental"],
                    default="offline")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--ours", default=None)
    ap.add_argument("--author", default=None)
    ap.add_argument("--pretrain-epochs", type=int, default=0,
                    help="gradient-based pre-training epochs per round "
                         "(0 = inference-only, the earlier un-trained arm)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "ours":
        recs = run_ours(args.rounds, args.mode_flag, args.seed, args.trials,
                        args.pretrain_epochs)
        summary = rt_metrics(recs)
        summary["per_trial"] = recs
        summary["mode_flag"] = args.mode_flag
        summary["pretrain_epochs"] = args.pretrain_epochs
        out = args.out or (
        f"reports/ours_{args.mode_flag}"
        f"{'_trained' if args.pretrain_epochs else ''}.json")
        os.makedirs(Path(out).parent, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
        print(json.dumps({k: v for k, v in summary.items()
                          if k != "per_trial"}, indent=2, default=float))
        print(f"wrote {out}")
        return 0

    # compare
    with open(args.ours) as fh:
        ours = json.load(fh)
    author = json.load(open(args.author)) if args.author and \
        args.author.endswith(".json") else None
    if author is None and args.author:
        sys.path.insert(0, str(ROOT / "vendor" / "author-tpne" / "scripts"))
        from extract_metrics import load_result, round_metrics
        author = round_metrics(load_result(args.author))
        author = author.mean(numeric_only=True).to_dict()

    print(f"\n{'metric':<16}{'author (TPNE-XGB)':>22}{'ours':>22}{'delta':>14}")
    print("-" * 74)
    for k in ["RT-f1_0_start", "RT-f1_0_end", "RT-f1_0_gap",
              "RT-avg_f1_t>0", "RT-recall_0"]:
        a = author.get(k)
        o = ours.get(k)
        if isinstance(a, (int, float)) and isinstance(o, (int, float)):
            print(f"{k:<16}{a:>22.4f}{o:>22.4f}{o - a:>14.4f}")
        else:
            print(f"{k:<16}{str(a):>22}{str(o):>22}{'-':>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main())