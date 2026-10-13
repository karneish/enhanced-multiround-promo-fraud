#!/usr/bin/env python
"""
Author-vs-ours scoreboard for the multi-round adversarial promo fraud benchmark.

The author's numbers are their own five-trial control, reproduced from
`vendor/author-tpne/result/` with their `extract_metrics.py`:

    RT-f1_0 start 66.56 +/- 0.75     RT-avg_f1 57.09 +/- 2.33
    RT-recall_0    27.50 +/- 5.62    final_precision 31.01 +/- 4.16
    final_auc      64.00 +/- 5.63

Recall is reported by both sides, but recall alone is not comparable: it is
bought by lowering the decision threshold. This script therefore also reports
recall at a fixed precision floor, which is threshold-independent and is the
only fair way to state the recall comparison.

    python scripts/compare_report.py
    python scripts/compare_report.py reports/refsched_author_3t.json
"""

import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUTHOR = {
    "RT-f1_0_start": (66.56, 0.75),
    "RT-f1_0_end": (67.49, 2.24),
    "RT-f1_0_gap": (0.93, 1.58),
    "RT-avg_f1_t>0": (57.09, 2.33),
    "RT-recall_0": (27.50, 5.62),
    "final_precision": (31.01, 4.16),
    "final_auc": (64.00, 5.63),
}
AUTHOR_TRIALS = 5

METRIC_LABEL = {
    "RT-f1_0_start": "round-0 F1",
    "RT-f1_0_end": "round-0 F1, final",
    "RT-f1_0_gap": "round-0 F1 gap (forgetting)",
    "RT-avg_f1_t>0": "on-round F1",
    "RT-recall_0": "final recall",
}
HIGHER_IS_BETTER = {k: True for k in METRIC_LABEL}
HIGHER_IS_BETTER["RT-f1_0_gap"] = False      # smaller forgetting is better


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = 100 * sum(xs) / len(xs)
    s = 100 * st.stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def series(rec, key, last_only=False):
    """Per-trial values for a top-level metric, taken from the raw rounds."""
    out = []
    for t in rec.get("per_trial", []):
        rounds = t.get("rounds", [])
        if not rounds:
            continue
        if key == "RT-f1_0_start":
            out.append(t["round0"]["f1_initial"])
        elif key == "RT-f1_0_end":
            out.append(rounds[-1]["f1_round0_nodes"])
        elif key == "RT-avg_f1_t>0":
            out.append(sum(r["f1_query"] for r in rounds) / len(rounds))
        elif key == "RT-recall_0":
            out.append(rounds[-1]["recall_query"])
    return out


def extra(rec):
    """Precision / AUC / recall-at-precision over the final round."""
    prec, auc, rp30, rp50 = [], [], [], []
    for t in rec.get("per_trial", []):
        rounds = t.get("rounds", [])
        if not rounds:
            continue
        r = rounds[-1]
        if r.get("precision_query") is not None:
            prec.append(r["precision_query"])
        if r.get("auc_query") is not None:
            auc.append(r["auc_query"])
        if r.get("recall_at_p30") is not None:
            rp30.append(r["recall_at_p30"])
        if r.get("recall_at_p50") is not None:
            rp50.append(r["recall_at_p50"])
    return {
        "final_precision": mean_std(prec),
        "final_auc": mean_std(auc),
        "recall@p>=0.30": mean_std(rp30),
        "recall@p>=0.50": mean_std(rp50),
    }


def load(path):
    if os.path.isabs(path):
        return json.load(open(path))
    return json.load(open(os.path.join(ROOT, path)))


def block(name, path, rec):
    ours = {k: mean_std(series(rec, k)) for k in METRIC_LABEL}
    ex = extra(rec)
    print(f"\n{'='*84}\n{name}   ({os.path.basename(path)}, n={len(rec.get('per_trial', []))}, "
          f"rounds={rec.get('n_rounds')})\n{'='*84}")
    print(f"{'Metric':<26}{'Author':>17}{'Ours':>19}{'Delta':>10}{'  ':<4}")
    print(f"{'':<26}{'n=5':>17}{'n='+str(len(rec.get('per_trial', []))):>19}")
    print("-" * 84)
    wins = ties = losses = 0
    for k, label in METRIC_LABEL.items():
        am, asd = AUTHOR[k]
        om, osd = ours[k]
        if om is None:
            print(f"{label:<26}{am:>10.2f} ±{asd:<5.2f}{'n/a':>19}")
            continue
        d = om - am
        better = (d > 0) if HIGHER_IS_BETTER[k] else (d < 0)
        if abs(d) < 0.5:
            mark, ties = "~", ties + 1
        elif better:
            mark, wins = "WIN", wins + 1
        else:
            mark, losses = "LOSS", losses + 1
        print(f"{label:<26}{am:>10.2f} ±{asd:<5.2f}{om:>11.2f} ±{osd:<5.2f}"
              f"{d:>+10.2f}  {mark}")
    print("-" * 84)
    print("Not comparable across sides (author reports these, ours are new):")
    for k in ("final_precision", "final_auc", "recall@p>=0.30", "recall@p>=0.50"):
        am = AUTHOR.get(k)
        om = ex.get(k)
        astr = f"{am[0]:.2f} ±{am[1]:.2f}" if am else "not reported"
        ostr = f"{om[0]:.2f} ±{om[1]:.2f}" if om and om[0] is not None else "not recorded"
        print(f"  {k:<20} author: {astr:<16} ours: {ostr}")
    print(f"\n  headline metrics: {wins} win, {ties} tie, {losses} loss")
    return wins, ties, losses


def main():
    args = sys.argv[1:]
    if args:
        paths = args
    else:
        paths = sorted(glob.glob(os.path.join(ROOT, "reports", "*.json")))
        paths = [p for p in paths
                 if os.path.getmtime(p) > os.path.getmtime(__file__) - 86400 * 3]
    if not paths:
        print("no report files given and none recently modified")
        return 0
    for p in paths:
        try:
            rec = load(p)
        except Exception as e:                        # noqa: BLE001
            print(f"skip {p}: {e}")
            continue
        if "per_trial" not in rec:
            continue
        block(os.path.basename(p), p, rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())