#!/usr/bin/env python
"""Extract the paper's three multi-round metrics from an author-run result CSV.

Implements the definitions printed in Section 6.1.4 of
"A multi-rounded adversarial scenario for graph-based promo fraud detection"
(Prasetya, Liu, Murata & Matono, Social Network Analysis and Mining):

  RT-f1_0        Initial F1        - how much is learned in round 0 and retained
  RT-avg_f1_{t>0}  On-round F1      - mean F1 over the rounds AFTER round 0
  RT-recall_0    Final recall      - recall measured on the final round's query set

The CSV written by the author's ``scripts/main.py`` holds one row per
(round, eval_type) with a ``trial`` column, so the aggregation below mirrors
``src/utils/nb_utils.py::get_metric_df`` but is standalone and prints a table
suitable for pasting into a comparison.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


def load_result(path):
    """Load combined_result.csv from a run directory, a config name, or the tree."""
    if os.path.isfile(path):
        return pd.read_csv(path)
    if os.path.isdir(path):
        direct = os.path.join(path, "combined_result.csv")
        if os.path.isfile(direct):
            return pd.read_csv(direct)
        hits = glob.glob(os.path.join(path, "**", "combined_result.csv"), recursive=True)
        if hits:
            return pd.read_csv(sorted(hits)[-1])
    raise SystemExit(f"no results found under {path!r}")


def round_metrics(df):
    """The three paper metrics, averaged over trials, per configuration group."""
    rows = []
    # One row per (trial, round, eval_type). Config identity comes from the
    # author's own config blocks, which main.py stamps onto every row. Metric
    # columns must NOT be part of the key or each row becomes its own group.
    drop = {"round", "eval_type", "time", "trial", "timestamp", "Unnamed: 0",
            "tp", "fp", "tn", "fn", "rec", "prec", "f1", "auc", "mf1",
            "recon_loss", "weighted_recon_loss", "disen_loss",
            "weighted_disen_loss", "temp_loss", "weighted_temp_loss",
            "final_loss", "predicted_pos", "predicted_pos_og",
            "prediction_speed", "num_nodes", "num_edges"}
    config_cols = [c for c in df.columns
                   if c not in drop and not c.startswith("Unnamed")]
    config_key = df[config_cols].astype(str).agg("|".join, axis=1)

    for key, grp in df.assign(_cfg=config_key).groupby("_cfg"):
        for trial, tgrp in grp.groupby("trial"):
            # RT-f1_0 (Equation 14): F1 on the ROUND-0 NODE SET, measured at the
            # start (round_0/test_set) and again at the end of every later round
            # (round_t/round_0_nodes). The gap start->end is how much of round 0
            # the model retained, i.e. how much it forgot.
            r0_start = tgrp[(tgrp["round"] == "round_0")
                            & (tgrp["eval_type"] == "test_set")]
            r0_retention = tgrp[tgrp["eval_type"] == "round_0_nodes"]
            if r0_start.empty or r0_retention.empty:
                continue
            f1_0_start = float(r0_start["f1"].iloc[0])
            f1_0_end = float(r0_retention["f1"].iloc[-1])

            # RT-avg_f1_{t>0} (Equation 15): mean F1 over the QUERY SET of every
            # round after round 0. The query set is reported as
            # eval_type == "<round>_nodes"; round_0_nodes is the *retention*
            # set and is deliberately excluded here.
            sub = tgrp[tgrp["eval_type"].astype(str)
                       == (tgrp["round"].astype(str) + "_nodes")]
            sub = sub[sub["round"] != "round_0"]
            if sub.empty:
                continue
            on_round = float(sub["f1"].mean())

            # RT-recall_0 (Equation 16): recall on the FINAL round's query set.
            final_round = sub["round"].max()
            fr = sub[sub["round"] == final_round]
            final_recall = float(fr["rec"].iloc[0])
            final_prec = float(fr["prec"].iloc[0])
            rows.append({
                "config": key[:60],
                "trial": trial,
                "RT-f1_0_start": f1_0_start * 100,
                "RT-f1_0_end": f1_0_end * 100,
                "RT-f1_0_gap": (f1_0_end - f1_0_start) * 100,
                "RT-avg_f1_t>0": on_round * 100,
                "RT-recall_0": final_recall * 100,
                "final_precision": final_prec * 100,
                "final_auc": float(fr["auc"].iloc[0]) * 100,
                "n_rounds": int(sub["round"].nunique()) + 1,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default="../result",
                    help="run directory, config name, or combined_result.csv")
    ap.add_argument("--csv", default=None, help="also write the summary here")
    args = ap.parse_args()

    df = load_result(args.path)
    m = round_metrics(df)
    if m.empty:
        raise SystemExit("no multi-round rows matched; check eval_type values")

    agg = m.groupby("config")[["RT-f1_0_start", "RT-f1_0_end", "RT-f1_0_gap",
                               "RT-avg_f1_t>0", "RT-recall_0",
                               "final_precision", "final_auc"]].agg(["mean", "std"])

    pd.set_option("display.width", 200)
    print(f"\ntrials per config: {m.groupby('config')['trial'].nunique().to_dict()}")
    print("\n=== per-trial ===")
    print(m.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n=== aggregated (mean +/- std, percent) ===")
    print(agg.to_string(float_format=lambda v: f"{v:.2f}"))

    if args.csv:
        m.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())