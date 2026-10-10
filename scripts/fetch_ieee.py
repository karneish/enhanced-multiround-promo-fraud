"""Fetch the IEEE-CIS Fraud Detection dataset into data/.

One-shot fetches (if credentials/tools are available) then print placement
instructions. The dataset is gated by Kaggle competition terms, so nothing in
this script downloads silently when authentication is missing.

Usage:
    python scripts/fetch_ieee.py            # try kagglehub, then fall back to instructions
    python scripts/fetch_ieee.py --mirror   # the pre-merged mirror dataset (same columns)
"""
import argparse
import os
import shutil
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

KAGGLE_DATASET = "ieee-fraud-detection"
MIRROR_DATASET = "lnasiri007/ieeecis-fraud-detection"
MERGED_MIRROR = "neeraja3/fraud-detection-preprocessed-dataset-ieee-cis"


def _instructions():
    return f"""

 Download destinations (data dir: {DATA_DIR})

 Option A - official train_transaction.csv + train_identity.csv (preferred)
   1. Accept the rules at https://www.kaggle.com/competitions/ieee-fraud-detection/data
   2. Download the two CSVs (or use kagglehub / kaggle CLI -- see below)
   3. Place them as:
        data/train_transaction.csv
        data/train_identity.csv

 Option B - pre-merged single file (no identity split)
   data/final_fraud_dataset.csv   (433 columns, 946 MB from the neeraja3 mirror)

 Optional label lists for the live Ethereum stream:
   data/scam_addresses.csv   (columns: Address,Label[,Type]) XBlock/Elliptic scrape
"""


def _try_kagglehub(dataset: str, destination: str) -> bool:
    try:
        import kagglehub
    except ImportError:
        return False
    try:
        path = kagglehub.dataset_download(dataset)
    except Exception as exc:
        print(f"kagglehub failed for {dataset}: {exc}")
        return False
    for f in os.listdir(path):
        if f.endswith(".csv"):
            shutil.copy(os.path.join(path, f), os.path.join(destination, f))
            print(f"  copied {f}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mirror", action="store_true",
                        help="use the Kaggle mirror instead of the official dataset")
    parser.add_argument("--merged", action="store_true",
                        help="use the pre-processed single-file mirror")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    dataset = MIRROR_DATASET if args.mirror else KAGGLE_DATASET
    if args.merged:
        dataset = MERGED_MIRROR

    print(f"Target data dir: {DATA_DIR}")
    if _try_kagglehub(dataset, DATA_DIR):
        print("kagglehub completed.")
    else:
        print("kagglehub not available (pip install kagglehub, and a Kaggle account with the rules accepted).")  # noqa: E501
        print(_instructions())

    ok = any(fname in os.listdir(DATA_DIR) for fname in
             ("train_transaction.csv", "final_fraud_dataset.csv"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()