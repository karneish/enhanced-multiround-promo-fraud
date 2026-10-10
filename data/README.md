# data/ — real datasets (never committed to the repo)

This folder is **gitignored** (`.gitignore` excludes `data/*.csv`, `data/*.zip`).
Download each file once and place it here; the backend auto-discovers them.

## IEEE-CIS Fraud Detection (real, labeled; replayed as a live stream)

Accepted layouts (any of these):

| File(s) | Notes |
|---|---|
| `train_transaction.csv` + `train_identity.csv` | Official two-file format (merges identity fields) |
| `final_fraud_dataset.csv` | Pre-merged single-file mirror (433 columns) |

Fetch helpers / instructions:

```
.venv\Scripts\python.exe scripts\fetch_ieee.py          # try kagglehub, else prints instructions
.venv\Scripts\python.exe scripts\fetch_ieee.py --merged # else place final_fraud_dataset.csv here
```

Permissions: the official dataset is under **Kaggle competition terms** —
accept them at https://www.kaggle.com/competitions/ieee-fraud-detection/data
and do **not** redistribute the CSVs (this is why the folder is gitignored).

## Scam-address list for the live Ethereum stream (optional)

`scam_addresses.csv` — known-bad Ethereum addresses that the label bridge
(`backend/labels.py`) matches against streamed transactions. Auto-detected
columns, e.g.:

```csv
Address,Label,Type
0x7f80299aae930a103254944a28db580c4d9a78ef,1,Phishing
0x47d9f... ,1,Ransomware
```

Sources: XBlock (Ethereum Phishing Transaction Network, Kaggle), Elliptic
labels, public abuse lists. Without the file the chain stream still works —
fraud then comes only from the human-review loop and the model's own scores.