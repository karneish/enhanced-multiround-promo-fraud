# enhanced-multiround-promo-fraud

Multi-round promotion-fraud detection framework: an **intelligent fraud
generator**, an **adaptive defensive layer**, and a **self-supervised
TPNE (temporal neighborhood-preserving embedding) + adaptive multi-model
ensemble detector** — all orchestrated behind one unified dashboard.

## What's inside

| Component | Path | What it does |
|---|---|---|
| Research framework | `multiround-promo-fraud/` | Multi-round graph experiment harness: TPNE embedders (`TemporalEmbedder`, `TemporalMixedEmbedder`), benchmark GNNs, XGBoost/GBDT + tree ensembles, budgeted ground-truth rounds, adversarial strategies |
| Adaptive Defensive Layer | `adaptive-defensive-layer/` | Live simulator + defense strategies (rule scoring, anomalies) over the account graph |
| Intelligent Fraud Generator | `intelligent-fraud-generator/` | Adversary that creates evolving fraud rings (fake identity, referral farming, device spray, VPN hops) |
| Adaptive Ensemble Detector | `adaptive-ensemble-detector/` | Standalone detector: self-supervised embeddings -> XGBoost + HistGradientBoosting + ExtraTrees -> **Adaptive Detector Score (ADS)** weighted blend -> fraud probability |
| Unified dashboard | `frontend/` | React + Vite dashboard that talks to all backends |
| Launchers | `run.py`, `run_all.py` | Start every backend + the UI on one command |

## Core detection idea

1. **TPNE embedder** — a self-supervised GNN learns a per-account embedding that
   preserves graph-neighborhood structure while separating *who the account is
   now* (`h_current`) from *how it drifts over rounds* (`h_temp`). No labels needed.
2. **Three base models** — XGBoost, HistGradientBoosting and ExtraTrees train on
   those embeddings every round.
3. **Adaptive Detector Score (ADS)** — each model is scored on `F1 + Recall +
   Stability + Historical (EMA)`, and the **weighted ensemble fraud probability**
   is the ADS-weighted average of the three models' outputs. Weights are recomputed
   every round, so the blend adapts as the adversary evolves.

## Getting started

```bash
# 1. create the environment (torch + dgl + xgboost + sklearn + flask)
cd multiround-promo-fraud
python -m venv .venv
.venv\Scripts\pip install torch dgl xgboost scikit-learn flask flask-cors

# 2. launch every backend + the UI
cd ..
python run.py            # or: python run_all.py (iframe dashboard hub)
```

Dashboard: http://127.0.0.1:3000 — backends: Main 5051, ADL 5052, IFG 5053, Ensemble 5054.

## Author

**Karneish P** — [GitHub](https://github.com/karneish)

## License

MIT License (see `multiround-promo-fraud/LICENSE`).