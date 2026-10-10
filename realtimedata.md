# REAL-TIME DATA INTEGRATION — AUTHORITATIVE REFERENCE

> **Status:** Design **locked and implemented** (single-app rebuild, 2026-09-06).
> **Purpose:** permanent reference for how the project ingests **real (non-synthetic),
> streamed** data, scores it in real time with the **3-model ensemble** (XGBoost +
> HistGradientBoosting + ExtraTrees + adaptive weighting), and applies
> **Allow / Review / Block** decisions. Future work should read this first, then
> `MODELS_REFERENCE.md` and `README.md`.
> **Last updated:** 2026-09-06.

---

## 1. The goal (one paragraph)

The old multi-app framework (research GNN framework, ADL, IFG, per-app dashboards)
was removed. The rebuild is a **single app** that streams **real data** and scores
it the moment it arrives, event-by-event:

- **IEEE-CIS Fraud Detection** — real labeled credit-card transactions, replayed
  at a configurable pace (real data meets real-time; labels are true `isFraud`).
- **Ethereum mainnet** — genuinely live transactions via a keyless public JSON-RPC
  gateway; ground truth from a known-scam address list (label bridge) plus the
  human-review labeling loop.
- **Webhook** — `POST /api/events/ingest` accepts any external producer in every mode.

Each source keeps its **own per-source 3-model ensemble** (dimension-agnostic —
only the feature builder differs). Models warm-start on stream cadence; decisions
and stats stream to the React dashboard over SSE.

---

## 2. Decisions (settled, do not revisit without reason)

| Decision | Choice | Notes |
|---|---|---|
| Combined approach | **Design A + B together** | *Handoff*: train/evaluate on the labeled replay, then switch to the live chain stream; *Concurrent*: both streams side-by-side, source-tagged; toggle via `live_mode` |
| Real labeled source | **IEEE-CIS** (Kaggle / Vesta) | Real `isFraud`; ~590k rows, ~3.5% positive; license-gated → CSVs live in `data/`, gitignored |
| Truly-live source | **Ethereum** live stream | Default `transport: rpc-poll` over the keyless Cloudflare public gateway; optional WebSocket mode with a paid/RPC provider |
| Chain | **Ethereum** | `eth_getBlockByNumber(blockTag='latest', fullTxs=true)` polling, deduplicated by block number/hash |
| Chain access | **Free WS/RPC provider** | Keyless default (Cloudflare); `chain.rpc_url` / `transport: ws` configurable |
| Delivery | **Replay + webhook + live chain** | All three produce the same canonical event |
| Detector | **Only 3 models** | XGBoost + HistGradientBoosting + ExtraTrees with adaptive (ADS-style) weighting — written from scratch in `backend/detector.py` |
| Decision tier | **Keep Allow / Review / Block** | Drives the review-budget labeling loop for unlabeled streams |
| Frontend | **React + Vite** | One dashboard for everything (`frontend/`) |
| Removed forever | DGL/GNN/TPNE/adversary/GAN/IFG/ADL app/main framework | None of it survives the rebuild; no synthetic data anywhere in live mode |

---

## 3. Architecture

```
 IEEE-CIS CSV ──► IeeeReplaySource   ─┐                    per-source model
 Ethereum RPC ─► EthereumSource     ─┼─► livesim consumer ─► ThreeModelEnsemble
 webhook POST ─► /api/events/ingest ─┘    (score-now)     (XGB + HistGB + ET, ADS weights)
                                                                    │
                 hummingbird: profile + source-native features      ▼
                 (features.py, computed before ingest)    Allow / Review / Block
                                                                    │
                                          review-budget label loop │ warm-start refits
                                                                    ▼
                                                 SSE feed ─► React dashboard
```

- **Canonical event** (`backend/events.py`): `{event_id, source, ts, account_id,
  counterparties, label?, amount?, nat, raw}` — identical shape for every source.
- **Features** (`backend/features.py`): `PROFILE_COLUMNS` (account history/
  velocity/network aggregates, source-independent) + source-native columns
  (`IEEE_NATIVE_COLUMNS`, `CHAIN_NATIVE_COLUMNS`, `WEBHOOK_NATIVE_COLUMNS`).
- **Detector** (`backend/detector.py`): per-source rolling labeled window,
  balanced down-sampling, warm-started XGBoost (booster resume) + HistGB
  (growing `max_iter`) + fresh-eta refit ExtraTrees; weights ∝ trailing-holdout F1.
- **Decision mapping:** `p ≥ block(0.70)` → Block; `p ≥ review(0.30)` → Review;
  else Allow. Known-fraud (`label==1`, replay native or scam-bridge hit) always
  Blocks.

---

## 4. IEEE-CIS dataset reference

| Property | Value |
|---|---|
| Source | Kaggle: **IEEE-CIS Fraud Detection** (Vesta, 2019 ICML workshop challenge) |
| Files | `train_transaction.csv` (~590,540 rows) (+ optional `train_identity.csv`), or pre-merged `final_fraud_dataset.csv` |
| Label | `isFraud` (0/1); ~3.5% positive |
| Key fields | `TransactionDT` (sec → stream order + age/time-of-day), `TransactionAmt`, `ProductCD`, `card1`–`card6`, `addr1/2`, `dist1/2`, `P/R_emaildomain`, `C1`–`C14`, `D1`–`D15`, `V1`–`V339`, identity `DeviceType`/`DeviceInfo` |
| Identity link | `TransactionID` joins transaction ↔ identity tables |
| Size | ~234 MB zipped; **license-gated (Kaggle terms)** → never committed |

### Feature mapping (documented approximations in `backend/sources/replay.py -> _event_from_row`)

| Category | Derivation |
|---|---|
| `age` / `since_last` / hour / dow | `TransactionDT` |
| `C*` counts → velocity | `c_sum/c_mean/c_max` aggregates |
| `D*` timedeltas → timing | `d_sum` aggregate |
| missing-signal indicator | `num_nan_v` (NaN count over `V1..V339`) |
| shared identity edges | `card*` → account id; `addr1/addr2`, email domains, `DeviceType/Info`, `ProductCD` → counterparties |
| amount | `TransactionAmt` (real), `log1p` |

> Labels and amounts are untouched real values; only concept aggregates are
> derived. The webhook and chain sources reuse `PROFILE_COLUMNS` untouched.

---

## 5. Files (final layout)

| Path | Responsibility |
|---|---|
| `backend/config.py` | defaults + optional `config.local.json` merge |
| `backend/events.py` | canonical event schema + validation |
| `backend/features.py` | `PROFILE_COLUMNS`, per-source native columns, `FeatureBuilder` + `ProfileTracker` |
| `backend/detector.py` | `ThreeModelEnsemble` — XGB/HistGB/ExtraTrees, warm-start, ADS weights, decide() |
| `backend/labels.py` | `ScamAddressMatcher` (address-list label bridge) |
| `backend/livesim.py` | `LiveSimulation` — producers/consumer, quadrature refits, review budget, SSE feed, status |
| `backend/app.py` | Flask API + SSE (`/api/live/*`, `/api/events/ingest`, `/api/stream`, serves built frontend) |
| `backend/sources/replay.py` | IEEE load/discovery + timed `stream()` |
| `backend/sources/ethereum.py` | `EthereumSource` rpc-poll / ws transports |
| `backend/sources/webhook.py` | webhook payload → canonical event |
| `frontend/` | React + Vite dashboard (Live feed, stats, controls, per-model weights) |
| `data/` | real datasets (gitignored) — `README.md` documents placement |
| `scripts/fetch_ieee.py` | IEEE-CIS fetch helper / instructions |
| `run.py` | single launcher (port 5051, threaded) |

---

## 6. Config surface

```
live_mode: replay-only | chain-only | handoff | concurrent   (default concurrent)
events_per_sec: 20          stream_slice: 100000
decision_thresholds: {review: 0.30, block: 0.70}
budget_pos: 10  budget_neg: 200        min_fit_labels: 120
retrain_interval_events: 500  retrain_interval_seconds: 60  buffer_max_labels: 3000
xgb_rounds/xgb_extra_rounds  hist_max_iter/hist_extra_iter  et_n_estimators
ieee: {data_dir, replay_skip_first}
chain: {transport: rpc-poll|ws, rpc_url, ws_url, poll_interval_seconds,
        max_txs_per_block, scam_list_path: data/scam_addresses.csv}
```
Overrides: `config.local.json` at repo root, or the `/api/live/start` request body.

---

## 7. How the model retrains (continual learning)

- Labeled events (replay native, scam-bridge matches, webhook labels) append to the
  source's rolling window (`buffer_max_labels`, balanced before fit).
- Refit when `retrain_interval_events` OR `retrain_interval_seconds` elapsed **and**
  the window contains both classes.
- XGBoost resumes its booster; HistGB grows `max_iter`; ExtraTrees refits.
- The refit runs on a background thread and swaps the new estimators in
  atomically, so scoring never blocks on a fit.
- Weights = trailing-holdout F1 across the available models, floored at
  `WEIGHT_FLOOR_FRACTION` (0.05) *of the best model's* F1, so the blend is
  scale-invariant; uniform fallback when every F1 is 0. On real data the three
  models score within ~2% of each other, so the weights come out near-uniform —
  see `MODELS_REFERENCE.md` §4.2.
- Cold start: fallback probability clamped **[0.10, 0.5]**; UI marks the source
  "cold start" until `min_fit_labels` accumulate. Unlabeled cold-start events are
  forced into the review band so the review loop can bootstrap the model.

---

## 8. Verification (success criteria — no synthetic claims)

- Replay: real rows arrive in `TransactionDT` order at `events_per_sec`; every row
  scored with decision; SSE updates per event; per-model weights/metrics visible in
  `/api/live/status`.
- Handoff: chain source starts automatically when the replay slice ends (no restart,
  models preserved).
- Concurrent: replay + chain events coexist, tagged by `source`.
- Chain: block numbers advance, transactions flow from the gateway, scam-bridge
  events are labeled 1 and Block, review budget decrements on unlabeled Reviews.
- Webhook: `POST /api/events/ingest` mirrors replay behaviour and returns the
  per-event `{prob, decision}` immediately.

---

## 9. Risks / caveats

- **IEEE-CIS download** requires accepted Kaggle terms (or the merged mirror);
  `scripts/fetch_ieee.py` handles fetch/instructions. CSVs are gitignored.
- **Anonymized fields** → concept aggregates are approximations; labels/amounts are
  true values.
- **Free public gateways** are intended for low-volume use; the source probes
  `chain.rpc_fallback_urls` in order and re-probes after repeated failures, but
  for sustained volume set a paid `chain.rpc_url` or `transport: "ws"` with your
  own endpoint.
- **Chain labels are sparse** → the chain model may sit in cold start until the
  review loop/scam list yields enough labels; decisions still function via the
  clamped fallback probability.
- **Dashboard metrics are optimistic** → the per-model F1 shown live is computed
  on a slice of the window the models were just fitted on (and the warm-started
  boosters have seen those rows), so it reads ~0.97-1.00. It is a liveness
  signal, not a result. Quote `scripts/evaluate.py` (clean temporal holdout:
  ensemble F1 0.156, ROC-AUC 0.731) instead — see `MODELS_REFERENCE.md` §6b.
- **Retrain cost** is bounded by `buffer_max_labels` (3,000) and balanced
  down-sampling, keeping each warm-start refit sub-second at these data scales.