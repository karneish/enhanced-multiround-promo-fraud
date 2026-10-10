# Report 1 — Abstract (Demo)

**Project:** Real-Time Adaptive 3-Model Fraud Detector for Multi-Round Promo/Reward Abuse
**Repo:** `multiround-promo-fraud`
**Status:** Live demo build (single-app rebuild)
**Report date:** 2026-09-22

---

## 1. Project abstract (the idea)

The project is a **real-time (streaming) fraud detector** that scores every
incoming transaction *event-by-event* through an **adaptive 3-model ensemble**
written **from scratch**, converts the score into **Allow / Review / Block**
decisions, and **continually learns** (warm-start retraining) from streamed
labels plus a **human-review budget loop**. The "multi-round / promo" angle is
that fraudsters attack promotional rewards over **multiple bursts / rounds** —
high account event velocity, counterparty (card / email / device / address)
reuse, and volume spikes across accounts — which the system captures with
**rolling per-account behavior profiles** rather than a static batch snapshot.

The core architectural promise: **real data meets real-time**.
- Replay of a **real, labeled** dataset (IEEE-CIS Fraud Detection, ~590k real
  credit-card transactions, real `isFraud` labels) is treated as a timed live
  stream (the "real dataset + real-time" bridge), OR
- A **genuinely live** Ethereum mainnet stream (keyless public RPC polling),
  where ground truth is supplied by a **known-scam address label bridge** plus
  the human-review loop, OR
- A **webhook** source for arbitrary external producers.

Every source emits the **same canonical event**, and each source keeps its
**own per-source 3-model ensemble** that warm-starts/refits as labeled data
accumulates. All sources coexist (concurrent, handoff, replay-only,
chain-only) and stream into a React/Vite dashboard over Server-Sent Events.

The build intentionally dropped the earlier research framework's heavy pieces
(GNN, adversarial detection, TPNE, IFG, ADL, 6-model roster, per-app
dashboards) in favor of a **single streaming app** with exactly three tree
models (XGBoost + HistGradientBoosting + ExtraTrees) blended by adaptive
(F1-proportional) weights.

---

## 2. What is implemented so far (build status)

This is a **working live simulation / demo** — not a finished research
artifact. Delivered:

1. **A working end-to-end pipeline**: three producers (IEEE replay, live
   Ethereum, webhook) → canonical event normalization → streaming feature
   building → per-source 3-model adaptive ensemble → Allow/Review/Block
   decisions → review-budget labeling loop → warm-start refits → SSE feed → a
   live React dashboard (event log, node graph, per-model weights, decisions,
   review budget).

2. **Written-from-scratch machine learning** (`backend/detector.py`):
   - **XGBoost** (gradient-boosted trees), **warm-started** by resuming the
     underlying booster (`xgb_model=`) on each refit.
   - **HistGradientBoosting** (sklearn), warm-started by growing `max_iter`
     (`warm_start=True`).
   - **ExtraTrees** (maximally-randomized bagging), refit on the rolling
     balanced labeled window (no warm-start in sklearn).
   - **Adaptive ADS-style weighting**: weight per model = trailing holdout F1,
     floored at 5% *of the best model's* F1 (scale-invariant); blend = weighted
     blend of fraud probabilities. Uniform fallback when every F1 is 0.
   - **Cold-start fallback** probability clamped [0.10, 0.5] with a label-prior.

3. **Streaming feature engineering** (`backend/features.py`):
   - A `PROFILE_COLUMNS` block (account-history aggregates computed identically
     for every source) + source-native numeric columns.
   - `ProfileTracker` — rolling per-account histories bounded to rolling
     windows (events in trailing 60s / 1h / 24h, amounts, counterparty reuse).
   - Per-source column lists via `columns_for(source)`.

4. **Label bridge** (`backend/labels.py`) — `ScamAddressMatcher` auto-detects
   address/label columns from a scam-address CSV and labels chain events whose
   account or counterparty is a known-bad address.

5. **Live simulation orchestrator** (`backend/livesim.py`) — `LiveSimulation`
   with producer threads → bounded queue → single consumer thread, timing
   controls (`events_per_sec`, `stream_slice`), per-source ensembles,
   review-budget budget loop (`budget_pos` / `budget_neg`), node-graph
   building (accounts/counterparties + reuse edges), status snapshot, and SSE
   feed.

6. **Flask API + SSE** (`backend/app.py`):
   - `/health`, `/api/schema`, `/api/live/start`, `/api/live/stop`,
     `/api/live/status`, `/api/events/ingest` (webhook), `/api/stream` (SSE),
     `/` serving the built React dashboard.

7. **Live sources** (`backend/sources/`):
   - `replay.py` — IEEE-CIS timed replay (`IeeeReplaySource`), auto-discovers
     `train_transaction.csv` / `train_identity.csv` or a pre-merged
     `final_fraud_dataset.csv`; deterministic labeled sample fallback when no
     real CSVs exist.
   - `ethereum.py` — `EthereumSource`: keyless RPC polling with fallback URL
     failover and `eth_subscribe newHeads` WebSocket transport; labels events
     touching known-scam addresses via `ScamAddressMatcher`.
   - `webhook.py` — external producer ingestion with payload parsing.

8. **Config system** (`backend/config.py`) — built-in defaults merged with an
   optional `config.local.json` at repo root, plus per-request overrides in
   `/api/live/start`.

9. **Frontend** (`frontend/`) — **React 18 + Vite 5** dashboard (`App.jsx` +
   components: StatusBar, SessionControls, SourceCards, ModelPanel,
   DecisionMix, LiveFeed, NodeGraph, EventLog, EventIntake, ConfigInspector),
   SSE-powered live feed, graph polling, schema/status visibility.

10. **Launcher** (`run.py`) — dependency check + threaded Flask server
    (default `http://127.0.0.1:5051`).

11. **Utilities**: `scripts/fetch_ieee.py` (dataset fetch instructions /
    kagglehub helper), `data/README.md`, `realtimedata.md`,
    `MODELS_REFERENCE.md` (model hyper-parameters + empirical F1 rationale).

---

## 3. Technical stack

**Backend (Python 3, `.venv`)**
- **Flask** + **flask_cors** — HTTP API + SSE streaming
- **SSE (Server-Sent Events)** feed, per-connection cursor (`FeedCursor`) so
  each client receives every record exactly once
- **Threading / queue / collections** — producer threads, bounded queue, ring
  buffers, rolling profiles
- **numpy / pandas** — feature vectors, balanced windows, dataset loading
- **scikit-learn** — `HistGradientBoostingClassifier` (warm_start),
  `ExtraTreesClassifier`, `HistGradientBoostingClassifier` (the three used),
  holdout splitting, F1/precision/recall metrics
- **xgboost** — gradient-boosted trees (warm-started booster resume)
- **requests** — Ethereum JSON-RPC polling (keyless public gateways)
- **websocket-client** — optional WebSocket transport (`eth_subscribe`)
- Flask serves the built React dashboard from `frontend/dist`

**Frontend (React + Vite)**
- React 18.3, ReactDOM, Vite 5, `@vitejs/plugin-react`
- Components (from `App.jsx` imports): `StatusBar.jsx`, `SessionControls.jsx`,
  `SourceCards.jsx`, `ModelPanel.jsx`, `DecisionMix.jsx`, `LiveFeed.jsx`,
  `NodeGraph.jsx`, `EventLog.jsx`, `EventIntake.jsx`, `ConfigInspector.jsx`
- `api.js` — schema/status/graph/live start-stop/ingest calls; SSE stream
  consumed in `App.jsx` (message + status events, dedup by identity)
- Vite dev proxy → backend (`/api`, `/health`)
- Built output at `frontend/dist`, `frontend/package.json` (vite build)

---

## 4. Components (detailed)

### 4.1 Canonical event schema (`backend/events.py`)
Every source produces the same canonical event dict:
```
event_id, source (ieee|chain|webhook), ts (epoch seconds), account_id,
counterparties (list), label (0|1|None), amount, nat (source-native numerics),
raw (original fields)
```
`SOURCES = ("ieee", "chain", "webhook")`, required keys
`("event_id", "source", "ts", "account_id")`.

### 4.2 Feature building (`backend/features.py`)
- `PROFILE_COLUMNS` (source-independent): age_hours, hours_since_last,
  n_events, events_60s/1h/24h, total/avg/max amount, amt_ratio,
  n_unique_cp, cp_reuse_max.
- `IEEE_NATIVE_COLUMNS`: amt, log_amt, hour, dow, dist1, dist2, c_sum/c_mean/
  c_max (C1..C14 velocity aggregates), d_sum (D1..D15 time aggregates),
  num_nan_v, product_cd_code, has_email, has_r_email.
- `CHAIN_NATIVE_COLUMNS`: value_eth, log_value_eth, gas_gwei, tx_index,
  scam_match.
- `WEBHOOK_NATIVE_COLUMNS`: txn_amt, n_attrs, label_hint.
- `ProfileTracker` builds features **before** the event is absorbed (velocity
  profile BEFORE ingest). `FeatureBuilder` merges profile + native columns per
  source.

### 4.3 Detector (`backend/detector.py`) — `ThreeModelEnsemble`
Per-source rolling labeled windows (balanced down-sampling), warm-start
refits:
- XGBoost warm-started by resuming booster (`xgb_model=`)
- HistGradientBoosting warm-started by growing `max_iter`
- ExtraTrees refit on the balanced rolling labeled buffer
- Decision mapping (`decision_from_prob`): prob >= block(0.70) → block;
  prob >= review(0.30) → review; else allow. Known-fraud (label==1) always
  blocks.
- Cold start: prior clamped [0.10, 0.5]; review spending governed by
  `budget_pos`/`budget_neg`.

### 4.4 Label bridge (`backend/labels.py`) — `ScamAddressMatcher`
Auto-detects address+label CSV columns; `is_scam()` and `label_for(ev)` label
chain/webhook events whose account or counterparty is a known-bad address.
Without the CSV, streams are unlabeled and rely on the review loop.

### 4.5 Live simulation (`backend/livesim.py`) — `LiveSimulation`
- `live_mode` (config / start body): replay-only | chain-only | handoff |
  concurrent (default concurrent).
- Producers run in threads (`_run_replay`, `_run_chain`, handoff thread when
  mode calls for it); events pushed into a bounded queue; one consumer thread
  scores + ingests; feedback/refit events pushed to the SSE feed.
- Review budget loop: unlabeled "review" decisions spend review budget
  (positive/negative buckets by amount median); labeled events warm the
  ensemble w/ refit cadence (`retrain_interval_*`).
- Node-graph building (accounts + counterparties as nodes, sharing edges with
  weights) exposed via graph API; graph revision counter.
- `status()` exposes counters, decisions, review budget, per-model weights,
  metrics, ensemble metrics, source states, chain state, scam-label bridge.

### 4.6 API server (`backend/app.py`)
Endpoints: `/health`, `/api/schema` (config + data availability + event
schema), `/api/live/start|stop|status`, `/api/events/ingest` (webhook),
`/api/stream` (SSE), `/` (serves `frontend/dist`), `/assets/*`. SSE nodes:
messages, status, keepalive, source_status, source_error, refit, refit_error,
process updates, graph revision.

### 4.7 Sources (`backend/sources/`)
- `replay.py` — `IeeeReplaySource`: loads real IEEE-CIS CSVs (auto-discovered
  layout), converts rows → canonical events (`_event_from_row`), streams in
  `TransactionDT` order at `events_per_sec`; deterministic labeled sample
  fallback when no real data.
- `ethereum.py` — `EthereumSource`: `rpc-poll` (keyless HTTP polling with
  fallback URL failover) or `ws` transport; emits block transactions as chains;
  scam-label via `ScamAddressMatcher`.
- `webhook.py` — accepts canonical or light payloads → canonical event.

### 4.8 Frontend dashboard (`frontend/src/App.jsx` + components)
Live dashboard: schema/status fetch, SSE connect (EventSource `/api/stream`),
dedup feed (identity-based cursor), event log + live feed, graph polling
(every 2.5s), start/stop controls (`handleStart`/`handleStop`), event intake,
config inspector, model panel, per-source source cards, decision mix.

---

## 5. Datasets used and where they are collected

### 5.1 IEEE-CIS Fraud Detection (replay source — real, labeled)
- **What:** Real credit-card fraud transactions from the **IEEE-CIS Fraud
  Detection challenge** (Kaggle, Vesta). ~590,540 rows, ~3.5% fraud rate, real
  `isFraud` labels, fields: `TransactionDT`, `TransactionAmt`, `ProductCD`,
  `card1..card6`, `addr1/2`, `dist1/2`, `C1..C14`, `D1..D15`, `V1..V339`,
  email domains, identity file.
- **Collected where:** downloaded manually / via `scripts/fetch_ieee.py`
  (kagglehub or instructions) into `data/` — `data/train_transaction.csv` +
  `data/train_identity.csv`, or pre-merged `data/final_fraud_dataset.csv`.
  **License-gated by Kaggle terms; CSVs are gitignored.** Optional optimized
  parse into `train_optimized` (described in docs). Sample fallback generates a
  deterministic labeled synthetic sample when no real CSVs are present.

### 5.2 Ethereum mainnet (chain source — genuinely live, unlabeled)
- **What:** Live Ethereum mainnet block/transaction stream over a **keyless
  public RPC gateway** (default `https://ethereum-rpc.publicnode.com`, with
  fallback RPC URLs: cloudflare-eth, blastapi, flashbots, drpc). Optional
  WebSocket transport with a keyed provider.
- **Collected where:** streamed live from the chain via `eth_getBlockByNumber`
  polling / `eth_subscribe newHeads`; **no stored dataset** (stream-only).
- **Labels:** unlabeled by default; ground truth supplied by a known-scam
  address list (`data/scam_addresses.csv`) via the label bridge + human-review
  loop.

### 5.3 Webhook (external producer)
- **What:** Any external service posts canonical-ish events (`account_id`,
  amount, counterparties, label, native fields) to `/api/events/ingest`.
- **Collected where:** from the caller (no bundled dataset).

### 5.4 Known-scam address labels (for chain streams)
- CSV with address/label columns (auto-detected) at
  `data/scam_addresses.csv` — known-bad Ethereum addresses (e.g., from
  scam lists / public abuse lists). Without it, chain labels come only from
  the review loop.

---

## 6. Empirical evidence / rationale (from `MODELS_REFERENCE.md`)

From an ablation (on `tolokers`, ADAPTIVE config, mixed embedder + DFEAT,
special graph, 2 rounds; historical numbers preserved as rationale):

| Model | mean F1 | best F1 |
|---|---|---|
| HistGradientBoosting | 0.6638 | 0.6868 |
| XGBoost | 0.6520 | 0.6668 |
| ExtraTrees | 0.6367 | 0.6494 |
| RandomForest | 0.6280 | 0.6416 |
| LightGBM | 0.6026 | 0.6937 |
| LogisticRegression | 0.5065 | 0.5198 |

Whole-ensemble 3-model F1 ≈ 0.637 (round_0) / ≈ 0.664 (round_1), matching
the larger 6-model baseline (~0.626 / ~0.628) at roughly half the training
cost — the rationale for the canonical trio.

### 6b. Clean temporal holdout on IEEE-CIS (the numbers we report)

The table above comes from a *different* dataset (`tolokers`) and a retired
framework; it justifies the trio, not this system's accuracy. For this pipeline
the reportable numbers come from `scripts/evaluate.py`: fit on one strictly
temporal window, score a disjoint holdout no model ever saw (no warm-start
continuation, no resampling across the boundary). Measured 2026-09-28 on real
`data/train_transaction.csv`, threshold 0.50, 26 features:

| Run | Split | Ensemble F1 | Precision | Recall | Accuracy | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 40k events | 30k train / 10k holdout | 0.156 | 0.098 | 0.376 | 0.891 | 0.099 | 0.731 |
| 150k events | 100k train / 50k holdout | 0.137 | 0.094 | 0.254 | 0.910 | 0.076 | 0.716 |

Per-model at 40k: ExtraTrees F1 0.167, XGBoost 0.144, HistGB 0.131
(ROC-AUC 0.736 / 0.718 / 0.699).

Stated plainly:

- Absolute F1 is **0.13-0.17**, not the ~0.99 the dashboard displays. The
  dashboard number is a *streaming holdout*: it scores the deployed models on a
  slice of the window they were just fitted on, and the warm-started boosters
  have already seen those rows. It is a liveness signal and is labelled as such;
  it is not a result.
- The base rate is ~2.7%. PR-AUC of 0.076-0.099 is roughly 3-4x that, and
  ROC-AUC ~0.72, so there is real ranking signal; F1 at 0.50 is precision-bound,
  which is what a 2.7%-prevalence problem looks like.
- The adaptive blend is **not** better than its best single member on F1. It wins
  on accuracy and ROC-AUC, and the three models land within ~2% of each other on
  the trailing holdout, so the weights come out near-uniform (0.334/0.329/0.337).
  With comparable models there is little for a weighted average to recover, and
  a wide spread would be fitting holdout noise.
- F1 is **not monotonic in window size** (0.156 at 40k, 0.137 at 150k): the
  longer window covers a period with a lower fraud rate and more regime drift,
  and precision tracks the base rate. Both runs are reported rather than the best.
- These figures are **replay ground truth only**. Chain and webhook events are
  unlabeled by construction, so no honest F1 exists for them yet; the claim there
  is "learns from human/LLM review labels", which is exactly what the review
  loop is for.

---

## 7. Config surface (defaults in `backend/config.py`)

```jsonc
port: 5051, host: 127.0.0.1
live_mode: concurrent          # replay-only | chain-only | handoff | concurrent
events_per_sec: 20             stream_slice: 100000
score_batch_max: 256           score_batch_window_ms: 25   retain_event_records: 200
decision_thresholds: {review: 0.30, block: 0.70}
budget_pos: 10  budget_neg: 200
review: {queue_max: 200, expire_seconds: 3600}   # the /api/review queue
min_fit_labels: 120  retrain_interval_events: 500  (per source)
buffer_max_labels: 3000
xgb_rounds/xgb_extra_rounds  hist_max_iter/hist_extra_iter  et_n_estimators
ieee: {data_dir, replay_skip_first(s), sample_fallback, sample_size}
chain: {transport: rpc-poll|ws, rpc_url, rpc_fallback_urls, poll_interval_seconds,
        max_txs_per_block, scam_list_path: data/scam_addresses.csv}
graph_node_cap: 2000
seed: 42
```

Overrides: `config.local.json` at repo root, or per-request in
`POST /api/live/start` body (highest precedence).

---

## 8. Streaming/continual-learning loop (as implemented)

1. Producers (replay / chain / webhook) push canonical events into a bounded
   queue.
2. A single consumer thread drains that queue in **micro-batches** (`score_batch_max`
   rows, or a `score_batch_window_ms` timeout) and scores them in one vectorised
   call per source: features are built BEFORE absorption and absorbed in arrival
   order, probability maps to Allow/Review/Block, then the graph/feed/label side
   effects run. Batching took the replay path from ~108 to >7000 rows/s; the
   decision for a given event is identical to single-row scoring
   (`tests/test_livesim.py`).
3. **Human-review budget loop:** unlabeled events that land in `review` draw
   from the review budget (pos/neg buckets by amount median); the candidate is
   queued *with the feature vector that was scored*, and only the label that
   comes back via `POST /api/review` (dashboard click or LLM script) trains the
   model. Unlabeled cold-start streams are forced into the review band, otherwise
   the loop could never bootstrap.
4. **Retrain cadence:** when `retrain_interval_events` OR
   `retrain_interval_seconds` elapse AND the window has both classes, a
   background thread warm-starts (XGBoost booster resume, HistGB max_iter
   growth, ExtraTrees refit) on a balanced rolling window, recomputes F1-
   proportional weights on a trailing holdout, and swaps the new estimators in
   atomically — scoring never blocks on a fit. A `refit` event goes to the feed.
5. **Cold start:** probability clamped [0.10, 0.5]; known-fraud always blocks.
6. All events/decisions/weights/metrics stream to the dashboard via SSE and
   the graph node/edge feed.

---

## 9. How to run (current demo)

```bash
# backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
# real IEEE-CIS data (license-gated; optional; sample fallback otherwise)
.venv\Scripts\python.exe scripts\fetch_ieee.py
# dashboard
cd frontend && npm install && npm run build && cd ..
# run
.venv\Scripts\python.exe run.py
# Dashboard: http://127.0.0.1:5051   (Vite dev: http://127.0.0.1:5173)
```

---

## 10. Files / docs map

| Path | Responsibility |
|---|---|
| `backend/config.py` | defaults + `config.local.json` merge |
| `backend/events.py` | canonical event schema + normalization + validation |
| `backend/features.py` | profile/native feature columns + FeatureBuilder + ProfileTracker |
| `backend/detector.py` | adaptive 3-model ensemble (XGBoost/HistGB/ExtraTrees), warm-start refits, ADS weights |
| `backend/labels.py` | ScamAddressMatcher (scam-address label bridge) |
| `backend/livesim.py` | LiveSimulation orchestrator (producers/queue/review budget/SSE/graph) |
| `backend/app.py` | Flask API + SSE + static frontend serving |
| `backend/sources/replay.py` | IEEE-CIS timed replay source |
| `backend/sources/ethereum.py` | live Ethereum RPC/WS source |
| `backend/sources/webhook.py` | webhook/event-ingest source |
| `backend/review.py` | bounded, expiring human/LLM review queue |
| `run.py` | single launcher (port 5051) |
| `frontend/` | React + Vite dashboard (incl. the review-queue panel) |
| `data/` | real datasets (gitignored) + README |
| `scripts/fetch_ieee.py` | IEEE-CIS fetch helper / instructions |
| `scripts/evaluate.py` | clean temporal holdout evaluation (reportable numbers) |
| `tests/` | 137-test `unittest` regression suite |
| `realtimedata.md` | real-time data integration design reference |
| `MODELS_REFERENCE.md` | exact model hyper-parameters + weighting spec + results |
| `requirements.txt` | backend (flask, flask_cors, numpy, pandas, scikit-learn, xgboost, requests, websocket-client) |

---

## 11. Notes / caveats

- IEEE-CIS dataset is **license-gated** (Kaggle competition terms); the CSVs
  must never be committed (they live only in gitignored `data/`).
- The Ethereum stream uses **keyless public gateways** with ordered failover
  (`chain.rpc_fallback_urls`), probed on the producer thread so
  `POST /api/live/start` returns immediately; swap `chain.rpc_url` /
  `transport: ws` for a paid RPC provider at higher volume.
- Raw chain transactions are unlabeled by nature; ground truth comes from the
  scam-address bridge and the human-review labeling loop.
- ExtraTrees has no warm-start → it refits from scratch on the bounded
  balanced window (fast at the small scale).
- The current app replaced graph-embedding inputs with behavioural streaming
  features, so absolute F1 numbers shift — but the model ordering / trio
  rationale transfers directly (see `MODELS_REFERENCE.md`).
- **The dashboard's live F1 is not a result.** It is a streaming holdout on
  data the deployed models were just fitted on (~0.97-1.00). The reportable
  numbers are the clean temporal holdout in §6b: ensemble F1 0.156, PR-AUC
  0.099, ROC-AUC 0.731 at a ~2.7% base rate.

---

**Author:** Karneish P — [GitHub](https://github.com/karneish)
**License:** MIT
**Folder:** `Research paper/report-1 abstract (demo).md`
