# ENHANCED MULTI-ROUND PROMO FRAUD DETECTION — FULL PROJECT SUMMARY

## 1. One-paragraph overview

This project detects **promotional fraud** (fake accounts abusing "refer-a-friend, earn-a-reward" programs) using **graph-based node classification inside a multi-round adversarial game**. In each round, a fraud **detector** is trained on the current graph, then an **adversary** attacks by injecting new fraud that adapts to whatever escaped detection; the detector is then retrained on the growing graph, and so on. It is an enhanced fork of the paper *"A multi-rounded adversarial scenario for graph-based promo fraud detection"* (Prasetya, Liu, Murata, Matono — Social Network Analysis and Mining, Springer 2025). The two main **enhancements** added on top of the original paper are: **(A) an "Intelligent Fraud Generator"** — a GAN/probabilistic adversary that *learns from missed fraud and synthesizes new, diverse, ring-structured fraud* instead of naively copying the old fraud — and **(B) an interactive Flask web dashboard** for configuring, launching, and live-monitoring experiments. There is also a second, standalone **demo app** (`intelligent-fraud-generator/`) that runs the same ideas on a fully synthetic marketplace world so the whole loop can be watched in a browser without a GPU.

---

## 2. The two folders in this project

```
multiround-promo-fraud\
├── multiround-promo-fraud\    ← MAIN research framework (DGL + PyTorch + XGBoost)
└── intelligent-fraud-generator\  ← COMPANION demo app (Flask, synthetic world)
```

---

## 3. Core concept: the multi-round adversarial scenario

Real-world fraud is a cat-and-mouse game, not a one-shot problem. The framework simulates this:

- The data is a **graph** where **nodes = accounts** and **edges = relationships** (referrals, shared device, shared IP). Node labels are binary: `0 = genuine`, `1 = fraud`.
- There is an **initial graph** (round 0) and each subsequent round the graph **grows** with new nodes created by the adversary (new fraud) plus randomly duplicated genuine nodes (new negatives).
- Every node carries metadata tracked across rounds:
  - `creation_round` (which round the node appeared),
  - `predicted` (has the model ever flagged it),
  - `true_predicted_round` (the round the model *correctly* flagged it),
  - `train_mask` / `val_mask` / `test_mask` (data split).
- Each round the detector is retrained on the nodes it *knows* about (train mask), then predicts on the *entire* graph. The adversary then attacks, new nodes are appended, and the loop repeats for `round_num` rounds.
- Metrics (Recall, Precision, macro-F1, AUC, confusion matrix) are recorded for the whole graph, per-round nodes, and per-seed after every round, so you can see the detector degrade/recover as the adversary evolves.

---

## 4. Main framework: repository layout (the `multiround-promo-fraud/` folder)

```
multiround-promo-fraud/
├── setup.py                    # pip package metadata (tpne-xgb v1.1), dependency list
├── run_example.sh              # HPC (UGE/Univa) launcher that calls main.py
├── scripts/
│   ├── main.py                 # ENTRY POINT: reads a config JSON, runs experiments
│   ├── config_example.json     # full example search grid
│   ├── config_cpu.json         # CPU smoke test (REPLAY/naive adversary)
│   └── config_intelligent.json # CPU smoke test (INTELLIGENT generator)
├── src/
│   ├── experiment/supervised_multi.py  # MultiroundExperiment: the round loop core
│   ├── models/                 # detector backbones (GNNs + boosters + EmbedBoost)
│   ├── meta_strategies/        # augmentation (NONE/RANDOM/REAGE) + prediction add-ons
│   ├── adversary/
│   │   ├── choose/             # which fraud nodes to attack (RANDOM/GREEDY/OGREEDY)
│   │   └── modify/             # how to generate next-round fraud (REPLAY/PERTURB/MIXING/INTELLIGENT)
│   └── utils/                  # config defaults (utils_const.py), helpers (utils_func.py)
├── dashboard/                  # Flask app (app.py) + static/ + templates/
├── notebook/                   # Jupyter examples (run + data processing)
├── dataset/tolokers_bid        # the hosted DGL graph dataset
├── checkpoint/                 # temp model checkpoints written during runs
└── result/                     # ALL experiment outputs (CSVs + meta.txt)
```

---

## 5. Experiment execution flow (`scripts/main.py`)

1. Loads a config JSON (`-c <name>` → `scripts/<name>.json`).
2. Loads the DGL dataset from `dataset/<dset>`; normalizes labels/features.
3. **Auto-adjusts budgets**: `round_new_pos = 5% of positives`, `round_new_neg = 5% of negatives`; sets `round_budget=0`, and augment ratios = `(class_share × 0.125)`.
4. Builds the **Cartesian product** of `EXP_DICT` (every combination of hyper-parameters) and, for each combination, runs `TRIAL_NUM` repeated trials.
5. Each trial instantiates `MultiroundExperiment` and runs its round loop (`exp.one_round_node(r)` for `r in range(round_num)`).
6. If a trial hits `FAILURE_LIMIT` consecutive failures (model got stuck / no data), the run aborts.
7. Results are saved as DataFrames → CSV under `result/<config_name>/<timestamp>/`, plus `meta.txt` containing every config value used.

---

## 6. The round loop — `MultiroundExperiment` (src/experiment/supervised_multi.py)

The heart of the system. Each round (`one_round_node`) does, in order:

1. **`init_round`** — set current round number.
2. **`assign_train_graph`** (ADVER variant):
   - Round 0: split the initial graph into a train-graph (used for training) and a test pool (held out). If a separate pre-training graph is provided (`LIST_TRAIN_DSET`), use it.
   - Round ≥ 1: the graph used for evaluation last round *becomes* the training graph this round (no re-splitting; incremental).
3. **`split_train_test_adver`** — build train/val masks from a pool = original nodes (`creation_round < 0`) + all previously predicted nodes (TP∪FP from earlier rounds) + budgeted ground-truth nodes. Stratified `train_test_split` with `ratio_train=0.6`.
4. **`model_round_train`** — train the detector:
   - Optionally resets the model every round (`round_reset_model`).
   - Deep-copies the graph, sets it on the model, applies **augmentation** (REAGE/RANDOM/NONE) which adds synthetic training nodes.
   - Computes class-imbalance weight `ce_weight = (#neg)/(#pos)` in the train mask.
   - For GNN models: trains by epochs with early stopping + stuck-detection, checkpointing the best model (by best val-F1) and restoring it at the end. For `XGB`/`XGB-SP`: a single fit call on the embeddings.
5. **`model_round_predict`** — set the full graph, predict probabilities, choose threshold via `get_best_f1` (grid search over 0.05–0.95 maximizing macro-F1 on val), binarize, and optionally OR in the **prediction add-on** (rule-based flags). Produces TP/FP/TN/FN node masks.
6. **Evaluate** (`one_round_node_eval` + `final_eval_adver`) — REC/PRE/MF1/AUC + TP/FP/TN/FN for: entire graph, train, val, test, each round's nodes, current seeds, previous seeds. Also tracks `predicted_pos` (fraud left un-predicted) and `prediction_speed` (how many rounds on average before fraud is caught).
7. **`adversary_round_generate`** — the attack:
   - **Chooser** picks `round_new_pos` seed fraud nodes.
   - **Modifier** transforms seeds into the next round's fraud (this is where the Intelligent Generator lives).
   - The current round's `missed_probs` (model probabilities on the seeds) are passed to the modifier — this is the feedback signal the Intelligent Generator learns from.
   - Also generates negatives by duplicating random genuine nodes.
   - New nodes/edges are appended to the graph with `add_generated_nodes`.
8. **`close_round_adver`** — pick a **budgeted ground-truth set** (`round_budget_pos`/`round_budget_neg`): a small random subset of unseen fraud + new negatives that the detector will be "told" about next round (simulates a human reviewing a limited number of alerts).

If the model gets stuck (no meaningful TP/FP or TN/FN) for `stuck_stopping` epochs, the round returns "failed" and the trial restarts.

---

## 7. Detector models (src/models/)

- **Benchmark GNNs**: `GCN`, `GCNII`, `GraphSAGE`, `GIN`, `GAT` (src/models/benchmarks_supervised/simple.py) and `BWGNN`, `GHRN` (spectral.py).
- **Boosters**: `GraphBoost` (GNN-embedding → XGBoost), plus non-parametric backbones `GIN_noparam`, `RoundGIN_noparam`, `SplitRoundGIN_noparam`.
- **The proposed hybrid `XGB-SP`** (`EmbedBoost`, src/models/proposed_supervised/mixed.py):
  - A **GNN embedder** produces node embeddings *self-supervised* (no labels needed) with a loss that preserves neighborhood distance structure (non-parametric vs parametric distances; `ndist` or `ndot` variants) and optionally *temporal* disentanglement.
  - Three embedders exist:
    - `VanillaEmbedder` — plain multi-layer GIN/GraphConv embedding.
    - `TemporalEmbedder` — embeds each node across a sliding window of `round_window` rounds using the `age` attribute; produces a "current" embedding `h_current` and a "temporal drift" embedding `h_temp`; trains with reconstruction loss + disentanglement loss (correlation between current and temporal) + temporal-maximization loss, weighted by `alpha`/`beta`.
    - `TemporalMixedEmbedder` — like Temporal but adds a learned attention gate that blends current vs temporal components.
  - The embeddings are fed to a **trained XGBoost** classifier (`xgb.train`, `binary:logistic`, `scale_pos_weight`, hist tree method), warmed-up across rounds via `xgb_model=self.predictor` (continual learning).
  - The XGBoost predictor predicts fraud probability; the threshold is chosen to maximize macro-F1.

---

## 8. Adversary components (src/adversary/)

### 8.1 Choosers (`choose/`) — WHICH fraud to attack with next
Base class `BaseAdversarialChoice` provides `random_node_seeds` (samples from a priority pool, falls back to random fraud if the pool is too small) and `duplicate_nodes` (copies node+edge features, remaps to new IDs).
- `RANDOM` — any fraud nodes.
- `GREEDY` — prioritizes fraud nodes **not yet predicted** by the detector (`predicted == False`).
- `OGREEDY` — same, but only *original* fraud (`creation_round < 1`).

### 8.2 Modifiers (`modify/`) — HOW to generate next-round fraud
Base class `BaseAdversarialMod.modify_seeds(graph, node_data, edge_data, seed_ids, modified_ids, **kwargs)`.

The **original paper's** (naive) modifiers:
- `REPLAY` (ReplayMod) — returns the copied seeds unchanged → next round is a **duplicate** of the missed fraud.
- `PERTURB-ABS` (AbsolutePerturbMod) — adds a fixed-magnitude randomized noise to features; rewires edges by deleting/adding a connection budget distributed over relation types.
- `PERTURB-REL` (RelativePerturbMod) — adds noise proportional to feature std; rewires edges proportional to each node's degree.
- `MIXING` (MixingMod) — shuffles a fraction of feature columns and edge destinations *between* seed copies.

**The enhancement: `INTELLIGENT` (IntelligentMod)** — see next section.

---

## 9. The Intelligent Fraud Generator (THE main contribution)

File: `src/adversary/modify/intelligent_mod.py`. It replaces copy-based fraud generation with an adaptive generative attacker. It maintains state across rounds inside the modifier object (warm-started, carried round-to-round).

Per round, given the missed-fraud seeds (fraud that escaped detection) and the model's probabilities on them:

1. **ANALYZE** — builds a rolling **`_FraudProfile`** (bounded to `adver_gen_round_window` rounds, default 5):
   - feature pool (the successful fraud's feature vectors),
   - degree pool (their connection degrees),
   - a set of **familiar target nodes** (neighbors of successful fraud — the victims it keeps attacking),
   - mean/std/min/max feature stats.
2. **LEARN / EVOLVE** — two modes:
   - **GAN mode** (`adver_gen_type='GAN'`): trains a small generator MLP (`_GenMLP`: noise → features, sigmoid output) vs discriminator MLP (`_DiscMLP`) with standard minimax BCE loss for `adver_gen_epochs` epochs. The weights persist across rounds, so the generator keeps drifting toward fraud the current detector is bad at.
   - **PROB mode** (`adver_gen_type='PROB'`): probabilistic resampling — 50% of dimensions drawn as (random seed row + Gaussian drift), 50% as (pool mean + drift), clamped to [0,1].
3. **GENERATE** — samples `n` brand-new feature vectors (new devices, amounts, timing, etc.), applies `scale_to_range` so outputs live inside the observed fraud feature range, and optionally adds extra diversity via `adver_gen_feat_coef`.
4. **BUILD STRUCTURE** (`_build_structure`) — new connection patterns:
   - With probability `adver_gen_ring_ratio`: connect each new node to sibling new nodes → **fraud ring / referral chain** (degree follows seed degree × `adver_gen_conn_coef`).
   - Otherwise: rewire onto **familiar targets** (previously attacked genuine/fraud nodes).
5. **LOG** — writes `gen_*` diagnostics (below).

### The `gen_*` diagnostics (how to verify it works)
Written into each round's log and the result CSV:
- `gen_type` (GAN/PROB), `gen_seeds` (# missed fraud learned from),
- `gen_feat_div` — mean pairwise distance between generated features (**0.0 = identical copies**),
- `gen_feat_shift` — mean distance of generated features vs the seeds (**0.0 = exact replay**),
- `gen_new_edges`, `gen_ext_edges` (to external/familiar nodes), `gen_ring_edges` (ring/chains), `gen_ring_ratio`,
- `gen_missed_conf` — the detector's confidence that the seeds were genuine (should *rise* as the attacker learns to fool it),
- `gen_gan_g_loss` / `gen_gan_d_loss` (GAN training losses, if GAN mode).

**Why it beats the naive generator:** no duplicates (detector must learn new patterns), the attacker adapts to what works, richer supervision for retraining, and everything is observable through the logged diagnostics.

---

## 10. Meta-strategies (src/meta_strategies/)

### 10.1 Augmentation (`augment.py`) — how training data is enriched per round
- `NONE` — nothing.
- `RANDOM` (RandomReplaySampling) — duplicates train-masked positive/negative nodes in `augment_round_split` passes, aging them back across rounds.
- `REAGE` (ReAge) — does NOT duplicate; it **randomizes the `creation_round` (age)** of round-0 training nodes into recent rounds, forcing the temporal embedder to treat old nodes as if they appeared in multiple rounds. (Used in the smoke-test configs.)

### 10.2 Prediction add-ons (`prediction_addon.py`) — extra flags OR-ed into the model prediction
- `NONE` — nothing.
- `FTHR` (FeatureDistThreshold) — DBSCAN on recent 1-hop-neighborhood features; flags dense spam clusters.
- `AFTHR` (AggFeatureDistThreshold) — same but on *learned embeddings* instead of raw features.
- `DEGREE` (DegreeActivityThreshold) — flags recent nodes connected to top-percentile high-degree nodes.
- `DFEAT` = FTHR OR/AND DEGREE; `DAFEAT` = AFTHR OR/AND DEGREE (`addon_internal_agg` picks OR/AND).

---

## 11. Config system

All experiment settings are JSON files in `scripts/`. Top-level keys:
- `TRIAL_NUM` — repetitions per config combination.
- `FAILURE_LIMIT` — consecutive failures allowed before aborting.
- `EXPERIMENT_DESC` — human-readable description (goes to `meta.txt`).
- `LIST_DSET` — datasets (e.g. `["tolokers_bid"]`).
- `LIST_TRAIN_DSET` — optional pre-training graph per dataset (`"NONE"` to skip).
- `EXP_DICT` — the **Cartesian search grid** of hyper-parameters.

Every key in `EXP_DICT` is applied onto five default config dicts defined in `src/utils/utils_const.py`:
`DEFAULT_MAIN_CONFIG`, `DEFAULT_TRAIN_CONFIG`, `DEFAULT_MODEL_CONFIG`, `DEFAULT_STRAT_CONFIG`, `DEFAULT_ADVER_CONFIG`.

Key options (full lists in `utils_const.py`):
- Models: `GCN, GCNII, GraphSAGE, GIN, GAT, BWGNN, GHRN, XGB, XGB-SP`.
- Augmentation: `NONE, RANDOM, REAGE`.
- Chooser: `RANDOM, GREEDY, OGREEDY`.
- Modifier: `REPLAY, PERTURB-ABS, PERTURB-REL, MIXING, INTELLIGENT`.
- Prediction add-on: `NONE, FTHR, AFTHR, DEGREE, DFEAT, DAFEAT`.
- Intelligent-generator keys: `adver_gen_type` (GAN/PROB), `adver_gen_epochs`, `adver_gen_noise_dim`, `adver_gen_hidden`, `adver_gen_feat_coef`, `adver_gen_conn_coef`, `adver_gen_ring_ratio`, `adver_gen_round_window`.
- Training keys: `num_epoch`, `num_round_epoch`, `early_stopping`, `stuck_stopping`, `learning_rate`, `ratio_*`.
- Model keys: `h_feats`, `num_layers`, `round_window`, `temporal_agg`, `loss_type` (`ndist`/`ndot`), `norm_name`, `alpha`, `beta`.

---

## 12. Outputs / results (what "the result" looks like)

Everything is written under `result/<config_name>/<timestamp>/`:
- `meta.txt` — the exact configs used (main/train/model/strat/adver + EXP_DICT), i.e., full reproducibility record.
- `<dataset>-<grid-values>-E.csv` — one file per EXP_DICT combination; rows are per-round evaluations. Columns: `round`, `eval_type` (e.g. `entire_graph`, `train_set`, `val_set`, `test_set`, `round_N_nodes`, `seed_current_pred`, `seed_prev_pred`, `val_set_best`), `time`, `rec`, `prec`, `f1`, `auc`, `tp`, `fp`, `tn`, `fn`, plus the round log columns (`predicted_pos`, `prediction_speed`, training losses, `gen_*` diagnostics when INTELLIGENT), plus every config key as a column.
- `combined_result.csv` — all combinations concatenated.

Sample names seen in this repo: `tolokers_bid-cpu-ADVER-2-XGBSP-False-tempor-64-2-7-3-3-3-ndist-layer-weight-...-GREEDY-INTELL-GAN-50-10-05-05-...-E.csv` (the suffix is the sanitized EXP_DICT combination).

---

## 13. The web dashboard (multiround-promo-fraud/dashboard/)

Flask app on **http://127.0.0.1:5050** (`python app.py`). Features:
- **Dataset inspection** — `/api/datasets` returns node/edge/label-count stats; `/api/datasets/<dset>/graph?n=180` returns a sampled subgraph (BFS from high-degree seeds, spring layout) with node label/degree/position for rendering.
- **Schema endpoint** — `/api/schema` exposes all models, augmentations, choosers, modifiers, add-ons and defaults, so the UI is always in sync with the code.
- **Launch experiments** — POST `/api/run` writes a `scripts/run_<timestamp>.json` and starts `main.py -c <name>` as a **subprocess**.
- **Live streaming** — `/api/run/<id>/stream` is **Server-Sent Events (SSE)**; a reader thread tails stdout and regex-parses it (`parse_progress_line`) into structured events: trial start, round start, per-round `Best Val` metrics, eval lines, failures, elapsed time, best F1. The browser updates in real time.
- **Stop runs** — `/api/run/<id>/stop` (terminates the process, `taskkill /T /F` on Windows).
- **Browse results** — `/api/experiments` lists past runs, `/meta` shows `meta.txt`, and CSV endpoints return columns+rows as JSON for in-browser tables.

---

## 14. The companion demo app (intelligent-fraud-generator/)

A self-contained, GPU-free re-implementation for demonstrating the concept live. Launched with `run.bat` (or `python run.py`) → **http://127.0.0.1:5050**.

### 14.1 Synthetic world (`backend/world.py`)
Models a referral-reward marketplace:
- **Genuine accounts** (label 0) — normal users with healthy attributes and a few real referrals.
- **Fraud accounts** (label 1) — drawn from **5 initial strategy templates**: `fake_identity`, `referral_farming`, `device_spray`, `vpn_hop`, `quiet_sampler`. Each template = a target feature profile (`means`, `noise`) + structural preferences (`ring_affinity`, `device_spray`, `ip_reuse`).
- 11 **intrinsic behaviors** per account: age, email_disposable, phone_verified, device_fresh, ip_proxy, loc_entropy, login_night, amount_mean, amount_std, txn_count, txn_freq (all in [0,1]).
- 3 edge types: **referral** (account→account), **device** (account→device), **ip** (account→ip). All seeded for reproducibility.

### 14.2 Features (`backend/features.py`)
Builds an (n×17) matrix: the 11 intrinsics + 6 graph signals — referral_count, degree, clustering, shared_device, shared_ip, and `fraud_neighbor_ratio` (computed from **known/supervised labels only**, never the hidden truth → no leakage).

### 14.3 Detector (`backend/detector.py`)
A thin XGBoost wrapper: class-balanced with `scale_pos_weight`, threshold chosen by grid-search maximizing macro-F1 on a validation split, sklearn-style `train`/`predict`. `evaluate()` returns rec/prec/f1/auc/tp/fp/tn/fn/threshold.

### 14.4 Simulation engine (`backend/simulation.py`)
The round loop (mirror of the main framework):
1. Build world (round 0: genuine + initial fraud; a fraction `supervised_ratio` of labels are revealed).
2. **Retrain detector** on recently created accounts (`forget_window` — so the detector goes stale as the attacker evolves).
3. Predict everything; compute metrics (overall + fresh accounts from this round).
4. **Collect missed fraud** (supervised fraud with prob < threshold) → feed to generator's `FraudProfile`.
5. **Generator produces next round's fraud** + fresh genuine users → inject into world.
6. **Reveal a small budget** of new-round ground truth (`budget_pos`/`budget_neg`).
7. Repeat; every round's metrics + generator stats are stored and streamed.

### 14.5 Generator engine (`backend/generator/engine.py`)
Two interchangeable engines:
- `ReplayGenerator` — **baseline**: exact copies of missed fraud (`gen_feat_div = 0`, `gen_feat_shift = 0`). Used for comparison.
- `IntelligentFraudGenerator` — **the add-on**: learns a GAN (`gan.py`, small MLP generator/discriminator, minimax BCE) or probabilistic model (`sampler.py`) over the missed-fraud features, samples new behavior rows, adds diversity, then **mutates** (`mutators.py`) each row into a strategy (`parent+tags` like `referral_farming+vpn+new_device`) and **builds structure** (`build_structure`): rings/referral chains among new nodes + attachments to familiar victims. Falls back to mutated initial templates when no fraud has been missed yet.
- `FraudProfile` (`profile.py`) — rolling window of missed fraud (features, strategy names, confidences, familiar victim set) — the generator's memory.

### 14.6 API (`backend/app.py`)
`/api/health`, `/api/schema`, `/api/datasets` (presets: Quick 3 rounds / Standard 8 / Long 10), POST `/api/run`, GET `/api/stream/<id>` (SSE live log), `/api/run/<id>`, `/api/report/<id>` (full metrics + generator stats), `/api/graph/<id>` (node/edge dump for the graph explorer), `/api/history`.

---

## 15. How to run everything (verified on this machine)

Environment: Python 3.11.9; venv at `multiround-promo-fraud\.venv` with all deps (dgl 2.0.0, torch 2.2.2+cpu, xgboost 2.0.3, flask, pandas, sklearn, etc.). Dataset `tolokers_bid`: 11,758 nodes / 1,049,758 edges / 10 features / 9,192 genuine / 2,566 fraud.

```powershell
# CLI experiment (main framework)
cd multiround-promo-fraud\multiround-promo-fraud\scripts
..\.venv\Scripts\python.exe main.py -c config_intelligent   # INTELLIGENT GAN generator, 2 rounds, XGB-SP
..\.venv\Scripts\python.exe main.py -c config_cpu           # naive/REPLAY baseline for comparison

# Web dashboard (main framework) -> http://127.0.0.1:5050
cd ..\dashboard
..\.venv\Scripts\python.exe app.py

# Companion demo app -> http://127.0.0.1:5050
cd intelligent-fraud-generator
run.bat   # (uses the same .venv)
```

**How to read results:** open `result\<config>\<timestamp>\meta.txt` (exact config) and the CSV files. Per-round metrics appear in the console as `Best Val: REC … PRE … MF1 … AUC …`. With `INTELLIGENT`, check the `gen_*` columns: `gen_feat_div > 0` and `gen_feat_shift > 0` prove the generator creates **new** fraud (Replay shows 0.0); `gen_ring_ratio ≈ 0.5` shows rings/chains are formed; rising `gen_missed_conf` shows the attacker learning to fool the detector.

---

## 16. Key terminology glossary

- **Round** — one detector-retrain + one adversary-injection cycle.
- **Detector** — the GNN/XGB model being evaluated.
- **Adversary** — chooser (which fraud) + modifier (how to generate).
- **Seeds** — fraud nodes that escaped detection (fed to the generator).
- **Budget** — limited ground-truth labels revealed per round (simulates manual review).
- **REAGE / RANDOM** — training-data augmentation strategies.
- **Add-on** — extra rule-based flags OR-ed into model predictions.
- **`gen_*` columns** — diagnostics proving the generator produces diverse, shifted, ring-structured fraud.

---

**Bottom line:** a pluggable multi-round adversarial framework (`detector ↔ adaptive attacker`) for graph-based promo-fraud detection, whose key contribution is the `INTELLIGENT` (GAN/PROB) fraud generator that learns from what slipped through and synthesizes genuinely new attack patterns — with full observability via `gen_*` metrics, reproducible JSON configs, CSV results, and two ways to watch it live (the research dashboard and the synthetic-world demo app).
