# Adaptive 3-Model Ensemble — Reference Document

> **Purpose:** permanent reference for the machine-learning models used in this
> project and exactly how they are implemented, so future work can reproduce,
> reconfigure, or extend the ensemble without re-reading the source.
>
> Last updated: 2026-09-06 (refreshed for the single-app, real-time rebuild).

---

## 1. The 3 models (the project's canonical trio)

| # | Model | Library | Type | Role in the ensemble |
|---|-------|---------|------|----------------------|
| 1 | **XGBoost** | `xgboost` | Gradient-boosted trees (**GBDT**) | Primary detector; warm-started across refits |
| 2 | **HistGradientBoosting** | `scikit-learn` | Histogram-based GBDT | Fast GBDT variant with a different split strategy |
| 3 | **ExtraTrees** | `scikit-learn` | Extremely-randomized trees (**bagging**) | Maximally-randomized bagger; diversity for the blend |

> **Decision (2026-08-30, retained in the single-app rebuild):** the project
> runs exactly these **best 3** — two complementary GBDT families plus one
> randomized bagger, so the adaptive weighting has something orthogonal to
> rebalance. RandomForest, LogisticRegression and LightGBM were dropped and are
> **no longer in the codebase** (see §6 for the empirical rationale).

---

## 2. Where the models are implemented

Single implementation in the current app: **`backend/detector.py`**
(`ThreeModelEnsemble`, written from scratch).

- Each **source** (IEEE replay / Ethereum / webhook) keeps its **own per-source
  ensemble + rolling labeled window**, so the detector is dimension-agnostic —
  refitting on a live labelled stream is what the system does by default.
- Input features come from `backend/features.py`: an account-history/profile
  aggregate block (`PROFILE_COLUMNS`, computed for every source identically)
  concatenated with source-native fields (`IEEE_NATIVE_COLUMNS`,
  `CHAIN_NATIVE_COLUMNS`, `WEBHOOK_NATIVE_COLUMNS`).
- No embeddings, no graphs: three tree models on a compact behavioural vector.

---

## 3. Per-model hyper-parameters (exact, `backend/detector.py`)

### 3.1 XGBoost
```python
params = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.1,
    "eval_metric": "logloss",
    "seed": 42,
    "scale_pos_weight": max(1.0, #neg / #pos),      # class balancing
}
xgb.train(params, dtrain,
          num_boost_round=200,       # first fit
          xgb_model=self._xgb_booster)   # later: 50 extra rounds, resuming the booster
```
- **warm-started across streaming refits** by passing the previous `Booster` back
  in as `xgb_model` (continual learning).

### 3.2 HistGradientBoosting
```python
HistGradientBoostingClassifier(
    max_iter=200, learning_rate=0.1, max_depth=6,
    class_weight="balanced", warm_start=True, random_state=42,
)
```
- **warm-started** by increasing `max_iter` (+30 per refit, cap 800) with
  `warm_start=True` and re-fitting.

### 3.3 ExtraTrees
```python
ExtraTreesClassifier(
    n_estimators=200, min_samples_split=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
```
- sklearn's ExtraTrees has no warm-start; it is **refit from scratch** on the
  bounded rolling labeled window (fast at the ~thousands-of-rows scale used
  online).

---

## 4. How the ensemble is trained & combined

### 4.1 Streaming loop (`backend/livesim.py`)
1. Each labeled event is appended to the source's rolling window
   (`buffer_max_labels`, default 3000, balanced down-sampled before fits).
2. A refit is triggered when either `retrain_interval_events` labelled events
   (default 500) or `retrain_interval_seconds` (default 60) have passed **and**
   the window has both classes.
3. Every model is (warm-start) fit on the window; a trailing 20% holdout slice
   is scored per model at threshold 0.5.

### 4.2 Adaptive weighting (the ADS idea, simplified for streaming)
Per model: `score = max(F1, 0)` on the trailing holdout. Weights are split
proportionally across the available models, with a floor expressed **relative to
the best model** so a collapsed model is bounded without flattening the blend:

```
best   = max(score_j)
floor  = WEIGHT_FLOOR_FRACTION * best          # 0.05
weight_i = max(score_i, floor) / sum(max(score_j, floor))
ensemble_prob = sum(weight_i * prob_i)         # over the models present
```

```
# fallback when every score is 0 (nothing to distinguish):
weight_i = 1/n
```

The floor is deliberately relative rather than an absolute constant like
`f1 + 0.01`: with real IEEE-CIS F1s in the 0.13-0.17 band, an absolute floor is
5-10% of the score and its effect depends on the arbitrary F1 scale, which
quietly reduces "adaptive weighting" to uniform averaging. The relative form is
scale-invariant (scaling every F1 by k leaves the blend unchanged) and expresses
the intended constraint directly: *no model gets less than 5% of the best
model's weight, and no model gets more than the best model's relative skill.*

**Honest note on the observed weights.** On real IEEE-CIS data the three models
score within ~2% of each other on the trailing holdout, so the weights come out
near-uniform (0.334 / 0.329 / 0.337). That is the correct response to
indistinguishable models, not a bug: the F1 estimates come from a few hundred
holdout rows, and forcing a wide spread would be fitting holdout noise. The
weighting is exercised and verified to track F1 by `tests/test_detector.py`
(`TestWeights`), including the collapsed-model and scale-invariance cases.

### 4.3 Decision tier
`ensemble_prob` maps to **Allow / Review / Block** via
`decision_thresholds` (default review ≥ 0.30, block ≥ 0.70).
Known-fraud events (replay labels / scam-address bridge hits) always Block.
Review events that are unlabeled draw from the human-review budget
(`budget_pos`/`budget_neg`); the model is supervised by whatever labels resolve.

### 4.4 Continual learning constraints
- XGBoost and HistGradientBoosting **warm-start** from the previous model.
- Everything lives **in memory per source** (no disk persistence; a re-start
  retrains from the replayed/evaluated stream).

### 4.5 Diagnostics
`GET /api/live/status` exposes per-source `weight_<model>`, per-model
`precision/recall/f1`, buffered label counts, decisions histogram and review
budget — the online analogue of the old result-CSV columns.

---

## 5. Config keys that control the models

| Key | Where | Effect |
|---|---|---|
| `xgb_rounds` / `xgb_extra_rounds` | `backend/config.py` (or `config.local.json`) | XGB first-fit rounds / extra warm-start rounds |
| `hist_max_iter` / `hist_extra_iter` | same | HistGB initial / per-refit `max_iter` growth |
| `et_n_estimators` | same | ExtraTrees tree count |
| `decision_thresholds` | same | Allow/Review/Block bands |
| `min_fit_labels` | same | labelled events before first fit |
| `retrain_interval_events` / `retrain_interval_seconds` | same | warm-start cadence |
| `buffer_max_labels` | same | rolling labelled window size |
| `budget_pos` / `budget_neg` | same | human-review label budget |

---

## 6. Empirical F1 evidence (why the best 3 were chosen)

> Historical record from the 2026-08-30 ablation (on `tolokers_bid`, ADAPTIVE
> config, mixed embedder + DFEAT, special graph, 2 rounds). The original result
> CSVs lived in the pre-rebuild research framework and were removed with it;
> the numbers are preserved here as the rationale for the trio.

**Per-model mean / best validation macro-F1:**

| Model | mean F1 | best F1 |
|---|---|---|
| **HistGradientBoosting** | **0.6638** | 0.6868 |
| **XGBoost** | 0.6520 | 0.6668 |
| **ExtraTrees** | 0.6367 | 0.6494 |
| RandomForest | 0.6280 | 0.6416 |
| LightGBM | 0.6026 | 0.6937 |
| LogisticRegression | 0.5065 | 0.5198 |

**Measured 3-model whole-ensemble F1** (XGB + HistGB + ET, 3 trials × 2 rounds):
mean round_0 **≈0.637**, val_best round_1 **≈0.664** — matching or beating the
6-model baseline (round_0 0.626, round_1 0.628) for ~half the training cost.

**Net takeaway:** dropping the three weakest members costs no meaningful F1;
XGBoost + HistGradientBoosting + ExtraTrees is the right default. (The current
app replaced graph-embedding inputs with behavioural streaming features, so
expect the absolute F1 numbers to shift — the *ordering* and the trio's
diversity rationale transfer directly.)

---

## 6b. Clean temporal holdout on IEEE-CIS (the reportable numbers)

> Measured with `scripts/evaluate.py` on the real `data/train_transaction.csv`
> (2026-09-28). These are the numbers to cite. The dashboard's own per-model F1
> (~0.97-1.00) is a **streaming holdout (optimistic)** — the deployed models are
> scored on a slice of the window they were fitted on, and the warm-started
> boosters have already seen those rows. Do not quote it.

**Protocol.** Strictly temporal and one-way: rows `[0, n_train)` are the
training window, rows `[n_train, n_total)` are the clean holdout. Holdout
features are built after the last training event is absorbed (real history), and
no model is ever fitted, warm-started or resampled across the boundary.

```bash
.venv\Scripts\python.exe scripts\evaluate.py --limit 40000 --holdout 10000
```

**40,000 events (30,000 train / 10,000 holdout, 26 features, threshold 0.50):**

| Model | F1 | Precision | Recall | Accuracy | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.1442 | 0.0873 | 0.4126 | 0.8682 | 0.0978 | 0.7178 |
| HistGradientBoosting | 0.1313 | 0.0806 | 0.3532 | 0.8743 | 0.0855 | 0.6987 |
| ExtraTrees | 0.1665 | 0.1067 | 0.3792 | 0.8979 | 0.0945 | 0.7361 |
| **ENSEMBLE (adaptive)** | **0.1560** | **0.0984** | **0.3755** | **0.8907** | **0.0993** | **0.7310** |

**150,000 events (100,000 train / 50,000 holdout):**

| Model | F1 | Precision | Recall | Accuracy | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| XGBoost | 0.1192 | 0.0792 | 0.2406 | 0.8998 | 0.0644 | 0.6816 |
| HistGradientBoosting | 0.1317 | 0.0889 | 0.2541 | 0.9056 | 0.0735 | 0.6997 |
| ExtraTrees | 0.1450 | 0.0913 | 0.3520 | 0.8830 | 0.0789 | 0.7076 |
| **ENSEMBLE (adaptive)** | **0.1371** | **0.0939** | **0.2541** | **0.9099** | **0.0764** | **0.7155** |

**Reading these numbers honestly.**

- Absolute F1 is **low (0.13-0.17)** because the base rate is ~2.7% and these are
  the first honest (never-fitted) holdout numbers for this pipeline. The
  ensemble's ROC-AUC of 0.72-0.73 and PR-AUC of 0.076-0.099 (~3-4x the 0.027
  base rate) show real ranking signal; F1 at threshold 0.50 is dominated by the
  precision side, which is what a 2.7%-prevalence problem looks like.
- The blend is **not** better than its best single member (ExtraTrees) on F1 at
  threshold 0.50. It wins on accuracy and ROC-AUC, and it is the only variant
  with one weight per model rather than a hand-picked threshold. With three
  near-equal models there is little for a weighted average to recover; the
  adaptive weights land near uniform (see §4.2).
- F1 is **not monotonic in window size** (0.156 at 40k, 0.137 at 150k). The
  longer window covers a different period with a slightly lower fraud rate
  (0.0256 vs 0.0287) and more regime drift, and precision tracks the base rate.
  Reporting only the best run would be cherry-picking.
- The 40k/150k figures come from **replay ground truth**. The chain and webhook
  models cannot be scored this way at all — they are unlabeled by construction,
  which is precisely why the review loop exists and why the honest claim for
  those sources is "learns from human/LLM labels", not a measured F1.

---

## 7. Quick reminders (gotchas)

- ExtraTrees is refit per cadence (no warm-start in sklearn) — fine at the
  bounded window scale.
- A single-class labeling window (all-0 or all-1) skips refits to avoid
  degenerate models; the cold-start fallback probability is clamped [0.10, 0.5].
- Chain events are unlabeled until the scam-address bridge matches them; the
  review-budget loop is the main source of new labels, so the chain model may
  stay in cold start until enough reviews accumulate.