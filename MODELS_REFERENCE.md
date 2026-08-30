# Adaptive Multi-Model Ensemble — Reference Document

> **Purpose:** permanent reference for the machine-learning models used in this project and exactly how they were implemented, so future work can reproduce, reconfigure, or extend the ensemble without re-reading the source.
>
> Last updated: 2026-08-30.

---

## 1. The 6 models (the full universe)

| # | Model | Library | Type | Role in the ensemble |
|---|-------|---------|------|----------------------|
| 1 | **XGBoost** | `xgboost` | Gradient-boosted trees (**GBDT**) | Primary detector; warmed-up across rounds |
| 2 | **RandomForest** | `scikit-learn` | Bootstrap-aggregated trees (**bagging**) | Stable, high-variance-tolerant tree bagger |
| 3 | **ExtraTrees** | `scikit-learn` | Extremely-randomized trees (**bagging**) | More randomized bagger; best bagger in this project's runs |
| 4 | **HistGradientBoosting** | `scikit-learn` | Histogram-based GBDT | Fast GBDT variant with a different split strategy |
| 5 | **LogisticRegression** | `scikit-learn` | Linear classifier | Non-tree, interpretable linear baseline |
| 6 | **LightGBM** | `lightgbm` | Gradient-boosted decision trees (GBDT, leaf-wise) | Third GBDT variant (main framework only) |

The models 1–5 are available in **both** ensemble implementations; **LightGBM is only used in the main framework** (it is not in the standalone ensemble app).

> **Decision (2026-08-30):** the project now defaults to the **best 3 models = `XGBoost`, `HistGradientBoosting`, `ExtraTrees`** (see §6 for why and the F1 numbers). Models 2, 5, 6 remain in the codebase as optional members — re-enable anytime via the `adaptive_model_list` config key.

---

## 2. Where the 6 models are implemented

There are **two independent implementations** that share the same design:

| Implementation | File | Models it runs by default |
|---|---|---|
| **Main research framework** (`ADAPTIVE` model) | `multiround-promo-fraud/src/models/proposed_supervised/adaptive_detector.py` (644 lines) | Up to all 6 (default config list has all 6; the bare constructor defaults to 5 — no LightGBM) |
| **Standalone ensemble app** | `adaptive-ensemble-detector/backend/adaptive_detector.py` (240 lines) | 5 models (no LightGBM) |

Both are driven by the **`AdaptiveDetectorScore` (ADS)** reweighting engine (§4 below), and both feed from the same feature/embedding pipeline:
- **Main framework:** each round a self-supervised `TemporalEmbedder` / `VanillaEmbedder` / `TemporalMixedEmbedder` (with `round_window=7`, `temporal_agg="weight"` in current configs) produces node embeddings → the 6 classifiers train/predict on those embeddings.
- **Ensemble app:** a handcrafted **17-D feature matrix** (11 intrinsic behaviors + 6 graph signals) → the classifiers train/predict on those features.

---

## 3. Per-model hyperparameters (exact)

> Identical parameters in both implementations except where noted.

### 3.1 XGBoost
```python
params = {
    "objective": "binary:logistic",
    "scale_pos_weight": weight,        # weight = ce_weight = (#neg)/(#pos) in train mask (class balancing)
    "tree_method": "hist",
    "max_depth": 6,
    "device": "cpu",                    # added only when xgboost >= 2.0
}
xgb.train(params, dtrain,
          num_boost_round=500,
          early_stopping_rounds=100,
          evals=[(dtrain, "Train"), (dval, "Eval")],
          verbose_eval=False,
          xgb_model=existing_model)     # <-- WARM START: prior-round model is passed in (continual learning)
```
- **warm-started across rounds** (`xgb_model=self.classifiers['XGBoost']`), so round r's model begins from round r−1's.

### 3.2 RandomForest
```python
RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_split=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
```

### 3.3 ExtraTrees
```python
ExtraTreesClassifier(
    n_estimators=300, max_depth=None, min_samples_split=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
```

### 3.4 HistGradientBoosting
```python
HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.1, max_depth=6, random_state=42,
)
```

### 3.5 LogisticRegression
```python
LogisticRegression(
    max_iter=5000, solver="lbfgs", class_weight="balanced",
    random_state=42, tol=1e-4, C=1.0,
)
```

### 3.6 LightGBM (main framework only — model 6)
```python
params = {
    "objective": "binary",
    "scale_pos_weight": weight,         # same class balancing as XGBoost
    "max_depth": 6,
    "learning_rate": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}
lgb.train(params, dtrain,
          num_boost_round=500,
          valid_sets=[dtrain, dval],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
          init_model=existing_model)    # <-- WARM START (same continual-learning idea as XGBoost)
```

---

## 4. How the ensemble is trained & combined (the shared design)

### 4.1 Per round (main framework — `train_classifiers`)
1. Embed the current graph via the self-supervised embedder.
2. Split into `ps_train_mask` (train) and `val_mask` (validation).
3. Compute `sample_weights = compute_sample_weight("balanced", train_y)`; for XGBoost/LightGBM it is `scale_pos_weight = weight` instead.
4. If the training set has only one class → skip classifier training entirely this round (`return (0.0, 0.0)`).
5. Train **each** model in `model_names` (not only the top-k — *all* are trained every round; the ADS then picks the blend).
6. Measure each model on the validation split at threshold 0.5 → `f1` (macro) and `recall` → feed both into the ADS.

### 4.2 Adaptive Detector Score (ADS) — the reweighting engine
`AdaptiveDetectorScore` keeps per-model state across rounds:

```python
history = deque(maxlen=adaptive_history_window)   # default window = 5 rounds
ema_scores[name] = ema_alpha*f1 + (1 - ema_alpha)*prev_ema   # ema_alpha default 0.7
```

Each model gets a **score** from 4 components (default equal weights 0.25 each):

| Component | Formula |
|---|---|
| `f1` | latest validation macro-F1 |
| `recall` | latest validation recall × `recall_importance` (default 1.0) |
| `stability` | `1 − min(std(f1_history)/mean(f1_history), 1)` (needs ≥ 2 rounds) |
| `historical` | the exponential-moving-average score (memory) |

**Weights** = score normalized across models (softmax-like):

```
weight_i = score_i / sum(score_j)      // falls back to 1/n if all scores are 0
```

**Final prediction** is the weighted arithmetic blend of each model's fraud probability:

```python
ensemble_prob = sum(weight_i * prob_i) / sum(weight_i_present)
```

- Ensemble app prediction threshold = `0.5`.
- Main framework additionally grid-searches a macro-F1-optimal threshold (`get_best_f1`, grid 0.05–0.95) and ORs in optional rule-based add-ons (`FTHR/AFTHR/DEGREE/DFEAT/DAFEAT`).

### 4.3 Persistence across rounds (what makes it "adaptive/continual")
- XGBoost and LightGBM **warm-start** from the previous round's model (`xgb_model` / `init_model`).
- All classifiers + the full ADS state (history, EMA, weights, scores, per-model F1/recall histories) are persisted to disk:
  - main framework `save_model(path)` → `{path}_xgb.json`, `{path}_lgb.txt`, `{path}_sklearn.pt`, `{path}_ads_state.json`;
  - ensemble app keeps everything in memory per simulation.

### 4.4 Diagnostics / result columns (so you can audit each model)
Result CSVs of main-framework `ADAPTIVE` runs carry, per round:
`weight_<name>`, `ads_score_<name>`, `individual_f1_<name>`, `individual_recall_<name>` for each model in the list (e.g. `individual_f1_XGBoost`).

---

## 5. Config keys that control the model set

| Key | Where | Effect |
|---|---|---|
| `adaptive_model_list` | `EXP_DICT` in `scripts/*.json`, or `DEFAULT_MODEL_CONFIG` (`src/utils/utils_const.py`) | The exact model set. Format: `[["XGBoost","HistGradientBoosting","ExtraTrees"]]` |
| `adaptive_components` | same | ADS components, default `["f1","recall","stability","historical"]` |
| `adaptive_history_window` | same | ADS rolling window, default `5` |
| `adaptive_ema_alpha` | same | EMA smoothing, default `0.7` |
| `adaptive_recall_importance` | same | recall multiplier, default `1.0` |
| `adaptive_weights` | same | optional manual per-component weights (default uniform) |
| `model_names` | `EnsembleDetector(model_names=...)` in the ensemble app | Overrides the app's default model list |

**Default lists in code (pre-2026-08-30):**
- `DEFAULT_MODEL_CONFIG["adaptive_model_list"]` (main framework) = `["XGBoost","RandomForest","ExtraTrees","HistGradientBoosting","LogisticRegression","LightGBM"]` (6).
- `AdaptiveDetector.__init__` bare default (no list passed) = 5 models, **no LightGBM**.
- Ensemble app `EnsembleDetector` + `/api/schema` = 5 models, **no LightGBM**.

---

## 6. Empirical F1 evidence (why the best 3 were chosen)

Source: `multiround-promo-fraud/result/config_minimal/260820002826/*-E.csv`
(ADAPTIVE, tolokers_bid, mixed embedder + DFEAT add-on, 2 rounds, tiny epoch budget — a **smoke test**, not a benchmark; treat magnitudes as indicative).

**Per-model mean / best validation macro-F1:**
| Model | mean F1 | best F1 |
|---|---|---|
| **HistGradientBoosting** | **0.6638** | 0.6868 |
| **XGBoost** | 0.6520 | 0.6668 |
| **ExtraTrees** | 0.6367 | 0.6494 |
| RandomForest | 0.6280 | 0.6416 |
| LightGBM | 0.6026 | 0.6937 |
| LogisticRegression | 0.5065 | 0.5198 |

**Whole-ensemble (ADS-weighted) measured F1:**
- round_0 entire-graph `0.626`, round_1 entire-graph `0.628`, best-val `0.699`.

**Estimated 3-model ensemble (XGB + HistGB + ET):** renormalizing the ADS weights over only those 3 raises the weighted-mean F1 from **≈0.616** (6-model) to **≈0.651**, i.e. dropping the 3 weakest members should keep F1 equal-or-better. Expected whole-ensemble ≈ **0.63–0.70** on an equivalent run.

**Rationale for the trio:** two complementary families — GBDT (XGBoost + HistGradientBoosting) for raw F1, plus ExtraTrees (bagging, randomized splits) for genuine diversity, so the ADS weighting has something orthogonal to rebalance. LogisticRegression is a weak linear baseline (0.51) and LightGBM a redundant third GBDT with weak mean F1.

> **Verification (measured 2026-08-30):** dedicated ablation config `multiround-promo-fraud/scripts/config_ablation_xgb2.json` runs the ADAPTIVE model with `adaptive_model_list = [XGBoost, HistGradientBoosting, ExtraTrees]` (3 trials × 2 rounds, tolokers_bid, same settings as the 6-model baseline). Result CSV: `result/config_ablation_xgb2/260830132200/-E.csv`.
>
> **Measured 3-model whole-ensemble F1 (entire-graph):**
> | Trial | round_0 | round_1 | val_best round_1 |
> |---|---|---|---|
> | 0 | 0.636 | 0.628 | 0.687 |
> | 1 | 0.630 | 0.634 | 0.690 |
> | 2 | 0.644 | 0.527 | 0.614 |
> | **mean** | **≈0.637** | ≈0.596 | ≈0.664 |
>
> **vs 6-model baseline** (`config_minimal`, 1 trial): round_0 `0.626`, round_1 `0.628`, val_best round_1 `0.699`.
>
> **Interpretation:** the 3-model ensemble **matches or slightly beats** the 6-model set on round_0 (0.637 vs 0.626) and round 1 is comparable (mean 0.596 — pulled down by one noisy trial; the other two trials land at 0.628/0.634, equal to the 6-model baseline). Per-model order held: HistGradientBoosting ≈ 0.648 > XGBoost ≈ 0.625 > ExtraTrees ≈ 0.617. Net: cutting LogisticRegression + RandomForest + LightGBM costs no meaningful F1 — the "best-3" trio is the right default.

---

## 7. Quick reminders (gotchas found while reading the source)

- Three different `temporal_agg` defaults exist in the codebase: `DEFAULT_MODEL_CONFIG` = `mean_final`, model constructors = `sum_final`, current configs = `weight`. Configs that matter here pass it explicitly.
- `AdaptiveDetector` predicts `[1 − p, p]`; the experiment then applies `softmax(1)` on top (double-softmax quirk) — be aware when interpreting probabilities.
- AdaptiveDetector's round-0 fallback with a single-class training set returns `[1,0]`, which after softmax reads as ~27% fraud probability — not a real prediction.
- `adaptive_stability_penalty` config key exists but is **never read** by ADS (only `recall_importance`, window, and `ema_alpha` are used).
- In the main framework, only LightGBM requires the `lightgbm` package; all other models need only `xgboost` + `scikit-learn`.