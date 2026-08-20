# Adaptive Multi-Model Ensemble Detector

A standalone demonstration module that implements a **5-model adaptive ensemble** for fraud detection within a multi-round adversarial simulation. Part of the [Enhanced Multi-Round Promo Fraud Detection](../PROJECT_SUMMARY.md) project.

---

## 1. Overview

This module runs a self-contained simulation where:

1. A **synthetic referral marketplace** is seeded with genuine and fraudulent accounts.
2. Five ML models (XGBoost, RandomForest, ExtraTrees, HistGradientBoosting, LogisticRegression) are trained each round on 17 handcrafted features.
3. An **Adaptive Detector Score (ADS)** dynamically reweights model contributions based on F1, recall, stability, and historical performance.
4. Missed fraud is fed back into an evolved fraud generator, and new genuine accounts are added.
5. A small **review budget** reveals ground-truth labels for the next round's training.

The frontend displays live SSE-streamed metrics, per-round model comparison, round leaders, and overall F1/AUC scores.

### How it fits into the project

```
multiround-promo-fraud/
├── multiround-promo-fraud/     ← MAIN framework (GNN + XGBoost on real dataset, 34.9% F1)
├── adaptive-defensive-layer/   ← ADL add-on (attacker + detector + defense ecosystem)
├── intelligent-fraud-generator ← IFG companion demo (GAN adversary + XGBoost detector)
└── adaptive-ensemble-detector/ ← THIS MODULE (5-model ensemble + ADS on synthetic data)
```

This module is the **ensemble research prototype** — it demonstrates the concept of multi-model adaptive weighting but uses simplified synthetic data. See [Section 2](#2-architecture-comparison-this-module-vs-the-paper) for why its F1 scores differ from the main framework.

---

## 2. Architecture Comparison: This Module vs The Paper

### 2.1 Side-by-Side Comparison

| Aspect | Paper (Main Framework) | This Module (Ensemble Demo) |
|--------|----------------------|---------------------------|
| **Dataset** | Real (`tolokers_bid`): 11,758 nodes, 1M+ edges | Synthetic: 500-700 base accounts, 5 templates |
| **Features** | GNN embeddings (GCN/GraphSAGE/GIN → 64-128D vectors) | 17 handcrafted tabular (11 intrinsic + 6 graph) |
| **Models** | XGB-SP (XGBoost on GNN embeddings) | 5 sklearn models + XGBoost (tabular) |
| **Ensemble** | Single model (XGBoost) | 5-model weighted ensemble + ADS |
| **Adversary** | GAN-based Intelligent Generator (learns from missed fraud) | Gaussian perturbation of templates |
| **Threshold** | Grid-search maximizing macro-F1 on validation | Fixed at 0.5 |
| **Training window** | Forgetting window (recent rounds only) | All data from round 0 |
| **Expected F1** | ~34.9% (realistic difficulty) | ~91-99% (trivially separable data) |

### 2.2 Why F1 Differs: Root Cause Analysis

#### Cause 1: Fraud templates are trivially separable

The synthetic world defines 5 fraud profiles with feature means that are **far from the genuine profile**:

| Feature | Genuine mean | Fake Identity mean | Referral Farming mean | Gap (min) |
|---------|-------------|-------------------|----------------------|-----------|
| `email_disposable` | 0.1 | 0.9 | 0.7 | **0.60** |
| `phone_verified` | 0.8 | 0.1 | 0.2 | **0.60** |
| `device_fresh` | 0.2 | 0.85 | 0.7 | **0.50** |
| `ip_proxy` | 0.1 | 0.7 | 0.5 | **0.40** |
| `txn_count` | 0.25 | 0.9 | 0.8 | **0.55** |
| `txn_freq` | 0.3 | 0.8 | 0.7 | **0.40** |

A single decision tree with one split on `email_disposable > 0.4` already separates most fraud from genuine. With 5 models and 17 features, the ensemble trivially achieves near-perfect classification.

#### Cause 2: Noise is too small

Each template uses `noise` values of 0.06-0.15. With means separated by 0.40-0.70, the Gaussians barely overlap:

```
Genuine email_disposable:   N(0.10, 0.15²)  → 95% range [−0.20, 0.40]
Fake Identity email_disp:   N(0.90, 0.12²)  → 95% range [0.66, 1.14]
Overlap zone:               [0.33, 0.40] — less than 3% of either distribution
```

The paper's real dataset has **significant class overlap** because real fraudsters closely mimic genuine behavior.

#### Cause 3: Graph features encode label information

`fraud_neighbor_ratio` (feature index 16) computes the fraction of a node's neighbors that are known fraud. Since the synthetic world creates dense fraud rings (`ring_affinity` up to 0.9), this feature directly reveals the label — a form of **implicit label leakage** through the graph topology.

#### Cause 4: No adversarial evolution

The paper's Intelligent Generator uses a **GAN** that learns from missed fraud and actively generates new patterns designed to fool the current detector. Our module's `_generate_evolved_fraud` only applies Gaussian noise to templates — the attacker doesn't learn or adapt.

---

## 3. Real-World Applicability Assessment

### 3.1 Validated Concepts (Transfer to Production)

The following architectural patterns are **well-established in industry** and transfer directly from this demo to real-world systems:

| Concept | Why It Works | Industry Use |
|---------|-------------|-------------|
| **Multi-model ensemble** | Diversity reduces variance; different models capture different patterns | Netflix Prize, Kaggle, PayPal fraud |
| **Dynamic model weighting** | Models degrade at different rates under concept drift | Stripe Radar, Alibaba risk scoring |
| **ADS-style scoring** | F1 + recall + stability + historical provides a balanced model quality signal | Similar to online model selection in production ML |
| **Round-based retraining** | Retraining on fresh data prevents staleness | Standard in production fraud/abuse systems |
| **Budget-limited labeling** | Simulates real-world constraint of limited human review capacity | Real SOC analyst workflows |

### 3.2 Gaps for Production Deployment

| Gap | Current Demo | Production Requirement |
|-----|-------------|----------------------|
| **Features** | 17 handcrafted tabular | GNN embeddings + behavioral sequences + temporal features + device fingerprints |
| **Graph structure** | Simple device/IP/referral edges | Heterogeneous graph with millions of nodes, edge types, and temporal dynamics |
| **Class overlap** | Templates are far apart | Real fraud mimics genuine users — feature distributions heavily overlap |
| **Adversary** | Gaussian noise | Adaptive attacker that learns detector weaknesses (GAN, reinforcement learning) |
| **Concept drift** | No drift modeling | Fraud patterns evolve monthly; need drift detection + automatic retraining triggers |
| **Threshold** | Fixed 0.5 | Per-model optimal threshold via F1 grid-search; cost-sensitive thresholding |
| **Scale** | ~1,000 nodes | Millions of accounts, real-time inference, batch + streaming pipelines |
| **Explainability** | None | Feature importance, SHAP values, model confidence calibration |

---

## 4. Improvement Roadmap

The following improvements are ordered by **impact on F1 realism** (moving toward the paper's 34.9%) and **implementation effort**.

### IMP-1: Harder Synthetic World

**Impact: HIGH | Effort: LOW | Expected F1: 91% → 60-75%**

Increase fraud template noise and shift means closer to genuine profiles to create realistic class overlap.

**Changes:**
- Increase `noise` from 0.06-0.15 to 0.20-0.35 across all templates
- Shift fraud means closer to genuine for key features (e.g., `email_disposable` from 0.9→0.45 for `quiet_sampler`)
- Add a `"sleeper"` template that nearly matches genuine profiles
- Increase genuine noise to 0.20 for more distribution overlap

**Files:** `backend/world.py` — `FRAUD_TEMPLATES` and `GENUINE_PROFILE` dicts

### IMP-2: Optimal Threshold Search

**Impact: HIGH | Effort: LOW | Expected F1: +3-8% improvement in realistic scenarios**

Replace the fixed 0.5 threshold with grid-search maximizing macro-F1 on validation data (matching the paper's `get_best_f1`).

**Changes:**
- Add threshold search in `EnsembleDetector.evaluate()` and `predict_proba()` usage
- Search thresholds from 0.05 to 0.95 in 0.05 steps
- Store optimal threshold per model per round
- Report threshold in round results

**Files:** `backend/adaptive_detector.py` — `evaluate()`, `predict_proba()`

### IMP-3: Stratified Train/Val Split

**Impact: MEDIUM | Effort: LOW**

Replace the random `supervised_mask` split with stratified sampling that preserves class ratios.

**Changes:**
- Use `sklearn.model_selection.train_test_split(stratify=labels)` for initial supervised set
- Ensure each round's revealed budget maintains class balance

**Files:** `backend/simulation.py` — initial supervised setup + reveal logic

### IMP-4: Forgetting Window

**Impact: MEDIUM | Effort: LOW | Expected F1: -5-10% (more realistic degradation)**

Only train on accounts from recent rounds (matching the paper's `forget_window=2`), so the detector goes stale as old fraud patterns change.

**Changes:**
- Add `forget_window` config parameter (default: 0 = use all data)
- Filter `train_mask` to only include accounts created within the last `forget_window` rounds
- This forces the detector to rely on recent patterns and degrades as the attacker evolves

**Files:** `backend/simulation.py` — training data selection

### IMP-5: Prediction Add-ons

**Impact: HIGH | Effort: MEDIUM | Expected F1: +5-12% in realistic scenarios**

Port the paper's rule-based prediction add-ons (FTHR, DEGREE) that OR extra flags into model predictions. These catch structural anomalies that ML models miss.

**Changes:**
- Implement `FeatureDistThreshold` (DBSCAN on recent 1-hop neighborhood features → flags dense spam clusters)
- Implement `DegreeActivityThreshold` (flags nodes connected to high-degree fraud)
- OR add-on predictions into final fraud probability

**Files:** New `backend/addons.py` + modifications to `backend/adaptive_detector.py`

### IMP-6: Intelligent Adversary (GAN)

**Impact: HIGH | Effort: HIGH | Expected F1: 60% → 40-55%**

Replace Gaussian perturbation with the GAN-based `IntelligentFraudGenerator` from the IFG/ADL modules. The GAN learns from missed fraud and generates genuinely new attack patterns.

**Changes:**
- Port `IntelligentFraudGenerator` from `intelligent-fraud-generator/backend/generator/engine.py`
- Port `FraudProfile` rolling memory from `profile.py`
- Port GAN training loop from `gan.py`
- Connect missed fraud → generator → new fraud injection

**Files:** New `backend/generator/` package (engine.py, gan.py, sampler.py, mutators.py, profile.py)

### IMP-7: GNN Embedding Bridge

**Impact: HIGH | Effort: HIGH | Expected F1: → 35-50% (closest to paper)**

Connect to the main framework's GNN embedder to produce graph-aware features instead of handcrafted ones. This is the single most impactful improvement for matching the paper's results.

**Changes:**
- Import and instantiate the main framework's `TemporalEmbedder` or `VanillaEmbedder`
- Build a DGL graph from the synthetic world
- Generate GNN embeddings as feature vectors for the 5 models
- Keep ADS ensemble on top of GNN embeddings (replacing tabular features)

**Files:** `backend/adaptive_detector.py` (model input) + `backend/simulation.py` (graph construction)

### IMP-8: Feature Importance & Leakage Audit

**Impact: MEDIUM | Effort: LOW**

Log feature importances from tree-based models to identify which features drive predictions and detect any label leakage.

**Changes:**
- Extract `feature_importances_` from RandomForest/ExtraTrees after each round
- Log top-5 features per model
- Flag any graph features (indices 11-16) that dominate predictions (potential leakage)

**Files:** `backend/adaptive_detector.py` — `train()` return value

### IMP-9: Multi-Dataset Support

**Impact: MEDIUM | Effort: HIGH**

Add the ability to load real datasets (like `tolokers_bid`) alongside the synthetic world, enabling direct comparison with the paper's results.

**Changes:**
- Add a `dataset` config option: `"synthetic"` (current) or `"tolokers_bid"` (load from main framework)
- Build DGL graph loader for real datasets
- Adapt feature computation to work with real graph features

**Files:** New `backend/datasets.py` + modifications to `backend/simulation.py`

### IMP-10: Ablation Presets

**Impact: LOW | Effort: LOW**

Add presets that enable systematic comparison of design choices.

**Changes:**
- "Single Model" preset (only XGBoost, no ensemble)
- "No ADS" preset (equal weights, no adaptation)
- "Hard World" preset (IMP-1 applied)
- "With Threshold Search" preset (IMP-2 applied)

**Files:** `backend/app.py` — `datasets()` endpoint

---

## 5. How Each Improvement Maps to the Paper

| Improvement | Paper Component | Section in Paper | Expected F1 Impact |
|------------|----------------|-----------------|-------------------|
| IMP-1: Harder World | Realistic data distribution | Dataset description | 91% → 60-75% |
| IMP-2: Threshold Search | `get_best_f1` grid search | Model prediction | +3-8% precision |
| IMP-3: Stratified Split | Stratified `train_test_split` | Data splitting | More stable training |
| IMP-4: Forgetting Window | `forget_window=2` | Round training | -5-10% (realistic) |
| IMP-5: Add-ons | FTHR/DEGREE prediction add-ons | Meta-strategies | +5-12% recall |
| IMP-6: Intelligent Adversary | `INTELLIGENT` modifier (GAN) | Adversary generation | 60% → 40-55% |
| IMP-7: GNN Embeddings | GNN embedder → XGBoost | `XGB-SP` / `EmbedBoost` | → 35-50% |
| IMP-8: Leakage Audit | Feature analysis | Evaluation | Validates correctness |
| IMP-9: Multi-Dataset | `tolokers_bid` dataset | Experiments | Direct comparison |
| IMP-10: Ablation | Config search grid (`EXP_DICT`) | Ablation study | Research rigor |

**Projected F1 trajectory:**

```
Current (baseline):     91-99%  ← trivially separable synthetic data
+ IMP-1 (harder world): 60-75%  ← realistic class overlap
+ IMP-4 (forgetting):   50-65%  ← detector goes stale
+ IMP-6 (GAN attack):   40-55%  ← adversary actively evolves
+ IMP-7 (GNN embeds):   35-50%  ← graph-aware features
≈ Paper baseline:        34.9%   ← real dataset difficulty
```

---

## 6. Module Technical Reference

### File Layout

```
adaptive-ensemble-detector/
├── run.py                          # Entry point (Flask server, port 5050)
├── run.bat                         # Windows launcher
└── backend/
    ├── __init__.py
    ├── app.py                      # Flask REST + SSE API
    ├── adaptive_detector.py        # 5-model ensemble + ADS scoring
    ├── simulation.py               # Multi-round simulation engine
    ├── world.py                    # Synthetic marketplace world
    └── features.py                 # 17-D feature engineering
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Liveness check |
| `/api/schema` | GET | Config defaults, model names, feature names |
| `/api/datasets` | GET | Presets: Quick (3 rounds), Standard (5), Long (8) |
| `/api/run` | POST | Launch simulation (JSON config body) |
| `/api/stream/<id>` | GET | SSE live feed (logs, round results) |
| `/api/run/<id>` | GET | Run status snapshot |
| `/api/report/<id>` | GET | Full report with per-round metrics |
| `/api/history` | GET | Session run history |

### Default Configuration

```python
DEFAULTS = {
    'rounds': 5,
    'seed': 42,
    'base_accounts': 500,
    'initial_fraud': 60,
    'genuine_per_round': 45,
    'fraud_per_round': 30,
    'supervised_ratio': 0.25,
    'budget_pos': 6,
    'budget_neg': 15,
}
```

### ADS Scoring Formula

```
ADS_score(model) = 0.25 × F1 + 0.25 × Recall + 0.25 × Stability + 0.25 × Historical
```

Where:
- **F1**: Most recent round's macro-F1
- **Recall**: Most recent round's recall (× importance weight)
- **Stability**: `1 - min(std(F1_history) / mean(F1_history), 1)` — reward consistent models
- **Historical**: EMA of F1 scores (α=0.7) — smoothed long-term performance

Weights are normalized: `w_i = ADS_score_i / sum(all_scores)`

### How to Run

```powershell
# Via unified launcher (recommended)
python run.py

# Standalone
cd adaptive-ensemble-detector
run.bat
# Or: .venv\Scripts\python.exe run.py
```

Dashboard: **http://127.0.0.1:5050** (standalone) or **http://127.0.0.1:3000** (unified, via proxy on port 5054)
