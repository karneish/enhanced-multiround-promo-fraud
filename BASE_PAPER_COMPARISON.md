# Base Paper vs. This Project — Full Comparison

**Base paper:** Hafizh Adi Prasetya, Xin Liu, Tsuyoshi Murata, Akiyoshi Matono,
"A multi-rounded adversarial scenario for graph-based promo fraud detection",
*Social Network Analysis and Mining* (2026) 16:24, doi:10.1007/s13278-025-01566-0.
30 pages, received 16 Jun 2025, accepted 3 Dec 2025.

**Project state compared:** `git HEAD = b37fdc0`, working tree clean.

---

## 1. What the base paper actually is

The paper has exactly **two named contributions**, and everything else is
evaluation scaffolding around them.

| # | Contribution | What it is |
|---|---|---|
| C1 | **Multi-round Adversarial Fraud Detection** | An evaluation *scenario*, not a model. Detection is trained and evaluated iteratively on an adversarially evolving graph. Per round it maintains three graphs: test graph `G_t`, query graph `G_t^Q` (only nodes generated this round), support graph `G_t^S` (previous support + new support). Two generators: query generator `g` (seeds from *fraud the detector has not yet caught*, then duplicates them — models coupon/referral spam) and support generator `h` (labels arrive only for nodes the model predicts fraud, modelling the fact that a blocked account never generates a complaint). |
| C2 | **TPNE** (Temporally Pre-trained Node Embedder) | A self-supervised, label-agnostic, model-agnostic node embedder. Splits output into a **static** part (used mainly at round 0) and a **temporal** part (corrects mistakes at `t>0`). Three objectives: neighbourhood reconstruction, temporal disentanglement, temporal maximisation, combined with weights α and β. |

**Headline claims of the paper:**
- Existing models struggle: on-round F1 "consistently hover below 56 percent".
- TPNE improves the best baseline by **up to 4.6 %** on-round F1 and **up to 32.9 %** final recall.
- On Tolokers: TPNE-XGB reaches recall **85.9 ± 3.36**, RT-f1₀ **69.1 ± 1.23**, RT-avg_f1 **55.2 ± 1.67**, overall rank **3.67**.

---

## 2. What is in the project root right now

| Path | Lines / size | Role |
|---|---|---|
| `Confirmed Base Paper.pdf` | 9.1 MB, 30 pp | the base paper |
| `backend/` | 4,631 | detector, embedder, scenario, streaming app |
| `backend/embedder.py` | 712 | TPNE — numpy + torch, offline & incremental modes |
| `backend/detector.py` | 640 | the 3-model ensemble (AMDL head) |
| `backend/sources/adversarial.py` | 474 | C1, the multi-round scenario |
| `backend/livesim.py` | 821 | real-time simulation / refit loop |
| `backend/app.py` | 247 | Flask API + SSE |
| `backend/review.py` | 180 | human/LLM review queue |
| `backend/sources/{ethereum,webhook,replay}.py` | 287/47/619 | live + replayed data sources |
| `scripts/` | 1,600 | `upgraded_tpne.py` (584), `compare_tpne.py`, `evaluate.py`, `embedder_training_curve.py`, `smoke_review_loop.py`, `fetch_ieee.py` |
| `tests/` | 9 files, **184** test functions | unit tests |
| `reports/` | 39 JSON artefacts | every measured run |
| `vendor/author-tpne/` | full | the authors' original code, used as control |
| `FINDINGS_TPNE.md` | 24 KB | the engineering log, including two retractions |
| `frontend/` | Vite app | dashboard |
| `Conference/` | — | the AMDL-TPNE paper (PDF + DOCX + build sources) |
| `data/` | README only | **no CSV present** — IEEE-CIS is licence-gated |
| `requirements-tpne.lock` | **0 bytes** | empty |

---

## 3. Current measured results (author = control, this repo = n=3)

The author control was reproduced from the authors' own code with their own
`extract_metrics.py`:

| Metric | Author TPNE-XGB (n=5) | This repo, masks protocol (n=3) | Delta |
|---|---|---|---|
| RT-f1₀ start | **66.56 ± 0.75** | 54.89 ± 0.82 | −11.67 |
| RT-f1₀ gap | +0.93 ± 1.58 | +0.28 | better |
| RT-avg_f1_{t>0} | **57.09 ± 2.33** | 54.51 ± 4.32 | −2.58 (≈1 SD) |
| RT-recall₀ | 27.50 ± 5.62 | 78.13 ± 2.78 | **+50.63 (2.8×)** |

On the `holdout` split: RT-f1₀ 57.25, RT-avg_f1 **53.06** (Δ −4.03), RT-recall₀
**92.19** (Δ +64.69, 3.4×).

> ⚠ **Retracted figures, for the record only.** An intermediate run
> (`reports/parity_3t.json`) reported RT-avg_f1 **70.0** and was announced as
> beating the paper on all four metrics. It was train-on-test — the `masks`
> split had been collapsed into contiguous ranges, sweeping 100 % of the
> retention set into training and threshold tuning. It was retracted in
> commits `262b3cb` and `b37fdc0` (a second instance in the dual-view branch).
> **Do not quote 70.0 or 94.39.**

---

## 4. Table A — DEGRADED (the paper has it; this project does not, or has it weaker)

| # | Capability | Base paper | This project | Evidence |
|---|---|---|---|---|
| D1 | **Datasets** | 6: Tolokers, Amazon, YelpChi, Elliptic++, T-Finance, TC | **1** (Tolokers). IEEE-CIS is a separate streaming corpus, not a graph benchmark | `FINDINGS` §1, §4.2h |
| D2 | **Scenario horizon** | **10 rounds** per experiment; graph grows to 145 % of original | **3 rounds** | `config_author_control.json` `round_num: 3`; every report has `n_rounds: 3` |
| D3 | **Repetitions** | **10 repeats**, ± std reported on every number | **3 trials** (one arm at n=1) | `fixed_masks_*_3t.json` `trials: 3`; `fixed_masks_dual_1t.json` `trials: 1` |
| D4 | **Baseline breadth** | 13 models: GCN, GraphSAGE, GIN, GAT, GCNII, BWGNN, GHRN, PMP, MetaGAD, GAD-NR, XGB, XGB-GIN, TPNE-XGB | **3** tree models only; **every GNN was deleted** | `MODELS_REFERENCE.md` §1 — "RandomForest, LogisticRegression and LightGBM were dropped"; no GNN classifier anywhere in `backend/` |
| D5 | **Objective ablation** | Table 7: TPNE-XGB\A, \B, \AB, α=0, β=0 — isolates each of the three losses | **Absent** | no ablation artefacts in `reports/` |
| D6 | **α/β sensitivity** | Fig 6, full sweep of both weights | **Absent** — fixed at α=β=1 everywhere | no sweep artefacts |
| D7 | **Training-scheme study** | Table 6: CONTINUE vs FIRST vs RESET across datasets | **Partial** — only the modification default was changed (REPLAY), not a scheme study | `FINDINGS` §4.2a |
| D8 | **Threshold-strategy study** | Table 6, threshold strategies analysed | **Partial** — one threshold tuned on a validation split | `FINDINGS` §4.2 |
| D9 | **Runtime analysis** | Fig 7, per-round seconds for every model + complexity discussion | **Ad-hoc only** — seconds recorded per run, no systematic study | `seconds` key in reports |
| D10 | **Encoder fidelity** | Original implementation, ranking AUC ~0.90-class | **Ported and weaker** — ranking AUC pinned at **~0.82** regardless of classifier or class balancing | `FINDINGS` §4.2h |
| D11 | **Headroom on round-0 F1** | 66.56 | 54.89; root cause is the encoder, not thresholding or pre-training budget | `scripts/embedder_training_curve.py`, spw sweep 1–5 left AUC at 0.814–0.820 |
| D12 | **Compute** | unspecified in the paper; the released config uses 500/300 epochs, early stop 75 | 4 CPU cores, 7 GB RAM, **no GPU**; pre-training only 10–60 epochs | `FINDINGS` §2 |
| D13 | **Data in-repo** | code released; datasets on inquiry | `data/` holds only a README; **no CSV committed**, so the IEEE-CIS numbers cannot be regenerated | `data/` |
| D14 | **Lockfile** | — | `requirements-tpne.lock` is **0 bytes** | repo root |

## 5. Table B — UPGRADED (this project has it; the paper does not)

| # | Capability | This project | Base paper | Evidence |
|---|---|---|---|---|
| U1 | **Adaptive multi-model head (AMDL)** | 3 heterogeneous learners — XGBoost, HistGradientBoosting, ExtraTrees — fused by an F1-proportional weighting with a relative floor (κ = 0.05) | single XGBoost head only | `backend/detector.py` `ThreeModelEnsemble` |
| U2 | **Incremental / online embedder** | second execution mode: propagate only the newest round, temporal half = round-over-round delta; a round costs 1 propagation instead of the window width | **does not exist** — only full-window recomputation | `backend/embedder.py` `mode="incremental"` |
| U3 | **Full streaming application** | Flask API + SSE, live refit loop, review queue, Vite dashboard, `run.py` entry point | **none** — a research benchmark, no deployable system | `backend/app.py`, `livesim.py`, `review.py`, `frontend/` |
| U4 | **Live data ingestion** | Ethereum mainnet JSON-RPC polling + webhook ingest, alongside paced IEEE-CIS replay | offline datasets only | `backend/sources/ethereum.py`, `webhook.py`, `replay.py` |
| U5 | **Real labelled corpus** | IEEE-CIS pipeline with a strictly temporal holdout protocol | no real transaction corpus | `scripts/evaluate.py` |
| U6 | **Test suite** | **184** test functions across 9 files | none released with the paper | `tests/` |
| U7 | **Leakage auditing** | found and fixed **two** real train-on-test defects in its own harness, retracted the results, added a regression test asserting no `[split:]` indexing remains | no equivalent | `FINDINGS` §4.2e, §4.2i, `tests/test_split.py` |
| U8 | **Scenario-parity fix** | REPLAY clones must inherit the seed's edges — a parity bug with the reference | — | commit `58f1426` |
| U9 | **Documented negative results** | pre-training tested at 100 epochs: loss 0.571→0.235, F1 unchanged — hypothesis **refuted and recorded** | paper reports the positive result only | `scripts/embedder_training_curve.py`, `FINDINGS` §4.2c |
| U10 | **Honest retraction culture** | a "beats the paper" claim was published in-repo then retracted in writing with root-cause analysis | — | `FINDINGS` §1 retraction box |

---

## 6. Scored totals

Rubric: each dimension weighted to 100, each side scored /10, weighted mean ×10.

| # | Dimension | Weight | Author | Repo | Weighted A | Weighted R |
|---|---|---|---|---|---|---|
| 1 | Scenario fidelity (`G_t`/`G_t^Q`/`G_t^S`, generators) | 12 | 10.0 | 8.0 | 12.00 | 9.60 |
| 2 | Encoder / TPNE fidelity | 12 | 10.0 | 6.0 | 12.00 | 7.20 |
| 3 | Metric implementation (all 3) | 8 | 10.0 | 9.5 | 8.00 | 7.60 |
| 4 | Protocol rigor (rounds × repeats × splits) | 15 | 10.0 | 4.0 | 15.00 | 6.00 |
| 5 | Dataset breadth | 12 | 10.0 | 2.5 | 12.00 | 3.00 |
| 6 | Baseline breadth | 12 | 10.0 | 2.0 | 12.00 | 2.00 |
| 7 | Ablation & sensitivity studies | 10 | 10.0 | 3.0 | 10.00 | 3.00 |
| 8 | Runtime / operational analysis | 5 | 9.0 | 4.0 | 4.50 | 2.00 |
| 9 | Reported performance | 9 | 10.0 | 5.0 | 9.00 | 4.50 |
| 10 | Reproducibility & hygiene | 5 | 8.0 | 7.5 | 4.00 | 3.75 |
| | **TOTAL** | **100** | | | **98.50** | **49.05** |

**Final score — base paper: 98.5 / 100. This project: 49.1 / 100.**

### Score without the engineering-only dimensions

If you score only what the two works *claim* (dimensions 1–3, 7, 9 — dropping
protocol breadth, dataset breadth, baseline breadth and runtime, where this
project is simply doing less science rather than doing it worse):

| | Author | Repo |
|---|---|---|
| Claim-only subtotal | 51.00 / 51 | 30.65 / 51 |

**60.1 % of the author.** That is the fairer reading of the research
contribution; the 49.1 figure is dominated by scale of evaluation, not by
quality of what was built.

---

## 7. Verdict

**What the project genuinely gained:** a deployable streaming system, an
adaptive multi-model head, an incremental embedder the paper does not have,
a real test suite, and — most valuable — the discovery that two of its own
results were train-on-test, with the retraction written up honestly. The
leakage audit (U7/U10) is work the published paper never had to do.

**What it gave up:** the entire breadth of the empirical study. One dataset
instead of six, three rounds instead of ten, three trials instead of ten,
three models instead of thirteen, no objective ablation, no α/β sweep. Most of
the score gap is that, not capability.

**Where it actually stands:** on the metric the paper is built around —
on-round F1 under the adversarial scenario — it is **2.58 points behind, which
is about one standard deviation, i.e. not a loss that is statistically
meaningful and certainly not a win**. On final recall it is **2.8× better**,
though that is substantially a threshold effect the project itself documents.
On round-0 F1 it is **11.7 points behind**, and the project's own evidence
attributes that gap to the ported encoder's AUC ceiling (~0.82 vs the ~0.90
needed), not to anything tunable.

**The single highest-value next step** is the one already identified in
`FINDINGS` §4.2c: use the authors' original `TemporalEmbedder` instead of the
port. That is the stated blocker for the one metric that is genuinely lost.