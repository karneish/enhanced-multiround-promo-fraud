# Enhanced Multi-Round Promo-Fraud — Real-Time Adaptive 3-Model Detector

> **This repository supersedes the previous multi-app framework** (the Adaptive
> Defensive Layer, the Intelligent Fraud Generator and the standalone Adaptive
> Ensemble Detector). Those components were removed by the single-app rebuild and
> are replaced here by one streaming application. The previous `main` is
> recoverable from history.

Real-time streaming fraud detection for **multi-round promotional and reward
abuse**, restoring the two contributions of the base paper:

> Hafizh Adi Prasetya, Xin Liu, Tsuyoshi Murata, Akiyoshi Matono.
> *"A multi-rounded adversarial scenario for graph-based promo fraud detection."*
> **Social Network Analysis and Mining** (2026).
> doi:[10.1007/s13278-025-01566-0](https://doi.org/10.1007/s13278-025-01566-0)

The single-app rebuild had deleted both of them — its own report listed graph
embeddings as out of scope. They are restored here.

---

## What this adds over the previous version

| Addition | Path | Status |
|---|---|---|
| **TPNE** — Temporally Pre-trained Node Embedder (`L_recon`, `L_disent`, `L_temp`) | `backend/embedder.py` | ✅ restored |
| **Online incremental embedder** — one propagation per round instead of a full re-encode | `backend/embedder.py` (`mode="incremental"`) | ✅ **new** |
| **Adversarial multi-round scenario** — `G_t` / `G_tQ` / `G_tS`, generators `g` and `h` | `backend/sources/adversarial.py` | ✅ restored |
| IEEE-CIS `V1..V339` pass-through (opt-in) | `--v-features` | ⚠️ wired, unproven |
| Gradient pre-training (Adam on the three losses) | `embedder.pretrain()` | ✅ added, A/B pending |

## Honest status

Measured on the base paper's own benchmark (`tolokers_bid`, adversarial
multi-round, identical protocol):

| Metric | Author TPNE-XGB (n=5) | This repo, offline (n=3) | This repo, incremental (n=3) |
|---|---|---|---|
| RT-avg_f1_{t>0} | **57.09 ± 2.33** | 15.63 | 28.24 |
| RT-recall₀ | **27.50 ± 5.62** | 0.00 | 17.97 |

**This project does not yet exceed the base paper.** The incremental embedder
does beat our own faithful port of the paper's embedder (+80.7% F1), but the
gap to the author's published code is real and is documented rather than
hidden. See **[`FINDINGS_TPNE.md`](FINDINGS_TPNE.md)** for the full audit,
including why the gap exists and what closes it.

## Layout

| Path | What |
|---|---|
| `backend/` | detector, features, embedder (TPNE), adversarial scenario, Flask + SSE |
| `backend/sources/` | replay (IEEE-CIS), ethereum, webhook, adversarial |
| `frontend/` | React + Vite dashboard |
| `tests/` | 167 `unittest` tests |
| `scripts/` | `evaluate.py` (clean holdout), `compare_tpne.py` (paper comparison) |
| `vendor/author-tpne/` | the author's published code + `tolokers_bid`, used only to reproduce their numbers |
| `reports/` | measured results from the reproduction and the A/B |
| `Confirmed Base Paper.pdf` | the base paper |
| `FINDINGS_TPNE.md` | full findings, defects fixed, and false claims corrected |

## Environments

Two, deliberately separate:

```bash
# 1. the streaming application
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. TPNE / the paper reproduction (torch, dgl, torch_geometric)
uv venv --python 3.11 .venv-tpne
uv pip install --python .venv-tpne -r requirements-tpne.lock
```

The application works without the second environment; `backend/embedder.py`
degrades cleanly when torch is absent.

## Running

```bash
.venv-tpne/bin/python scripts/compare_tpne.py --mode ours \
    --mode-flag incremental --rounds 3 --trials 3 --pretrain-epochs 10
```

## Licence

MIT. `vendor/author-tpne/` is third-party code from AIST and remains under its
own licence; `Confirmed Base Paper.pdf` is the published article.