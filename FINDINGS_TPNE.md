# TPNE Restoration — Findings

Work performed against the published base paper:

> Hafizh Adi Prasetya, Xin Liu, Tsuyoshi Murata, Akiyoshi Matono.
> **"A multi-rounded adversarial scenario for graph-based promo fraud
> detection."** *Social Network Analysis and Mining* (2026).
> doi:[10.1007/s13278-025-01566-0](https://doi.org/10.1007/s13278-025-01566-0)
> Author's code: `github.com/hafizhadi/multiround-promo-fraud`

---

## 1. Summary

The base paper's two named contributions — the **TPNE** self-supervised node
embedder and the **adversarial multi-round scenario** — had been dropped
entirely by the single-app rebuild, which the Phase I report explicitly listed
as out of scope. This work restores both into the streaming application,
reproduces the author's own numbers as a control, and measures an
**online incremental** variant of the embedder.

**The headline result is not the one I set out to get.** We did not beat the
author. The honest comparison is in §4, and it shows our restored pipeline
trailing the author's published code on the primary benchmark while clearly
beating our own faithful re-implementation of the same embedder.

---

## 2. Environment

The author's stack requires Python 3.11 and a GPU; this machine has Python
3.13, 4 CPU cores, 7 GB RAM and no GPU. Resolved with a separate environment so
the working application is untouched:

| Component | Author's pin | Installed | Why |
|---|---|---|---|
| Python | 3.11 | 3.11.17 | 3.13 has no torch 2.2 / dgl wheels |
| torch | 2.2.2+cu121 | **2.2.1+cpu** | no GPU; 2.2.1 is the newest with a DGL graphbolt `.so` |
| dgl | 2.0.0+cu121 | **2.1.0** | 2.0.0 ships no cp311 x86_64 wheel |
| torchdata | — | **0.7.1** | DGL 2.1 imports `torchdata.datapipes`, removed in 0.10+ |
| torch_geometric | 2.5.3 | 2.5.3 | pure Python |
| xgboost | 2.0.3 | 2.0.3 | `py3-none` wheel |
| numpy / pandas / sklearn / scipy | as pinned | as pinned | |

The author's code was patched for CPU: `device: 'cpu'`, the five hardcoded
`device='cuda:0'` defaults, and three `torch.cuda.mem_get_info()` prints. One
further patch creates the results directory, which their script assumed
already existed.

```bash
uv python install 3.11
uv venv --python 3.11 .venv-tpne
uv pip install --python .venv-tpne torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-tpne dgl==2.1.0 torchdata==0.7.1 torch_geometric==2.5.3 \
    xgboost==2.0.3 numpy==1.24.3 pandas==2.2.1 scikit-learn==1.3.0 scipy==1.12.0 \
    networkx==3.1 seaborn==0.12.2 setuptools pyyaml pydantic flask flask_cors \
    requests websocket-client
```

Locked in `requirements-tpne.lock`.

---

## 3. What was restored

### 3.1 TPNE — `backend/embedder.py`

The self-supervised embedder with all three paper objectives, in both modes:

| Mode | Behaviour |
|---|---|
| `offline` | Faithful to the paper — replays `round_window` rounds and recomputes the whole embedding each round |
| `incremental` | **The upgrade** — embeds only the newest round from the cached static state, so a round costs one propagation instead of `round_window` |

Losses: `L_recon` (neighbourhood-distance preservation, Eq. 9), `L_disent`
(cosine separation of the static and temporal halves, Eq. 10), `L_temp`
(temporal maximisation, Eq. 11), combined as
`(L_recon + α·L_disent + β·L_temp) / (1 + α + β)`.

The reconstruction term is computed **edge-wise** with a sparse incidence
matrix rather than as a dense `n × n × d` distance tensor. The dense form is
numerically identical but needed 72 GiB at 12k nodes; it is not usable.

### 3.2 Adversarial multi-round scenario — `backend/sources/adversarial.py`

Ported the author's generator machinery: `random` / `greedy` / `ogreedy` seed
selection, `replay` / `relative` / `absolute` / `mixing` modification, and
`none` / `random` / `reage` resampling. Round 0 is the real labelled graph; each
later round emits only its query graph `G_tQ`, per the paper's definition.
Model decisions are fed back so `greedy` can target the accounts being missed.

### 3.3 IEEE-CIS V1..V339 — `backend/features.py`, `backend/sources/replay.py`

`replay.py` was reading all 339 anonymised V columns and discarding all but a
missing-value count. They are now available to the model behind
`ieee.use_v_features` / `scripts/evaluate.py --v-features`, defaulting to off so
existing behaviour and reported numbers are unchanged.

---

## 4. Results

### 4.1 The control — the author's code, reproduced

`vendor/author-tpne`, `config_author_control.json` (`XGB-SP`, `temporal`,
`REAGE`, `round_num 3`, 5 trials, 500 epochs), on `tolokers_bid`:

| Metric | Author's run | Paper's claim |
|---|---|---|
| RT-f1₀ (initial) | **66.56 ± 0.75** | — |
| RT-f1₀ gap | **+0.93 ± 1.58** | — |
| RT-avg_f1_{t>0} | **57.09 ± 2.33** | "consistently hover below 56 percent" |
| RT-recall₀ | **27.50 ± 5.62** | — |

The reproduction lands on the paper's stated ~56% ceiling, so the control is
sound and the comparison below is meaningful.

### 4.2 Our restored pipeline, same benchmark and protocol

3 rounds, 3 trials, identical budgets (5% of pos/neg per round), same
`greedy` / `relative` / `reage` scenario.

| Metric | Author TPNE-XGB (n=5) | Ours, offline (n=3) | Ours, **incremental** (n=3) |
|---|---|---|---|
| RT-f1₀ initial | 66.56 | 55.33 | 56.32 |
| RT-f1₀ gap | +0.93 | −1.00 | −0.71 |
| **RT-avg_f1_{t>0}** | **57.09** | 15.63 | **28.24** |
| **RT-recall₀** | **27.50** | 0.00 | **17.97** |

**Read this honestly:**

- **We do not beat the author.** On the primary benchmark our restored
  pipeline is well behind their published code — by 28.9 points of on-round F1
  and 9.5 points of final recall.
- **The incremental upgrade is a real, large gain over our own faithful
  re-implementation**: on-round F1 **28.24 vs 15.63** (+81%) and final recall
  **17.97 vs 0.00**. The offline embedder collapses entirely on generated
  fraud; the incremental one does not.
- Incremental also improves **monotonically** across rounds where offline
  fluctuates: F1 0.270 → 0.311 → 0.359 vs 0.190 → 0.297 → 0.162.

### 4.3 Why we still trail — what I know and what I do not

The gap is real and I have not closed it. The likely causes, in order:

1. **Our scenario is a re-implementation, not the author's.** Round budgets,
   the augmentation schedule and the node-duplication semantics are ported from
   their source but not verified to be numerically identical. Any divergence
   here changes the difficulty of the benchmark itself.
2. **The classifier differs.** Their `XGB-SP` trains one XGBoost head on the
   TPNE embedding for up to 500 epochs per round. Our 3-model tree ensemble
   warm-starts on a bounded 3,000-row window. Under a scenario that
   deliberately generates new fraud each round, that buffer may simply be too
   small to carry the history.
3. **Our embedder is numpy, not the differentiable torch module.** Gradients
   never flow through it, so `L_recon` / `L_disent` / `L_temp` are *measured*
   but not *optimised*. The author's losses shape the representation during
   training; ours only score it afterwards. This is the deepest difference and
   probably the largest contributor.

Point 3 means the "incremental" win is currently a win over a *degenerate*
offline baseline rather than over the real method. Before claiming the
incremental embedder as a contribution, the offline path needs the same
gradient-based pre-training so both arms are trained rather than merely scored.

### 4.4 IEEE-CIS V-feature A/B — inconclusive

| Config | F1 | Precision | Recall | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| 26 columns (default) | 0.1686 | 0.0963 | 0.6769 | 0.1141 | 0.7788 |
| 353 columns (`--v-features`) | 0.1680 | 0.0962 | 0.6615 | 0.1128 | 0.7816 |

**This proves nothing.** `data/` holds no real CSV — both runs used the
deterministic synthetic fallback, whose V columns are all zero by construction
(`replay.py:399` seeds every `C*`/`D*`/`V*` to 0.0). The flag is wired
correctly and the A/B is ready, but it needs the real
`train_transaction.csv` under Kaggle terms to measure anything.

---

## 5. Defects fixed along the way

| Defect | Where | Fix |
|---|---|---|
| Sample fallback crashed on every run | `replay.py` `_build_sample` formatted a whole array with `%d` | per-row comprehension |
| Evaluator crashed when asked for more rows than the slice holds | `scripts/evaluate.py` | stop at the `None` end-of-stream sentinel |
| `--v-features` raised `KeyError` on columns a source never supplies | `backend/features.py` | `FeatureBuilder.columns_for_source` + optional `extra_columns` |

## 6. False claims corrected in the Phase I report

Three statements in `Research paper/_report_content_*.py` did not match the code:

1. **Appendix A.2** claimed the scam-address list "supplies a display-only
   oracle column and **never a training label**". `livesim.py:447` uses
   `matcher.label_for(ev)` as the real label, which reaches `add_labeled` and
   trains the ensemble. Corrected — and it contradicted the report's own §2.8,
   which describes the bridge as a label source.
2. **Appendix A.4** showed a fabricated SSE payload (`"type":"scored"`,
   `"score"`, uppercase `"BLOCK"`, a per-model `models{}` block, an ISO
   timestamp). The real record is `livesim.py:518-533`. Replaced with the actual
   shape, and the prose now says the weights come from the status endpoint.
3. **§4.4 / §4.7 / Table 3.1 / §2.5** claimed "above 4,000 events/s". The
   `rows_per_sec` field divides scored rows by `_score_seconds`, which covers
   in-lock processing only and **excludes queue wait and pacing** — it cannot
   measure stream throughput. Reworded to say what it actually is.

Also corrected: Appendix A.3's reproduction command used `--rows 40000`, a flag
that does not exist; `scripts/evaluate.py` takes `--limit`.

---

## 7. Tests

`160` tests, up from `137`:

| Suite | Tests |
|---|---|
| `tests/test_embedder.py` (new) | 10 |
| `tests/test_adversarial.py` (new) | 13 |
| existing 7 suites | 137 |

One failure remains: `test_replay_source.TestPacer.test_fast_rates_are_not_limited_by_the_windows_timer`.
It is a **pre-existing timing flake**, verified two ways:

- On the unmodified file (`git stash push -- backend/sources/replay.py`) it fails
  3 of 4 runs at low load and 5 of 6 runs under load.
- The diff to `backend/sources/replay.py` contains **zero** changes to `_Pacer`
  (`git diff | grep -c "_Pacer\|_MIN_SLEEP\|_MAX_CATCHUP"` → `0`); it touches only
  `__init__`, `_build_sample` and `_build_events`.

The cause is that the test's rate-2000 case requires ≥ 2.25 ms of elapsed sleep
across 5 waits, while `_Pacer.wait` advances `self._next` *before* deciding
whether to sleep, so the first wait at `period == _MIN_SLEEP == 0.0005` takes
the `delay > _MIN_SLEEP` branch as false and sleeps nothing — one period is lost
per five. The lower bound is therefore met or missed by sub-millisecond timing.
Left unfixed because changing `_Pacer` would alter replay pacing behaviour and is
outside the scope of this work; it should be addressed separately.

---

## 8. What I would do next

1. **Gradient-based pre-training** for both offline and incremental arms, so
   the comparison in §4.2 is between two trained embedders rather than one
   trained and one merely scored. Until then the incremental result is
   promising but not yet a defensible contribution.
2. **Verify the scenario port numerically** against the author's own generator
   before trusting any cross-implementation number further.
3. **Obtain the real IEEE-CIS CSV** and run the V-feature A/B properly.
4. **Request the remaining five datasets** from the corresponding author
   (`hafizh@net.comp.isct.ac.jp`); only `tolokers_bid` is public, so our
   comparison is narrower than the paper's.
5. Consider whether the ensemble window (3,000 rows) is adequate for a
   scenario whose whole point is carrying multi-round history.

## 9. Reproducing this

```bash
# control - the author's own code, ~2 h on 4 CPU cores
cd vendor/author-tpne && git pull
cd scripts
../../../.venv-tpne/bin/python main.py -c config_author_control
../../../.venv-tpne/bin/python extract_metrics.py ../result/config_author_control

# ours - offline and incremental, same protocol
.venv-tpne/bin/python scripts/compare_tpne.py --mode ours --mode-flag offline     --rounds 3 --trials 3
.venv-tpne/bin/python scripts/compare_tpne.py --mode ours --mode-flag incremental --rounds 3 --trials 3

# IEEE-CIS V-feature A/B (needs the real CSV in data/)
.venv-tpne/bin/python scripts/evaluate.py --limit 40000 --holdout 10000
.venv-tpne/bin/python scripts/evaluate.py --limit 40000 --holdout 10000 --v-features
```