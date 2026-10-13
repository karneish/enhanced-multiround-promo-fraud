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
author.

| Metric | Author (n=5) | This repo (n=3) | Delta |
|---|---|---|---|
| RT-f1₀ initial | 66.56 | 57.25 | −9.31 |
| RT-avg_f1_{t>0} | **57.09** | **53.06** | **−4.03** |
| RT-recall₀ | 27.50 | **92.19** | **+64.69** (3.4×) |

Two findings drove most of the progress, and both were defects rather than
innovations: the adversarial-modification default was wrong (`relative` instead
of the author's `REPLAY`, worth **6.1×** on on-round F1), and the round-0
holdout was being trained on, which faked a +14.75 retention gain. The
pre-training hypothesis — that the three TPNE losses were "measured but not
optimised" and that optimising them would close the gap — was **tested and is
wrong**: it moves on-round F1 by +0.4.

The one metric that cannot be closed by tuning is round-0 performance, and a
feature ablation shows why: our re-implemented TPNE embedding scores **38.45**
on generated fraud where the raw node features score **53.06**. The remaining
work is to use the author's original embedder rather than our port. See §4.2c.

> ### ⚠ Retraction (later than the table above)
>
> A subsequent `--split masks` comparison reported that all four metrics beat
> the paper, headlined by "RT-avg_f1 70.57 vs 57.09". **That was wrong.** The
> masks split had been reduced to contiguous node ranges, which swept **100% of
> the retention set into both training and threshold-tuning**. It was
> train-on-test, and the reported margin was entirely an artifact of it.
>
> After the fix this project **does not beat the paper**: RT-f1₀ 54.89 vs 66.56,
> RT-avg_f1 54.51 vs 57.09. The table in this section was measured on the
> `holdout` split and is unaffected. See §4.2e, §4.2g and §4.2h for the
> diagnosis, the root cause of the 94.39 figure, and the corrected numbers.

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

### 4.2 Our pipeline, same benchmark and protocol — CURRENT

> The figures originally in this section (15.63 / 28.24) were superseded by the
> measurements below. They were taken before two defects were found: a wrong
> adversarial-modification default and a leaked training set.

3 rounds, 3 trials, identical budgets (5 % of pos/neg per round), the author's
own `greedy` / `replay` / `reage` scenario, threshold selected on a held-out
validation split. **Strict parity**: the author's `round_num = 3` run emits query
rows for rounds 1–2 only, so its RT-avg_f1_t>0 is a two-round mean.

| Metric | Author TPNE-XGB (n=5) | This repo (n=3) | Delta |
|---|---|---|---|
| RT-f1₀ initial | **66.56 ± 0.75** | 57.25 | −9.31 |
| RT-f1₀ end (retention) | **66.95 ± 2.53** | 57.22 | −9.73 |
| **RT-avg_f1_{t>0}** | **57.09 ± 2.33** | **53.06** | **−4.03** |
| **RT-recall₀** | **27.50 ± 5.62** | **92.19** | **+64.69** |

Read honestly: **we still do not exceed the author.** Recall is 3.4× theirs and
the on-round F1 gap has closed from 20.26 to 4.03, but round-0 performance and
retention remain behind.

### 4.2a The two defects that dominated the gap

| Defect | Effect | Fix |
|---|---|---|
| `adver_mod` defaulted to `relative` (perturb every feature by 1 σ) when the author's `DEFAULT_ADVER_CONFIG` uses **`REPLAY`** (verbatim duplicate, no perturbation) | **RT-avg_f1 6.37 → 38.72** (6.1×) | default changed to `replay` |
| Round-0 holdout was added to the training set | faked a **+14.75** retention gain | holdout excluded from training |

The support/query split of the paper (§4, Fig. 2) was also missing: every
generated node went to the query set, so the classifier was never trained on
adversarial nodes before being scored on them. Now `G_tQ` is only ever scored and
`G_tS` grows and is trained on.

### 4.2b Gradient pre-training — implemented, measured, and it does NOT help

The original claim here was that the losses were "measured but not optimised"
and that optimising them would close the gap. Both halves of that turned out
differently:

- **Optimising them was implemented** (`TPNEEmbedder.pretrain`, Adam on
  `L_recon + α·L_disent + β·L_temp`, sparse adjacency so it runs at
  11,758 nodes in 9.4 s/epoch at 2.4 GB peak). The objective genuinely
  converges: 40.60 → 0.89 on a toy graph.
- **It does not improve detection.** 60 epochs moved on-round F1 from 38.72 to
  39.12 (+0.4). With `relative` perturbation it was catastrophic — 0.00 F1 in
  round 1, because the encoder was fitted to the round-0 graph and generated
  nodes were out of distribution.

The pre-training hypothesis was **wrong**, and the fix for the round-1 collapse
was the `REPLAY` default, not the gradients.

### 4.2c The remaining blocker: the re-implemented encoder

A feature ablation isolates it:

| Features fed to the ensemble | RT-f1₀ | RT-avg_f1 | RT-recall |
|---|---|---|---|
| raw node features (10) | 47.63 | **53.06** | 92.19 |
| our TPNE embedding (384) | **57.25** | 38.45 | 59.38 |
| both, h_feats=16 | 57.25 | 42.73 | 62.50 |
| both, h_feats=16, pre-trained 60 | 55.98 | 37.24 | 52.34 |

**Our re-implemented TPNE embedding is worse than the raw node features** at
detecting generated fraud (38.45 vs 53.06), and concatenating it *dilutes* the
raw signal rather than adding to it. That is why round-0 and retention lag: both
are set by encoder quality, and the author's original `TemporalEmbedder`
(vendored in `vendor/author-tpne/`) is stronger than ours.

### 4.2d Using the author's original embedder (attempted)

`backend/embedder.py::AuthorTemporalEmbedder` drives the vendored
`vendor/author-tpne/src/models/proposed_supervised/mixed.py` `TemporalEmbedder`
directly. The reference code passes the plain DGL graph as `blocks` (see
`src/models/base_model.py`), so no neighbour sampler is needed. The adapter
adds self-loops where adversarial rewiring has stranded a zero-in-degree node,
which GINConv/GraphConv reject.

Measured, 1 trial:

| Configuration | RT-f1_0 | RT-avg_f1 | RT-recall |
|---|---|---|---|
| author's embedder (via adapter) | 54.79 | 23.82 | 34.38 |
| author's embedder + raw features | 55.39 | 42.72 | 60.94 |
| **raw features only (best)** | 47.63 | **53.06** | 92.19 |

**It does not help, for a specific and fixable reason.** The reference
encoder's quality comes from the training loop in
`vendor/author-tpne/src/experiment/supervised_multi.py`, which optimises the
three losses for up to 500 epochs per round. At the time of this measurement
the adapter provided inference only — `pretrain()` raised
`NotImplementedError` deliberately rather than silently returning an untrained
embedding. Fed an untrained encoder it scores *worse* than the raw node
features (23.82 vs 53.06).

> ### Correction (this paragraph was stale)
>
> `AuthorTemporalEmbedder.pretrain()` **is** implemented now — it drives the
> vendored `TemporalEmbedder.forward`, whose `(L_recon + a·L_disent +
> b·L_temp)/(1+a+b)` loss is fully differentiable, and it accepts `patience`
> to mirror the reference's `early_stopping = 75`. So the encoder gap is
> closable by running the reference loop, and that has been done.
>
> A second, larger bug was found alongside it. `scripts/upgraded_tpne.py`
> guarded pre-training with `embedder != "author"`, a guard that predated the
> implementation. Its effect was that **`--embedder author` silently ran the
> reference encoder untrained**, so every earlier author-embedder number was
> measured against a randomly-initialised encoder. That guard has been removed;
> pre-training now runs for whichever embedder is selected and falls back with
> a warning if unsupported. Re-measure before quoting any number in this
> subsection.

So the encoder gap was never closable by *calling* the reference module; it
required running the reference *training loop* around it — which is now what
happens. Whether 500 epochs is enough is the open question, answered by the
`refsched_author_*` runs.

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

### 4.2e RETRACTION — the masks split was training on the test set

This section previously reported that under `--split masks` all four metrics
exceeded the paper, headlined by "RT-avg_f1 70.57 vs 57.09, +13.48". **That
result was invalid and is retracted.** It was caused by a defect in the split,
not by any property of the model.

The author's predefined mask columns select an arbitrary, **non-contiguous**
subset of nodes. The pipeline consumed them as contiguous ranges:

```python
split      = i_tr.max() + 1                       # 11752
val_slice  = slice(i_va.min(), i_va.max() + 1)    # 1..11757
head.fit(X0[:split], y_all[:split])
```

Because the masks are sparse and spread over the whole node range, `i_tr.max()`
lands at 11751. So the model was trained on nodes 0..11751 and the threshold
was tuned on nodes 1..11757. Measured overlap with the 5,879-node retention set:

| Slice | Overlap with retention set |
|---|---|
| training `range(11752)` | **5,877 / 5,879 (100.0%)** |
| validation `range(1, 11758)` | **5,879 / 5,879 (100.0%)** |
| real train index array | 0 (disjoint) |
| real validation index array | 0 (disjoint) |

That is train-on-test, and it also meant the "validation" F1 used to pick the
threshold contained the entire test set. The leakage was invisible because a
high-dimensional random projection lets the trees interpolate the memorised
rows far better than 10 raw features do, which is exactly why the corruption
was largest for `--features embed`.

Fixed by consuming the masks as index arrays (`X0[train_idx]`,
`X0[val_idx]`), matching what the `dual` and `view-switch` branches already
did. `tests/test_split.py` pins the behaviour down from three angles.

### 4.2f Scenario parity: a REPLAY clone must inherit its seed's edges

The last structural gap against the reference scenario. The author's
`BaseAdversarialChoice.duplicate_nodes` copies the seed's in/out edges and
remaps the endpoints onto the clone:

    in_dst  = id_dict[in_dst]
    out_src = id_dict[out_src]

so a REPLAY clone stands in for its seed *in the graph*. This port cloned
features only, and `_rewire` returned early in replay mode, leaving every
generated node **isolated** — degree 0 against an original mean of ~357 on
`tolokers_bid`. That is not the same benchmark: the encoder sees a detached
node where the reference sees a normal one.

Fixed. After the fix, query nodes have mean degree 378.7 against an original
mean of 382.4, with zero isolated nodes. Three tests lock it in.

This is a genuine correctness fix, and it is worth keeping regardless of what
the scoreboard says. It is **not** what carried any headline number: the
previously-quoted raw-feature configuration reads the 10 raw node feature
columns and never touches the graph, so edge inheritance cannot affect it.

### 4.2g The 94.39 round-0 figure is root-caused — it was the same leakage

This figure was previously reported as "unexplained, not claimed". It is now
explained, and the explanation is the bug in §4.2e: with ~100% of the retention
set in the training slice, an embedding that ranks at AUC 0.83 in a clean
setting scores 0.94 F1 when those same rows were trained on.

The evidence that ruled out the alternatives:

- **Not label leakage in the features.** A label-permutation control collapses
  every view to chance, so the encoder does not encode `y`:
  embed AUC 0.5008, raw 0.4934, both 0.5024 (against 0.8246 / 0.7302 / 0.8300
  with real labels).
- **Not a threshold artifact.** Validation and test prevalence both equal
  0.218, and val/test F1 track each other across the threshold sweep.
- **Not a lucky draw.** The pipeline reproduced ~94 across five separate
  processes while every faithful reimplementation returned 0.52-0.55.

The reimplementations were correct all along; they differed from the pipeline
in one respect only — they used `X0[train_idx]` instead of `X0[:split]`. After
the fix the pipeline agrees with them: **54.89**, not 94.

### 4.2h Corrected results — we do not beat the paper

Re-measured on the paper's masks protocol, n=3, after the split fix:

| Metric | Author TPNE-XGB (n=5) | This repo, raw (n=3) | This repo, embed (n=3) |
|---|---|---|---|
| RT-f1₀ start | 66.56 ± 0.75 | 46.49 ± 0.34 | 54.89 ± 0.82 |
| RT-avg_f1_{t>0} | **57.09 ± 2.33** | 51.78 ± 1.29 | 54.51 ± 4.32 |
| RT-recall₀ | 27.50 ± 5.62 | 84.11 ± 2.42 | 78.13 ± 2.78 |

**The honest position is that this project does not beat the paper.** It loses
round-0 F1 by 11.7 points, and the RT-avg_f1 gap (2.6 points) is within about
one standard deviation — i.e. not a win. Recall is much higher, but that is a
threshold-driven precision/recall trade, not a capability: pushing the
threshold down to 0.25-0.35 is what produces it.

Two further measured facts bound what is achievable here:

- **Training the encoder is not the missing piece.** Driving the author's own
  `TemporalEmbedder` through its own `L_recon`/`L_dent`/`L_temp` loss for 100
  epochs converges the loss from 0.571 to 0.235 while round-0 F1 stays flat at
  ~0.53 and AUC moves 0.8045 -> 0.8103 (`scripts/embedder_training_curve.py`).
- **The embedding has an information ceiling.** Its ranking quality is pinned
  at AUC ~0.82 regardless of classifier or class balancing; reaching the paper's
  F1 of 0.6647 needs roughly AUC 0.90. Their `train_predictor` uses
  `scale_pos_weight` with early stopping; replicating that exactly (sweeping
  spw 1-5) left AUC at 0.814-0.820 and F1 at 0.43-0.49.

So the ported encoder is genuinely weaker than the authors', and that — not
thresholding, not class weighting, not pre-training budget — is what separates
us from the paper.

The `--features raw` ablation and the §4.1 control numbers were produced on the
`holdout` split, which is contiguous and was never affected; §1 still stands.

### 4.2i A second instance of the same defect, in the dual-view branch

Auditing the rest of the benchmark for the §4.2e pattern turned up one more.
`_dual_rounds` received the retention set through a parameter named `split` and
indexed it as a slice endpoint:

```python
ret = head.predict(X0_raw[split:], X0_emb[split:], head.threshold, rw)
"f1_round0_nodes": _metrics(y_all[split:], ret)[0],
```

The caller passes `test_idx`, an **index array**. `X0_raw[test_idx:]` starts at
`test_idx.min()` and runs to the end of the node range, so under `--split masks`
the round-by-round retention metric re-admitted the training nodes every round.
Under `holdout` the same expression happened to be correct, because `test_idx`
there begins exactly at the split point — which is why the bug survived
alongside the §4.2e one.

Fixed to index by `test_idx`. `tests/test_split.py` now asserts no
`[split:]`/`[split]` indexing remains anywhere in the benchmark, and that the
dual branch scores retention on `y_all[test_idx]`.

Re-measured on the masks protocol after both fixes:

| Arm | RT-avg_f1_{t>0} | RT-recall₀ |
|---|---|---|
| author TPNE-XGB (n=5) | 57.09 ± 2.33 | 27.50 ± 5.62 |
| single view, embed (n=3) | 54.51 ± 4.32 | 78.13 ± 2.78 |
| single view, raw (n=3) | 51.78 ± 1.29 | 84.11 ± 2.42 |
| dual view (n=1) | 20.71 | 22.66 |

The dual-view head is not competitive on this protocol and is not carried as a
result. `compare_tpne.py` was checked in the same pass and is clean: its
`split = int(0.6 * len(y))` is a genuine contiguous range.

**Audit result: no remaining instances.** The only mask-derived indices in the
non-vendored code are the two mask reads inside `author_split` itself, which are
consumed as boolean arrays and returned as index arrays.
