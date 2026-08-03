# Enhanced Multi-Round Promo Fraud Detection

A multi-round adversarial framework for graph-based **promotional fraud detection** with an
adaptive **Intelligent Fraud Generator** and an interactive experiment **dashboard**.

This repository is an enhanced fork of the code accompanying the paper:

> **"A multi-rounded adversarial scenario for graph-based promo fraud detection"**,
> Prasetya, Liu, Murata, and Matono, *Social Network Analysis and Mining*, Springer, 2025.

The core contribution of this project is the **Intelligent Fraud Generator** (`INTELLIGENT`
adversary modifier), which replaces the naive copy-based fraud generation of the original paper
with a GAN / probabilistic generator that adapts across rounds, plus a web dashboard for
launching experiments, inspecting datasets, and visualizing per-round results.

---

## Table of Contents

- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
  - [Notebook](#notebook)
  - [Command line](#command-line)
  - [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Intelligent Fraud Generator (Add-on)](#intelligent-fraud-generator-add-on)
- [Citation](#citation)
- [License](#license)

---

## Highlights

- **Multi-round adversarial scenario** — a detector is re-trained every round against an
  adversary that keeps generating new fraud, mimicking real-world campaign dynamics.
- **Intelligent Fraud Generator** — GAN or probabilistic generative adversary that learns from
  the fraud that *escaped detection* and produces fresh, diverse, ring-structured fraud each
  round (instead of duplicating the previous round's missed fraud).
- **Interactive dashboard** — a Flask + Chart.js web UI to configure and launch experiments,
  inspect dataset statistics and sub-graphs, and stream live per-round metrics.
- **Flexible stack** — pluggable models (GCN, GCNII, GraphSAGE, GIN, GAT, BWGNN, GHRN, XGBoost,
  hybrid XGBoost-GNN `XGB-SP`), adversary choosers, modifiers, augmentation strategies, and
  prediction add-ons.

## Repository Structure

```
.
├── README.md
├── setup.py                      # Package metadata and dependencies
├── run_example.sh                # Example cluster (Univa/UGE) launcher
├── scripts/
│   ├── main.py                   # Entry point: run an experiment from a JSON config
│   ├── config_example.json       # Full-featured example config
│   ├── config_cpu.json           # CPU smoke-test config (tolokers_bid)
│   └── config_intelligent.json   # INTELLIGENT fraud-generator smoke test
├── src/
│   ├── experiment/supervised_multi.py   # MultiroundExperiment orchestration
│   ├── models/                          # GNN / boosting backbones and proposed models
│   ├── meta_strategies/                 # Augmentation and prediction add-ons
│   ├── adversary/
│   │   ├── choose/                      # Seed-selection strategies
│   │   └── modify/                      # Fraud-generation modifiers (incl. INTELLIGENT)
│   └── utils/                           # Configs, helpers, notebook utilities
├── dashboard/                    # Flask dashboard (app.py, static/, templates/)
├── notebook/                     # Jupyter examples (run & data processing)
└── dataset/                      # DGL graph datasets (see Datasets)
```

## Requirements

Developed and tested with Python 3.11 and the following library versions:

```
dgl==2.0.0+cu121
networkx==3.1
numpy==1.24.3
pandas==2.2.1
scikit-learn==1.3.0
scipy==1.12.0
seaborn==0.12.2
torch==2.2.2
torch_geometric==2.5.3
xgboost==2.0.3
```

## Installation

```bash
python setup.py install
```

or, from the project root, simply ensure `src/` is importable (the scripts add it to `sys.path`
automatically).

## Datasets

Place each dataset in `dataset/` in a DGL-readable format (`.bin` graph dump), as illustrated by
the hosted `tolokers_bid` dataset.

- `dataset/tolokers_bid` — hosted in this repository (Tolokers, node-level labels).
- Other datasets used in the paper (`tfinance`, `yelp`, `amazon`) are not bundled here due to
  size / licensing; see the original repository for the corresponding author contact.

## Usage

### Notebook

Run `notebook/example_experiment.ipynb` with a kernel that satisfies the requirements above.
`notebook/example_data_process.ipynb` demonstrates how to interpret and process the experiment
results.

### Command line

```bash
cd scripts
python main.py -c <config_name_without_json>
```

Example:

```bash
python main.py -c config_cpu            # quick CPU smoke test
python main.py -c config_intelligent    # INTELLIGENT fraud generator smoke test
```

### Dashboard

Launch the interactive experiment dashboard:

```bash
cd dashboard
python app.py
```

Then open `http://127.0.0.1:5050`. From the dashboard you can:

- Inspect dataset statistics and an interactive sample of the graph (nodes/edges, labels).
- Configure and launch experiments (models, augmentations, adversary, hyper-parameters).
- Stream live console output and per-round metrics (Rec / Prec / F1 / AUC).
- Browse completed experiments and their result CSVs.

> **Note:** the dashboard launches experiments in a subprocess using the current Python
> interpreter. The `scripts/run_*.json` files it generates are runtime artifacts and are
> git-ignored.

## Configuration

All experiment settings live in a `.json` file inside `scripts/`. See
`scripts/config_example.json` for a full example. The config file contains:

| Key                 | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `TRIAL_NUM`         | Number of repetitions per configuration                            |
| `FAILURE_LIMIT`     | Allowed consecutive failures before aborting                       |
| `EXPERIMENT_DESC`   | Human-readable description (written to `meta.txt`)                 |
| `LIST_DSET`         | Datasets to evaluate                                                |
| `LIST_TRAIN_DSET`   | Optional pre-training graph per dataset (`"NONE"` to skip)         |
| `EXP_DICT`          | Cartesian search space of hyper-parameters                         |

Every key listed in the `EXP_DICT` is applied onto the corresponding default config
(`DEFAULT_MAIN_CONFIG`, `DEFAULT_TRAIN_CONFIG`, `DEFAULT_MODEL_CONFIG`,
`DEFAULT_STRAT_CONFIG`, `DEFAULT_ADVER_CONFIG`). The full list of available keys and defaults is
defined in `src/utils/utils_const.py`.

Available choices include:

- **Models** (`model_name`): `GCN`, `GCNII`, `GraphSAGE`, `GIN`, `GAT`, `BWGNN`, `GHRN`, `XGB`,
  `XGB-SP`.
- **Augmentation** (`augment_name`): `NONE`, `RANDOM`, `REAGE`.
- **Adversary chooser** (`adver_choose_name`): `RANDOM`, `GREEDY`, `OGREEDY`.
- **Adversary modifier** (`adver_mod_name`): `REPLAY`, `PERTURB-ABS`, `PERTURB-REL`, `MIXING`,
  `INTELLIGENT`.
- **Prediction add-on** (`addon_name`): `NONE`, `FTHR`, `AFTHR`, `DEGREE`, `DFEAT`, `DAFEAT`.

## Outputs

All outputs are written under `result/`. Each experiment gets its own folder named after the
execution timestamp. Generated files include:

- `meta.txt` — experiment metadata including the actual values of all configurable parameters.
- `[dataset]-[exp_dict_item_1]-...-[exp_dict_item_n]-E.csv` — per-configuration run results.
- `combined_result.csv` — combined results of all runs spanned by the `EXP_DICT`.

The notebook `notebook/example_data_process.ipynb` shows how to interpret and process these
result files.

## Intelligent Fraud Generator (Add-on)

The original paper generates the next round of fraud by **copying** the fraud that escaped
detection (`REPLAY`) or by lightly perturbing it (`PERTURB-ABS`, `PERTURB-REL`, `MIXING`). This
add-on replaces that simple copy step with an **Intelligent Fraud Generator**
(`adver_mod_name = INTELLIGENT`) so the simulated attacker behaves the way real fraudsters do: it
observes what worked, adapts, and launches a *new* strategy instead of the same one again.

### How it works, per round

1. **Analyze** — a rolling profile of every fraud seed that escaped detection is maintained
   (features, connection degrees, familiar target nodes, and how confident the detector was).
2. **Learn / evolve** — either a small **GAN** (`adver_gen_type = GAN`) or a **probabilistic**
   model (`adver_gen_type = PROB`) is trained over the successful-fraud features. The learned
   generator is carried across rounds and warm-started, so it keeps evolving as new fraud
   succeeds.
3. **Generate** — instead of duplicates, brand-new feature vectors are sampled (new devices,
   transaction amounts, timings, ...) and new connection patterns are built: **fraud rings /
   referral chains** among the new nodes plus rewiring onto the familiar targets that were
   previously attacked.
4. **Log** — per-round diagnostics are written into the results CSV under the `gen_*` columns
   (`gen_feat_div` = feature diversity, 0.0 means identical copies; `gen_feat_shift` = shift vs.
   the seed, 0.0 means exact replay; `gen_ext_edges` / `gen_ring_edges` = external vs. ring
   edges; `gen_ring_ratio` = fraction of ring edges; `gen_missed_conf` = how confident the model
   was that the seed was genuine).

### Configuration keys (in `EXP_DICT`)

| Key                     | Default | Description                                              |
|-------------------------|---------|----------------------------------------------------------|
| `adver_mod_name`        | `REPLAY`| Set to `INTELLIGENT` to enable the add-on                |
| `adver_gen_type`        | `GAN`   | `GAN` (neural) or `PROB` (probabilistic)                 |
| `adver_gen_epochs`      | `300`   | GAN training epochs per round                            |
| `adver_gen_noise_dim`   | `16`    | GAN latent dimension                                     |
| `adver_gen_hidden`      | `64`    | GAN hidden layer size                                    |
| `adver_gen_feat_coef`   | `1.0`   | Extra feature-diversity multiplier (>1 adds more noise)  |
| `adver_gen_conn_coef`   | `0.5`   | Connection budget vs. the seed's degree                  |
| `adver_gen_ring_ratio`  | `0.5`   | Fraction of new nodes forming fraud rings / referral chains |
| `adver_gen_round_window`| `5`     | How many rounds of successful fraud to learn from        |

### Example config

```json
"EXP_DICT": {
    "adver_mod_name": ["INTELLIGENT"],
    "adver_gen_type": ["GAN"],
    "adver_gen_epochs": [300],
    "adver_gen_conn_coef": [0.5],
    "adver_gen_ring_ratio": [0.5]
}
```

A ready-to-run CPU smoke test is provided in `scripts/config_intelligent.json`
(`python main.py -c config_intelligent`). The dashboard exposes the full generator panel when
`INTELLIGENT` is selected as the adversary modifier.

### Why it is an advantage over the simple generator

- **No more duplicates.** The simple generator replays the same missed fraud next round, so the
  detector only learns to re-see known patterns. The intelligent generator produces fresh feature
  vectors and new graph structures every round.
- **The attacker adapts.** Because the generator is updated from each round of *successful* fraud,
  it drifts towards strategies the current detector is bad at — the exact signal the retrained
  detector needs to see.
- **Richer supervision.** The growing graph now contains original fraud + AI-generated fraud +
  genuine users, so each retrain teaches the model to recognize unseen and future fraud patterns.
- **Fully observable.** `gen_*` diagnostics are logged per round, so you can verify the generator
  is actually producing diverse, shifted, ring-structured fraud instead of copies.

## Citation

```bibtex
@article{prasetya2025multi,
  title={A multi-rounded adversarial scenario for graph-based promo fraud detection},
  author={Prasetya, Hafizh Adi and Liu, Xin and Murata, Tsuyoshi and Matono, Akiyoshi},
  journal={Social Network Analysis and Mining},
  year={2025},
  publisher={Springer}
}
```

## License

MIT license — see `LICENSE`. Code derived from the original
[multiround-promo-fraud](https://github.com/hafizhadi/multiround-promo-fraud) repository,
© 2025 National Institute of Advanced Industrial Science and Technology (AIST).
