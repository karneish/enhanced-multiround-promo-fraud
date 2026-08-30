# Adaptive Defensive Layer (ADL) — Architecture

This project simulates a **promo-referral marketplace** under continuous attack.
Real users invite friends and earn rewards; fraudsters create fake identities,
fake phones and fake IPs to steal those rewards. Each round the attacker gets
*better*, so the defense has to get better too. The result is a looping game:

```
World ──▶ Features ──▶ Detector ──▶ ADL (risk → Allow / Review / Block)
                                          │
            ┌─────────────────────────────┘
            ▼
        blocked fraud = soft-removed (never seen by the attacker again)
            │
            ▼
        attackers only study the fraud that SURVIVED the defense
            │
            ├──────▶ GAN / probabilistic model of survivors
            │                       │
            └── evolved new fraud + rings/victims ──▶ World (next round)
```

---

## The one golden rule

> **The attacker can only learn from the fraud that got away.
> Fraud that got caught is erased from the world before the attacker looks.**

Blocked fraud is *soft-removed* — the account object stays in memory so
indexes never shift, but it is flagged `blocked` and excluded from every
feature, metric, training split and generator seed (`backend/world.py:299-318`).
This is what forces real evolution: the attacker cannot copy a trick that no
longer exists; it must invent new ones.

---

## The simple story

Think of it as a shop with a "refer a friend, get money" program.

- **The world** is the shop: a list of customers (accounts) and how they are
  connected (who referred whom, which phone, which IP).
- **The camera** turns every customer into a row of 17 numbers the model can
  read (behaviour + graph signals).
- **The guard** (detector) reads those numbers and gives every customer a
  fraud score from 0 to 1.
- **The security chief** (ADL) combines the guard's score with two human-style
  clues ("is this person central in the referral web?" and "does this person sit
  inside a suspicious ring?") into one **risk score**, then decides:
  **let pass / review / block**. Blocked = removed from the books forever.
- **The crime boss** (generator) collects *all* the fraud that escaped,
  learns the pattern of it (with a GAN or a statistical model), and creates
  *new* fraud that is similar-but-different, plus referral rings and previously
  attacked victims. Old tricks are dead; only surviving tricks get cloned and
  mutated.

After 8 rounds the "overall performance" tells you how good your defense is
against attack patterns it could never have seen in round 0.

---

## Components

### 1. World — the synthetic marketplace (`backend/world.py`)

- Every actor is an `Account` (`world.py:106`) with **11 intrinsic attributes**
  (`world.py:29-33`): `age`, `email_disposable`, `phone_verified`,
  `device_fresh`, `ip_proxy`, `loc_entropy`, `login_night`, `amount_mean`,
  `amount_std`, `txn_count`, `txn_freq` — plus a binary label (0 genuine / 1 fraud).
- Accounts are connected by three relation types (`world.py:143-149`):
  **referral** (who invited whom), **device** (shared handsets), **ip**
  (shared addresses).
- Round-0 fraud comes from **5 attack templates** (`world.py:64-100`):
  `fake_identity`, `referral_farming`, `device_spray`, `vpn_hop`,
  `quiet_sampler`. Each template is a target feature profile plus
  `ring_affinity` / `device_spray` / `ip_reuse`, so even the initial fraud
  already forms rings and reuses devices/IPs.
- Everything is seeded (`world.py:139-141`): the same config always reproduces
  the same world, so experiments are comparable.
- **Soft removal** — `block(idx)` (`world.py:299`) sets `blocked=True`;
  `active_mask()` (`world.py:313`) therefore hides the account from every
  downstream step. Indexes never shift.

### 2. Features — world → numeric matrix (`backend/features.py`)

- `compute_features(world, supervised_mask)` (`features.py:44`) converts the
  world into an `(n_accounts, 17)` matrix: the 11 intrinsic values (clamped to
  [0,1]) plus **6 graph features** (`world.py:35-38`):
  `referral_count`, `degree`, `clustering` (local clustering coefficient),
  `shared_device`, `shared_ip`, `fraud_neighbor_ratio`.
- Two anti-cheat guarantees:
  - `fraud_neighbor_ratio` uses **only labels the system was actually given**
    (`known_fraud = supervised_mask & label == 1`, `features.py:57,97-103`) —
    never the hidden truth.
  - Blocked accounts contribute nothing: they are dropped from every adjacency
    list and get an all-zero row (`features.py:47-49,109-111`). Defense
    performance is measured on **live** accounts only.
- `intrinsic_matrix()` (`features.py:125`) is the behaviour-only variant handed
  to the generator (the attacker does not get the defense's graph statistics).

### 3. Detector — predicts fraud (`backend/detector.py`)

- A thin XGBoost wrapper (`detector.py:65-79`): 180 trees, depth 4,
  `scale_pos_weight = neg/pos` to balance the imbalanced classes.
- `best_threshold()` (`detector.py:16-28`) grid-searches 0.05 → 0.95 (37 steps)
  on the validation split, picking the cutoff that maximizes **macro-F1**.
- `evaluate()` (`detector.py:31-49`) returns recall, precision, macro-F1, AUC
  and the confusion matrix (TP/FP/FN/TN). The **FN's** are exactly the accounts
  that become the attacker's training material next round.

### 4. ADL — the defense layer itself (`backend/adl.py`)

The decision engine between the detector and the generator. For every live
account it fuses five signals into one **risk score**

```
R = w1·P_f + w2·C + w3·S + w4·V + w5·A          with  sum(w) == 1
```

| signal | meaning | default weight |
|---|---|---|
| `P_f` | fraud probability from the detector | 0.45 |
| `C`   | graph centrality (degree, min-max over active) | 0.20 |
| `S`   | fraud-ring participation (suspicious neighbours ÷ degree) | 0.15 |
| `V`   | transaction velocity (txn_count × txn_freq, min-max) | 0.10 |
| `A`   | account trust (1 − age) | 0.10 |

(`adl.py:13-45`; components computed in `compute_risk_components()`, `adl.py:77-136`;
`S` treats as suspicious anything the detector flagged *or* that is supervised-known fraud,
`adl.py:97-98`.)

Decision regions with two adaptive thresholds (`adl.py:197-200`):

```
R <  T1          → Allow
T1 <= R < T2     → Review   (review_catch_rate fraction of reviewed fraud is caught)
R >= T2          → Block    (soft-removed; never reaches the attacker)
```

- Defaults: `T1 = 0.40`, `T2 = 0.75`, `alpha = 0.05`, `review_catch_rate = 0.5`
  (`adl.py:47-50`).
- **Threshold adaptation** (`adl.py:351-367`):

  ```
  T_{r+1} = T_r + alpha · (false_block_rate − escape_rate)
  ```

  Too much fraud escaping → delta negative → thresholds drop → stricter defense.
  Too many innocent users blocked → delta positive → thresholds rise → more
  lenient. `T1` and `T2` are clamped so they always stay ordered and in range.
- **Decision latency** is simulated (`adl.py:55`, `278-290`): allow 1 ms,
  review 40 ms, block 5 ms per transaction.
- **Defense metrics** recorded per round (`adl.py:247-346`): escape rate,
  false-block rate, defense precision/recall, block rate, review rate, risk
  distribution (mean/median/p75/p95/max), a 20-bin risk histogram, and the
  used weights.

### 5. Generator — the attacker that learns (`backend/generator/`)

**Memory (`generator/profile.py`)** — `FraudProfile` is a *rolling window*
(default 5 rounds, config `profile_window`) of the fraud that **escaped**.
Each round it receives the survivors' feature rows, strategy names and which
accounts were attacked (`profile.py:35`). This is the *only* data the attacker
ever sees. It also keeps the `familiar` set of previously attacked accounts
(capped at 4000, `profile.py:62`).

**Engines (`generator/engine.py`)**:

| engine | behaviour |
|---|---|
| `IntelligentFraudGenerator` (default) | learns a **GAN** on the survivor pool (falls back to a probabilistic model if < 2 survivor samples, `engine.py:118`) and synthesises *new* strategies |
| `ReplayGenerator` | baseline: exact copies of the missed fraud, no learning (`engine.py:45-79`) |

**Synthesis pipeline (intelligent, `engine.py:136-189`)**:

1. `_learn` — train the GAN on the survivor pool (`generator/gan.py`:
   generator MLP `noise→[0,1] behaviour`, discriminator real/fake; Adam lr 1e-3,
   BCEWithLogits; returns `None` for too few samples).
2. `_sample` — draw fresh behaviour vectors from the trained GAN, or the
   probabilistic fallback.
3. `sample_probabilistic` (`generator/sampler.py:16`) — seeded row (or the pool
   mean, 50% of the time) + Gaussian drift scaled by the pool's per-feature
   std, clamped to [0,1]. `diversity > 1.0` pushes samples further from the
   source distribution (`sampler.py:37`).
4. `mutate_spec` (`generator/mutators.py:39`) — compares each new vector to its
   parent and tags the changes: `new_device`, `vpn`, `new_email`,
   `amount_shift`, `timing_shift`, `same_device`, `same_ip`… (`mutators.py:51-80`).
   The strategy name becomes `parent+tag1+tag2`, so the evolution of attack
   behaviour is traceable round after round.
5. `build_structure` (`generator/mutators.py:99`) — a `ring_ratio` fraction of
   new fraud is wired into **referral rings/chains among themselves**; the rest
   attach to **familiar victims** (previously attacked accounts). The edges are
   injected into the world via `world.add_fraud` / `world.add_referral`.

### 6. Simulation — the orchestrator (`backend/simulation.py`)

`Simulation` (`simulation.py:128`) runs the whole loop on a background thread and
streams every event to the frontend. Defaults (`simulation.py:30-63`): 8 rounds,
seed 7, 500 base accounts / 60 initial fraud, +45 genuine / +30 fraud per round,
`supervised_ratio = 0.25`, `forget_window = 2`, `budget_pos = 6`,
`budget_neg = 15`.

**One round, step by step** (`_step_round`, `simulation.py:254-293` + pipeline):

1. Add `genuine_per_round` genuine users (`world.add_genuine`).
2. Generator synthesises `fraud_per_round` new fraud specs + edges
   (`generator.generate`) → `world.add_fraud` (`simulation.py:267-277`).
3. **Reveal a label budget** (`_reveal_budget`, `simulation.py:499-519`) —
   simulates manual review cost: up to `budget_pos` fresh fraud and
   `budget_neg` fresh genuine accounts get their *true label revealed* and
   become training data.
4. **Retrain + predict** (`_retrain_and_predict`, `simulation.py:304-337`):
   retrain the detector on accounts that are active, supervised, and recent
   (`creation_round >= current − forget_window`); blocked accounts are gone.
5. **ADL defense** (`_run_defense`, `simulation.py:339-357`): score the whole
   world, make Allow/Review/Block decisions, apply blocks.
6. **Evaluate** (`_finish_round`, `simulation.py:359-418`): overall metrics over
   all active accounts plus fresh-only metrics.
7. **Collect survivors** (`_collect_missed`, `simulation.py:429-455`) —
   `active & supervised & fraud` are the fraud that survived everything
   (blocked/review-caught fraud was already soft-removed). Their features,
   strategy names and confidences, plus their neighbours as `familiar` victims,
   go into `generator.analyze()` → the attacker learns → next round.

After the last round the profile summary, per-round records, threshold history
and strategy evolution are all exposed through `report()` (`simulation.py:524-540`).

---

## Metrics — "overall performance"

Per round, `evaluate()` computes over the **active** population
(`simulation.py:365-373`):

- **macro-F1**, **recall**, **precision** and **AUC** of the detector,
- confusion matrix (TP / FP / FN / TN) and the chosen threshold,
- fresh-only metrics restricted to accounts created that round.

The ADL adds defense-level metrics per round (block/escape/false-block rates,
precision/recall, risk distribution, simulated latency). The **whole-run**
picture is the series of these per-round values plus the final generator profile
(how the attacker's strategies and diversity evolved), which is what the ADL
dashboard plots.

---

## Configuration reference (`simulation.py:30-63`)

| key | default | meaning |
|---|---|---|
| `rounds` | 8 | number of adversarial rounds |
| `seed` | 7 | world / GAN / ADL randomness (reproducibility) |
| `base_accounts` | 500 | genuine accounts in round 0 |
| `initial_fraud` | 60 | fraud accounts in round 0 |
| `genuine_per_round` | 45 | new genuine users every round |
| `fraud_per_round` | 30 | new fraud accounts every round |
| `generator_mode` | `intelligent` | `intelligent` (learns) or `replay` (clones) |
| `gen_type` | `GAN` | `GAN` or `PROB` synthesizer |
| `gan_epochs` | 120 | GAN training epochs |
| `gan_noise_dim` | 12 | GAN latent dimension |
| `gan_hidden` | 32 | GAN hidden layer size |
| `diversity` | 1.0 | push generated behaviour away from source (>1) |
| `conn_coef` | 0.6 | how many familiar victims each attacker attaches to |
| `ring_ratio` | 0.5 | fraction of new fraud wired into rings/chains |
| `profile_window` | 5 | rounds of survivors the attacker remembers |
| `supervised_ratio` | 0.25 | fraction of the world the system gets labeled |
| `forget_window` | 2 | detector retrains only on accounts created in the last N rounds |
| `budget_pos` | 6 | fraud labels revealed per round (manual review cost) |
| `budget_neg` | 15 | genuine labels revealed per round |
| `adl_enabled` | `True` | toggle the ADL decision layer on/off |
| `threshold_policy` | `adaptive` | `adaptive` or `fixed` thresholds |
| `t1` / `t2` | 0.40 / 0.75 | initial decision thresholds |
| `threshold_alpha` | 0.05 | adaptation step size |
| `review_catch_rate` | 0.5 | share of reviewed fraud that is caught |
| `w_pf` / `w_centrality` / `w_ring` / `w_velocity` / `w_trust` | 0.45 / 0.20 / 0.15 / 0.10 / 0.10 | risk weights (renormalised) |

---

## File map

```
adaptive-defensive-layer/
├── run.py                 # dashboard entry point → http://127.0.0.1:5050 (or $PORT)
├── backend/
│   ├── world.py           # marketplace: accounts, referral/device/IP graph, soft removal
│   ├── features.py        # world → 17-D feature matrix (no label cheating)
│   ├── detector.py        # XGBoost fraud score + threshold + evaluate()
│   ├── adl.py             # risk score → Allow/Review/Block + adaptive thresholds
│   ├── simulation.py      # the round orchestrator (attacker–detector–defense loop)
│   ├── app.py             # Flask API (start/stop simulation, stream events)
│   └── generator/
│       ├── engine.py      # Intelligent vs Replay generator
│       ├── profile.py     # rolling memory of surviving fraud + familiar victims
│       ├── gan.py         # GAN synthesis of behaviour vectors
│       ├── sampler.py     # probabilistic fallback + diversity
│       └── mutators.py    # mutation tags + ring/victim structure
└── frontend/              # streaming dashboard (per-round charts)
```

Run it: `python run.py` → open `http://127.0.0.1:5050`. When launched through
the root `run_all.py` launcher the project instead serves on port **5052**.