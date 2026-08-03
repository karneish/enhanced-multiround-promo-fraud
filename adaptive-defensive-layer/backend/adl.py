"""Adaptive Defensive Layer (ADL).

The ADL turns the framework from a *detector-vs-attacker* loop into a full
*attacker - detector - defense* ecosystem. It is the decision engine that sits
between the graph detector and the generator:

    detector -> P_f  +  graph metrics  ->  risk score  ->  Allow / Review / Block

Only fraud that survives the defense ever becomes training material for the
intelligent generator -- blocked fraud is soft-removed from the world and can
never reach the attacker again.

Risk score (weighted combination of five fraud indicators)::

    R = w1*P_f + w2*C + w3*S + w4*V + w5*A          with  sum(w) == 1

    P_f  fraud probability predicted by the detector
    C    graph centrality (degree centrality, min-max normalised)
    S    fraud-ring participation (share of edges to suspicious nodes)
    V    transaction velocity (txn_count x txn_freq, min-max normalised)
    A    account trust score (1 - age/age_max; older accounts trust = lower)

Decision regions are defined by two adaptive thresholds::

    R <  T1          -> Allow
    T1 <= R < T2     -> Review   (manual investigation, ``review_catch_rate``
                                  of reviewed fraud is caught, the rest passes)
    R >= T2          -> Block

Threshold adaptation (intended interpretation): if too much fraud escapes the
defense tightens; if too many genuine users are blocked it loosens.

    T_{r+1} = T_r + alpha * (false_block_rate - escape_rate)
"""

import numpy as np

# default risk weights (sum to 1)
DEFAULT_WEIGHTS = {
    'w_pf': 0.45,          # fraud probability from the GNN / booster
    'w_centrality': 0.20,  # C - graph centrality
    'w_ring': 0.15,        # S - fraud-ring participation
    'w_velocity': 0.10,    # V - transaction velocity
    'w_trust': 0.10,       # A - account trust
}

DEFAULT_T1 = 0.40
DEFAULT_T2 = 0.75
DEFAULT_ALPHA = 0.05
DEFAULT_REVIEW_CATCH_RATE = 0.5

RISK_COMPONENT_NAMES = ['pf', 'centrality', 'ring', 'velocity', 'trust']

# simulated per-decision latency (milliseconds per transaction)
DECISION_LATENCY_MS = {'allow': 1.0, 'review': 40.0, 'block': 5.0}


def normalize_weights(weights):
    w = np.array([
        float(weights.get('w_pf', DEFAULT_WEIGHTS['w_pf'])),
        float(weights.get('w_centrality', DEFAULT_WEIGHTS['w_centrality'])),
        float(weights.get('w_ring', DEFAULT_WEIGHTS['w_ring'])),
        float(weights.get('w_velocity', DEFAULT_WEIGHTS['w_velocity'])),
        float(weights.get('w_trust', DEFAULT_WEIGHTS['w_trust'])),
    ], dtype=float)
    w = np.maximum(w, 0.0)
    total = w.sum()
    if total <= 0.0:
        w = np.array(list(DEFAULT_WEIGHTS.values()), dtype=float)
        total = w.sum()
    return w / total


# ---------------------------------------------------------------------------
# Risk feature computation
# ---------------------------------------------------------------------------
def compute_risk_components(world, probs, predicted_fraud, known_fraud):
    """Return per-account risk components over the whole world.

    ``predicted_fraud``  - detector flags (boolean over accounts)
    ``known_fraud``      - labels the system has been given (supervised)

    Blocked accounts are invisible: their risk is forced to zero because they
    were already removed from the system by a previous round.
    """
    n = len(world.accounts)
    active = world.active_mask()

    # ---- referral adjacency over active accounts only ---------------------
    referral_nbrs = [set() for _ in range(n)]
    for s, d in world.referral:
        if active[s] and active[d]:
            referral_nbrs[s].add(d)
            referral_nbrs[d].add(s)

    degree = np.array([len(s) for s in referral_nbrs], dtype=float)
    suspicious = np.logical_or(np.asarray(predicted_fraud, dtype=bool),
                               np.asarray(known_fraud, dtype=bool))

    # ---- C: graph centrality (degree centrality, min-max over active) -----
    centrality = np.zeros(n)
    mx = degree[active].max() if active.any() else 0.0
    if mx > 0:
        centrality = degree / mx

    # ---- S: fraud-ring participation = suspicious_edges / total_edges -----
    ring = np.zeros(n)
    for i in range(n):
        if not active[i] or degree[i] <= 0:
            continue
        e_sus = sum(1 for nb in referral_nbrs[i] if suspicious[nb])
        ring[i] = e_sus / degree[i]

    # ---- V: transaction velocity (txn_count * txn_freq, min-max) ----------
    raw_v = np.array([
        a.attrs.get('txn_count', 0.0) * a.attrs.get('txn_freq', 0.0)
        for a in world.accounts
    ], dtype=float)
    velocity = np.zeros(n)
    mxv = raw_v[active].max() if active.any() else 0.0
    if mxv > 0:
        velocity = raw_v / mxv

    # ---- A: account trust = 1 - age/age_max (older -> less risk) ----------
    age = np.array([
        float(np.clip(a.attrs.get('age', 0.5), 0.0, 1.0)) for a in world.accounts
    ], dtype=float)
    trust = 1.0 - age

    pf = np.zeros(n)
    pf[active] = np.asarray(probs, dtype=float)[active]

    return {
        'pf': pf, 'centrality': centrality, 'ring': ring,
        'velocity': velocity, 'trust': trust,
    }


def risk_score(components, weights):
    """Weighted combination R = w1*P_f + w2*C + w3*S + w4*V + w5*A."""
    w = normalize_weights(weights)
    R = (
        w[0] * components['pf']
        + w[1] * components['centrality']
        + w[2] * components['ring']
        + w[3] * components['velocity']
        + w[4] * components['trust']
    )
    return np.asarray(R, dtype=float)


# ---------------------------------------------------------------------------
# The Adaptive Defensive Layer
# ---------------------------------------------------------------------------
class AdaptiveDefense:
    """Stateful decision + adaptation layer.

    Owns the risk weights, the two adaptive thresholds and the defence
    history. ``step()`` performs one full defensive pass over the current
    world: score every account, decide, apply the blocks (soft removal),
    measure the defence and update the thresholds for the next round.
    """

    def __init__(self, weights=None, t1=None, t2=None, alpha=None,
                 policy='adaptive', review_catch_rate=None, seed=0):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()
                                 if v is not None})
        self.t1 = float(t1 if t1 is not None else DEFAULT_T1)
        self.t2 = float(t2 if t2 is not None else DEFAULT_T2)
        self.alpha = float(alpha if alpha is not None else DEFAULT_ALPHA)
        self.policy = str(policy).lower()
        if self.policy not in ('adaptive', 'fixed'):
            self.policy = 'adaptive'
        self.review_catch_rate = float(review_catch_rate
                                       if review_catch_rate is not None
                                       else DEFAULT_REVIEW_CATCH_RATE)
        self.rng = np.random.default_rng(seed)
        self.history = []          # per-round defence records
        self.threshold_history = []  # {round, t1, t2, escape, false_block}

    # ------------------------------------------------------------------ #
    # one defensive pass
    # ------------------------------------------------------------------ #
    def step(self, world, probs, predicted_fraud, known_fraud, labels,
             round_idx, weights=None):
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()
                                 if v is not None})

        components = compute_risk_components(world, probs, predicted_fraud,
                                             known_fraud)
        R = risk_score(components, self.weights)
        active = world.active_mask()

        # ---- decision -------------------------------------------------
        decisions = np.full(len(R), 'allow', dtype=object)
        decisions[R >= self.t2] = 'block'
        decisions[(R >= self.t1) & (R < self.t2)] = 'review'

        # manual review: a fraction of reviewed fraud is caught, the rest passes
        caught = np.zeros(len(R), dtype=bool)
        rev_idx = np.where(active & (decisions == 'review'))[0]
        for i in rev_idx:
            acct = world.accounts[i]
            acct.reviewed = True
            if labels[i] == 1 and self.rng.random() < self.review_catch_rate:
                caught[i] = True

        to_remove = active & ((decisions == 'block') | caught)

        # ---- record per-account decision state -------------------------
        for i in range(len(R)):
            if not active[i]:
                continue
            acct = world.accounts[i]
            acct.decision = str(decisions[i])
            acct.risk = float(R[i])

        # ---- defence metrics (pre-removal, over the active population) --
        record = self._metrics(world, R, decisions, caught, labels, active,
                               round_idx)

        # ---- apply blocks: removed fraud can never reach the attacker ----
        for i in np.where(to_remove)[0]:
            world.block(i, round_idx)

        # ---- adaptive threshold update ---------------------------------
        if self.policy == 'adaptive':
            self._update_thresholds(record['escape_rate'],
                                    record['false_block_rate'], round_idx)
        else:
            self.threshold_history.append({
                'round': int(round_idx), 't1': float(self.t1),
                't2': float(self.t2),
                'escape': record['escape_rate'],
                'false_block': record['false_block_rate'],
            })

        self.history.append(record)
        return record

    # ------------------------------------------------------------------ #
    # metrics
    # ------------------------------------------------------------------ #
    def _metrics(self, world, R, decisions, caught, labels, active, round_idx):
        fraud = active & (labels == 1)
        legit = active & (labels == 0)
        n_fraud = int(fraud.sum())
        n_legit = int(legit.sum())
        n_total = int(active.sum())

        blocked = active & ((decisions == 'block') | caught)
        reviewed = active & (decisions == 'review')
        survived = active & (~blocked)

        fraud_blocked = int((blocked & fraud).sum())
        legit_blocked = int((blocked & legit).sum())
        fraud_escaped = int((survived & fraud).sum())
        legit_survived = int((survived & legit).sum())
        reviewed_fraud_caught = int((caught & fraud).sum())
        reviewed_fraud_survived = int(
            (reviewed & (~caught) & fraud).sum())

        n_allow = int((active & (decisions == 'allow')).sum())
        n_review = int((active & (decisions == 'review')).sum())
        n_block = int((active & (decisions == 'block')).sum())

        block_total = fraud_blocked + legit_blocked
        escape_rate = (fraud_escaped / n_fraud) if n_fraud else 0.0
        false_block_rate = (legit_blocked / n_legit) if n_legit else 0.0
        defense_precision = (fraud_blocked / block_total) if block_total else 0.0
        defense_recall = (fraud_blocked / n_fraud) if n_fraud else 0.0
        block_rate = (block_total / n_total) if n_total else 0.0
        review_rate = (n_review / n_total) if n_total else 0.0

        # ---- decision latency (simulated, per transaction) ---------------
        n_txn = np.array([
            max(1.0, float(a.attrs.get('txn_count', 0.0)) * 60.0)
            for a in world.accounts
        ])
        latency_ms = 0.0
        total_txn = 0.0
        for i in range(len(R)):
            if not active[i]:
                continue
            total_txn += n_txn[i]
            latency_ms += n_txn[i] * DECISION_LATENCY_MS[str(decisions[i])]
        decision_latency = (latency_ms / total_txn) if total_txn else 0.0

        # ---- risk distribution ------------------------------------------
        risk_active = R[active]
        risk_summary = {
            'mean': float(risk_active.mean()) if len(risk_active) else 0.0,
            'median': float(np.median(risk_active)) if len(risk_active) else 0.0,
            'p75': float(np.percentile(risk_active, 75)) if len(risk_active) else 0.0,
            'p95': float(np.percentile(risk_active, 95)) if len(risk_active) else 0.0,
            'max': float(risk_active.max()) if len(risk_active) else 0.0,
        }

        # ---- risk histogram (20 bins) with decision stacking ------------
        n_bins = 20
        hist = {'allow': np.zeros(n_bins, dtype=int),
                'review': np.zeros(n_bins, dtype=int),
                'block': np.zeros(n_bins, dtype=int)}
        for i in np.where(active)[0]:
            bin_idx = min(n_bins - 1, int(np.clip(R[i], 0.0, 0.999999) * n_bins))
            hist[str(decisions[i])][bin_idx] += 1
        risk_hist = {
            'nbins': n_bins,
            'allow': hist['allow'].tolist(),
            'review': hist['review'].tolist(),
            'block': hist['block'].tolist(),
        }

        return {
            'round': int(round_idx),
            't1': float(self.t1),
            't2': float(self.t2),
            'policy': self.policy,
            'n_total': n_total,
            'n_fraud': n_fraud,
            'n_legit': n_legit,
            'n_allow': n_allow,
            'n_review': n_review,
            'n_block': n_block,
            'block_rate': float(block_rate),
            'review_rate': float(review_rate),
            'escape_rate': float(escape_rate),
            'false_block_rate': float(false_block_rate),
            'defense_precision': float(defense_precision),
            'defense_recall': float(defense_recall),
            'fraud_blocked': fraud_blocked,
            'fraud_escaped': fraud_escaped,
            'legit_blocked': legit_blocked,
            'legit_survived': legit_survived,
            'reviewed_fraud_caught': reviewed_fraud_caught,
            'reviewed_fraud_survived': reviewed_fraud_survived,
            'avg_risk': float(risk_summary['mean']),
            'decision_latency_ms': float(decision_latency),
            'risk': risk_summary,
            'risk_hist': risk_hist,
            'survived_fraud': fraud_escaped,
            'weights': dict(self.weights),
        }

    # ------------------------------------------------------------------ #
    # adaptive thresholds
    # ------------------------------------------------------------------ #
    def _update_thresholds(self, escape_rate, false_block_rate, round_idx):
        """T_{r+1} = T_r + alpha * (false_block - escape)

        escape high   -> threshold falls  -> stricter  (more blocking)
        false block   -> threshold rises   -> more lenient (fewer blocks)
        """
        delta = self.alpha * (float(false_block_rate) - float(escape_rate))
        self.t1 = float(np.clip(self.t1 + delta, 0.05, 0.95))
        self.t2 = float(np.clip(self.t2 + delta, 0.05, 0.98))
        if self.t2 <= self.t1 + 0.05:
            self.t2 = float(np.clip(self.t1 + 0.05, 0.05, 0.98))
        self.threshold_history.append({
            'round': int(round_idx), 't1': float(self.t1),
            't2': float(self.t2),
            'escape': float(escape_rate),
            'false_block': float(false_block_rate),
        })

    # ------------------------------------------------------------------ #
    # reporting helpers
    # ------------------------------------------------------------------ #
    def component_means(self, world, probs, predicted_fraud, known_fraud):
        comp = compute_risk_components(world, probs, predicted_fraud,
                                       known_fraud)
        active = world.active_mask()
        return {
            name: float(np.mean(comp[name][active])) if active.any() else 0.0
            for name in RISK_COMPONENT_NAMES
        }

    def state(self):
        return {
            'policy': self.policy,
            't1': float(self.t1),
            't2': float(self.t2),
            'alpha': float(self.alpha),
            'review_catch_rate': float(self.review_catch_rate),
            'weights': dict(self.weights),
        }
