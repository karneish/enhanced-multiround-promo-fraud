"""Synthetic promo-referral world.

Models a marketplace running a "refer a friend, earn a reward" programme.
Every actor is an *account* that belongs to one of two classes:

  * genuine  (label 0) - normal users who transact and invite friends
  * fraud    (label 1) - attackers abusing the reward (fake identities,
    referral farming, device spraying, VPN hopping, quiet reward skimming)

Accounts carry a set of *intrinsic behaviours* (email/phone hygiene, device
freshness, IP reputation, timing, transaction statistics) and live inside a
graph built from three relation types:

  * referral  : account -> account   (who invited whom)
  * device    : account -> device    (which handset was used)
  * ip        : account -> ip        (which address the account came from)

Everything is seeded so that a given configuration always reproduces the
same world, which keeps experiments comparable.
"""

import random

import numpy as np

# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------
INTRINSIC_NAMES = [
    'age', 'email_disposable', 'phone_verified', 'device_fresh', 'ip_proxy',
    'loc_entropy', 'login_night', 'amount_mean', 'amount_std',
    'txn_count', 'txn_freq',
]

GRAPH_NAMES = [
    'referral_count', 'degree', 'clustering', 'shared_device',
    'shared_ip', 'fraud_neighbor_ratio',
]

FEATURE_NAMES = INTRINSIC_NAMES + GRAPH_NAMES

# upper bounds used to squash raw counts / amounts into [0, 1]
_MAX_COUNTS = {
    'txn_count': 60.0, 'txn_freq': 20.0, 'amount_mean': 5000.0,
    'amount_std': 2000.0, 'referral_count': 12.0, 'degree': 20.0,
    'shared_device': 30.0, 'shared_ip': 40.0,
}

# intrinsic attributes that are conceptually binary
_BINARY_INTRINSIC = {
    'email_disposable', 'phone_verified', 'device_fresh', 'ip_proxy',
}


def norm_count(value, key):
    cap = _MAX_COUNTS.get(key, 1.0)
    return np.clip(np.asarray(value, dtype=float) / cap, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Initial fraud strategy templates (round 0)
# Each is a target feature profile + structure preferences.
# ---------------------------------------------------------------------------
INITIAL_STRATEGIES = {
    'fake_identity': {
        'means': [0.35, 0.55, 0.30, 0.45, 0.35, 0.55, 0.55, 0.06, 0.04, 0.06, 0.07],
        'noise': 0.22,
        'ring_affinity': 0.25,
        'device_spray': 0.10,
        'ip_reuse': 0.50,
    },
    'referral_farming': {
        'means': [0.40, 0.40, 0.55, 0.30, 0.25, 0.30, 0.45, 0.08, 0.07, 0.16, 0.15],
        'noise': 0.20,
        'ring_affinity': 0.90,
        'device_spray': 0.50,
        'ip_reuse': 0.30,
    },
    'device_spray': {
        'means': [0.38, 0.35, 0.60, 0.20, 0.20, 0.25, 0.40, 0.08, 0.10, 0.15, 0.12],
        'noise': 0.18,
        'ring_affinity': 0.40,
        'device_spray': 0.95,
        'ip_reuse': 0.40,
    },
    'vpn_hop': {
        'means': [0.42, 0.40, 0.50, 0.40, 0.75, 0.85, 0.55, 0.07, 0.10, 0.10, 0.12],
        'noise': 0.20,
        'ring_affinity': 0.30,
        'device_spray': 0.60,
        'ip_reuse': 0.70,
    },
    'quiet_sampler': {
        'means': [0.45, 0.25, 0.70, 0.25, 0.15, 0.20, 0.75, 0.05, 0.03, 0.08, 0.06],
        'noise': 0.16,
        'ring_affinity': 0.15,
        'device_spray': 0.20,
        'ip_reuse': 0.40,
    },
}


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------
class Account:
    __slots__ = (
        'idx', 'attrs', 'label', 'creation_round', 'strategy', 'base_strategy',
        'tags', 'device_id', 'ip_id', 'detected_round',
    )

    def __init__(self, idx, attrs, label, creation_round, strategy='',
                 base_strategy='', tags=(), device_id=-1, ip_id=-1):
        self.idx = idx
        self.attrs = attrs
        self.label = label
        self.creation_round = creation_round
        self.strategy = strategy
        self.base_strategy = base_strategy
        self.tags = list(tags)
        self.device_id = device_id
        self.ip_id = ip_id
        self.detected_round = -1


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
class World:
    """Mutable collection of accounts plus their referral / device / IP edges."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.np = np.random.default_rng(seed)
        self.accounts = []
        self.referral = []          # (src, dst) account index pairs
        self.device_edges = []      # (account index, device id)
        self.ip_edges = []          # (account index, ip id)
        self.device_users = {}      # device id -> set of account indexes
        self.ip_users = {}          # ip id -> set of account indexes
        self._next_device = 0
        self._next_ip = 0

    # ---- helpers ----------------------------------------------------------
    def _new_device(self):
        d = self._next_device
        self._next_device += 1
        self.device_users.setdefault(d, set())
        return d

    def _new_ip(self):
        ip = self._next_ip
        self._next_ip += 1
        self.ip_users.setdefault(ip, set())
        return ip

    def _pick_device(self, is_fraud, spray):
        if not is_fraud:
            if self.device_users and self.rng.random() < 0.08:
                return self.rng.choice(list(self.device_users.keys()))
            return self._new_device()
        if self.device_users and self.rng.random() < max(0.0, spray):
            fraud_devices = [d for d, users in self.device_users.items()
                             if any(self.accounts[u].label == 1 for u in users)]
            pool = fraud_devices or list(self.device_users.keys())
            return self.rng.choice(pool)
        return self._new_device()

    def _pick_ip(self, is_fraud, reuse):
        if not is_fraud:
            if self.ip_users and self.rng.random() < 0.10:
                return self.rng.choice(list(self.ip_users.keys()))
            return self._new_ip()
        if self.ip_users and self.rng.random() < max(0.0, reuse):
            proxy_ips = [i for i, users in self.ip_users.items()
                         if any(self.accounts[u].label == 1 for u in users)]
            pool = proxy_ips or list(self.ip_users.keys())
            return self.rng.choice(pool)
        return self._new_ip()

    def _attach_device(self, account_idx, device_id):
        self.accounts[account_idx].device_id = device_id
        self.device_edges.append((account_idx, device_id))
        self.device_users.setdefault(device_id, set()).add(account_idx)

    def _attach_ip(self, account_idx, ip_id):
        self.accounts[account_idx].ip_id = ip_id
        self.ip_edges.append((account_idx, ip_id))
        self.ip_users.setdefault(ip_id, set()).add(account_idx)

    def add_referral(self, src, dst):
        src = int(src)
        dst = int(dst)
        if src != dst and 0 <= src < len(self.accounts) and 0 <= dst < len(self.accounts):
            self.referral.append((src, dst))

    # ---- population -------------------------------------------------------
    def add_genuine(self, round_idx, n, referral_density=0.04):
        """Create ``n`` genuine accounts and return their indexes."""
        added = []
        for _ in range(n):
            attrs = self._genuine_attrs()
            idx = len(self.accounts)
            device = self._pick_device(False, 0.0)
            ip = self._pick_ip(False, 0.0)
            self.accounts.append(Account(idx, attrs, 0, round_idx,
                                         device_id=device, ip_id=ip))
            self._attach_device(idx, device)
            self._attach_ip(idx, ip)
            added.append(idx)
        # a few genuine invites so the referral graph is not empty
        for src in added:
            if self.rng.random() < referral_density and len(added) > 1:
                dst = self.rng.choice([a for a in added if a != src])
                self.add_referral(src, dst)
        return added

    def add_fraud(self, spec, round_idx):
        """Add one fraud account described by ``spec`` (see generator engine)."""
        attrs = dict(spec['attrs'])
        base = spec.get('base', 'evolved')
        tags = spec.get('tags', [])
        strategy = spec.get('strategy', base)

        idx = len(self.accounts)

        requested_device = spec.get('device')
        if requested_device is not None and requested_device in self.device_users:
            device = requested_device
        else:
            device = self._pick_device(True, spec.get('spray', 0.3))

        requested_ip = spec.get('ip')
        if requested_ip is not None and requested_ip in self.ip_users:
            ip = requested_ip
        else:
            ip = self._pick_ip(True, spec.get('ip_reuse', 0.4))

        self.accounts.append(Account(idx, attrs, 1, round_idx, strategy,
                                     base_strategy=base, tags=tags,
                                     device_id=device, ip_id=ip))
        self._attach_device(idx, device)
        self._attach_ip(idx, ip)

        for dst in spec.get('referrals', []):
            self.add_referral(idx, dst)
        return idx

    # ---- sampling ---------------------------------------------------------
    @staticmethod
    def _genuine_attrs():
        rng = random.random
        return {
            'age': 0.30 + rng() * 0.70,
            'email_disposable': 1.0 if rng() < 0.06 else 0.0,
            'phone_verified': 1.0 if rng() < 0.88 else 0.0,
            'device_fresh': 1.0 if rng() < 0.06 else 0.0,
            'ip_proxy': 1.0 if rng() < 0.05 else 0.0,
            'loc_entropy': rng() * 0.25,
            'login_night': rng() * 0.20,
            'amount_mean': 0.08 + rng() * 0.20,
            'amount_std': 0.03 + rng() * 0.10,
            'txn_count': 0.10 + rng() * 0.35,
            'txn_freq': 0.05 + rng() * 0.20,
        }

    def sample_strategy_attrs(self, name):
        """Draw one fraud attribute vector from a round-0 strategy template."""
        template = INITIAL_STRATEGIES[name]
        vals = np.array(template['means'], dtype=float) + \
            self.np.normal(0.0, template['noise'], len(INTRINSIC_NAMES))
        vals = np.clip(vals, 0.0, 1.0)
        out = {}
        for j, fname in enumerate(INTRINSIC_NAMES):
            if fname in _BINARY_INTRINSIC:
                out[fname] = 1.0 if self.rng.random() < vals[j] else 0.0
            else:
                out[fname] = float(vals[j])
        return out

    def strategy_template(self, name):
        return INITIAL_STRATEGIES[name]

    # ---- structure --------------------------------------------------------
    def num_nodes(self):
        return len(self.accounts)

    def num_edges(self):
        return len(self.referral) + len(self.device_edges) + len(self.ip_edges)


def feature_names():
    return list(FEATURE_NAMES)


def intrinsic_names():
    return list(INTRINSIC_NAMES)
