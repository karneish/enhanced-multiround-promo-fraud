"""Synthetic promo-referral marketplace world."""
import random
import math
import numpy as np

INTRINSIC_NAMES = [
    'age', 'email_disposable', 'phone_verified', 'device_fresh',
    'ip_proxy', 'loc_entropy', 'login_night', 'amount_mean',
    'amount_std', 'txn_count', 'txn_freq',
]

FRAUD_TEMPLATES = {
    'fake_identity':     {'means': [0.15, 0.9, 0.1, 0.85, 0.7, 0.8, 0.7, 0.9, 0.6, 0.9, 0.8], 'noise': 0.12, 'ring_affinity': 0.3, 'device_spray': 0.6, 'ip_reuse': 0.5},
    'referral_farming':  {'means': [0.3, 0.7, 0.2, 0.7, 0.5, 0.6, 0.5, 0.6, 0.4, 0.8, 0.7], 'noise': 0.10, 'ring_affinity': 0.9, 'device_spray': 0.2, 'ip_reuse': 0.3},
    'device_spray':      {'means': [0.2, 0.8, 0.15, 0.9, 0.6, 0.7, 0.6, 0.8, 0.5, 0.95, 0.9], 'noise': 0.08, 'ring_affinity': 0.2, 'device_spray': 0.95, 'ip_reuse': 0.4},
    'vpn_hop':           {'means': [0.25, 0.6, 0.2, 0.6, 0.95, 0.9, 0.8, 0.7, 0.5, 0.7, 0.6], 'noise': 0.10, 'ring_affinity': 0.3, 'device_spray': 0.3, 'ip_reuse': 0.1},
    'quiet_sampler':     {'means': [0.5, 0.3, 0.5, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.2, 0.2], 'noise': 0.06, 'ring_affinity': 0.1, 'device_spray': 0.1, 'ip_reuse': 0.2},
}

GENUINE_PROFILE = {'means': [0.6, 0.1, 0.8, 0.2, 0.1, 0.4, 0.2, 0.3, 0.25, 0.3, 0.35], 'noise': 0.15}


class Account:
    __slots__ = [
        'id', 'attrs', 'label', 'strategy', 'base', 'device_id', 'ip_id',
        'creation_round', 'blocked', 'blocked_round', 'decision', 'risk',
        'features', 'predicted', 'predicted_round',
    ]

    def __init__(self, id, label, strategy, base, device_id, ip_id, creation_round, attrs=None):
        self.id = id
        self.label = label
        self.strategy = strategy
        self.base = base
        self.device_id = device_id
        self.ip_id = ip_id
        self.creation_round = creation_round
        self.attrs = attrs if attrs is not None else {}
        self.blocked = False
        self.blocked_round = -1
        self.decision = 'allow'
        self.risk = 0.0
        self.features = None
        self.predicted = False
        self.predicted_round = -1


class World:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.accounts = []
        self.referral_edges = []
        self.device_edges = []
        self.ip_edges = []
        self.device_users = {}
        self.ip_users = {}
        self._next_id = 0

    def _new_id(self):
        i = self._next_id
        self._next_id += 1
        return i

    def _sample_attrs(self, profile, rng=None):
        r = rng or self.rng
        attrs = {}
        for i, name in enumerate(INTRINSIC_NAMES):
            v = profile['means'][i] + r.gauss(0, profile['noise'])
            attrs[name] = max(0.0, min(1.0, v))
        return attrs

    def add_account(self, label, strategy, creation_round, attrs=None, device_id=None, ip_id=None):
        aid = self._new_id()
        if device_id is None:
            device_id = f'd{self.rng.randint(0, 500)}'
        if ip_id is None:
            ip_id = f'ip{self.rng.randint(0, 400)}'
        if attrs is None:
            tmpl = FRAUD_TEMPLATES.get(strategy, FRAUD_TEMPLATES['quiet_sampler']) if label == 1 else GENUINE_PROFILE
            attrs = self._sample_attrs(tmpl)
        acct = Account(aid, label, strategy, strategy, device_id, ip_id, creation_round, attrs)
        self.accounts.append(acct)
        self.device_users.setdefault(device_id, set()).add(aid)
        self.ip_users.setdefault(ip_id, set()).add(aid)
        return acct

    def add_referral(self, src, dst):
        self.referral_edges.append((src, dst))

    def build_initial(self, base_accounts=500, initial_fraud=60, seed=42):
        for _ in range(base_accounts):
            self.add_account(0, 'genuine', -1)
        fraud_per = max(1, initial_fraud // len(FRAUD_TEMPLATES))
        for strat, tmpl in FRAUD_TEMPLATES.items():
            for _ in range(fraud_per):
                acct = self.add_account(1, strat, 0)
                if self.rng.random() < tmpl['ring_affinity'] * 0.4:
                    candidates = [a for a in self.accounts if a.label == 1 and a.id != acct.id and a.creation_round == 0]
                    if candidates:
                        target = self.rng.choice(candidates)
                        self.add_referral(acct.id, target.id)
        genuine = [a for a in self.accounts if a.label == 0]
        fraud = [a for a in self.accounts if a.label == 1]
        for f in fraud:
            if self.rng.random() < 0.35 and genuine:
                self.add_referral(f.id, self.rng.choice(genuine).id)
        self._add_device_ip_edges()

    def add_round_accounts(self, round_idx, num_genuine, num_fraud, fraud_features=None, fraud_strategy='evolved'):
        for _ in range(num_genuine):
            self.add_account(0, 'genuine', round_idx)
        generated = []
        if fraud_features is not None:
            for i, feats in enumerate(fraud_features):
                attrs = {name: float(max(0, min(1, feats[j]))) for j, name in enumerate(INTRINSIC_NAMES)}
                acct = self.add_account(1, fraud_strategy, round_idx, attrs=attrs)
                generated.append(acct)
        else:
            for _ in range(num_fraud):
                strat = self.rng.choice(list(FRAUD_TEMPLATES.keys()))
                acct = self.add_account(1, strat, round_idx)
                generated.append(acct)
        self._add_round_edges(round_idx, generated)
        self._add_device_ip_edges()
        return generated

    def _add_round_edges(self, round_idx, new_fraud):
        recent_genuine = [a for a in self.accounts if a.label == 0 and a.creation_round == round_idx]
        for f in new_fraud:
            if self.rng.random() < 0.3 and recent_genuine:
                self.add_referral(f.id, self.rng.choice(recent_genuine).id)
            if self.rng.random() < 0.2 and len(new_fraud) > 1:
                others = [o for o in new_fraud if o.id != f.id]
                if others:
                    self.add_referral(f.id, self.rng.choice(others).id)

    def _add_device_ip_edges(self):
        self.device_edges.clear()
        self.ip_edges.clear()
        for devs in self.device_users.values():
            dl = sorted(devs)
            for i in range(len(dl)):
                for j in range(i + 1, min(i + 6, len(dl))):
                    self.device_edges.append((dl[i], dl[j]))
        for ips in self.ip_users.values():
            il = sorted(ips)
            for i in range(len(il)):
                for j in range(i + 1, min(i + 6, len(il))):
                    self.ip_edges.append((il[i], il[j]))

    def block(self, idx, round_idx):
        acct = self.accounts[idx]
        acct.blocked = True
        acct.blocked_round = round_idx
        acct.decision = 'block'

    def active_mask(self):
        return np.array([not a.blocked for a in self.accounts])

    def active_indices(self):
        return np.array([i for i, a in enumerate(self.accounts) if not a.blocked])

    def label_array(self):
        return np.array([a.label for a in self.accounts])

    def round_mask(self, r):
        return np.array([a.creation_round == r for a in self.accounts])

    def supervised_mask(self, ratio=1.0):
        rng = np.random.RandomState(42)
        mask = np.zeros(len(self.accounts), dtype=bool)
        for i, a in enumerate(self.accounts):
            if a.label == 1 and rng.random() < ratio:
                mask[i] = True
        return mask

    def node_count(self):
        return len(self.accounts)

    def edge_count(self):
        return len(self.referral_edges) + len(self.device_edges) + len(self.ip_edges)

    def label_counts(self):
        labels = self.label_array()
        active = self.active_mask()
        return {
            'total': int(active.sum()),
            'genuine': int((active & (labels == 0)).sum()),
            'fraud': int((active & (labels == 1)).sum()),
            'blocked': int((~active).sum()),
        }

    def graph_dict(self):
        nodes = []
        for i, a in enumerate(self.accounts):
            nodes.append({
                'id': a.id,
                'idx': i,
                'label': a.label,
                'round': a.creation_round,
                'strategy': a.strategy,
                'base': a.base,
                'blocked': a.blocked,
                'decision': a.decision,
                'risk': round(a.risk, 4),
                'attrs': {k: round(v, 3) for k, v in a.attrs.items()} if a.attrs else {},
            })
        return {
            'nodes': nodes,
            'edges': {
                'referral': list(self.referral_edges),
                'device': list(self.device_edges[:2000]),
                'ip': list(self.ip_edges[:2000]),
            },
        }

    def strategy_counts(self, mask=None):
        counts = {}
        for i, a in enumerate(self.accounts):
            if mask is not None and not mask[i]:
                continue
            s = a.strategy or 'genuine'
            counts[s] = counts.get(s, 0) + 1
        return counts
