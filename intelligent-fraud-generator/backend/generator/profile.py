"""Rolling profile of the fraud that escaped detection.

This is the "memory" of the intelligent generator. Each round the framework
hands over the fraudulent accounts that the detector failed to flag (the
false negatives). ``FraudProfile`` keeps a bounded window of those examples so
the generator can learn from recent successes instead of only the current one.
"""

import random

import numpy as np


def _pairwise_mean_distance(feats):
    if feats is None or len(feats) < 2:
        return 0.0
    dists = np.sqrt(((feats[:, None, :] - feats[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(len(feats), k=1)
    return float(dists[iu].mean())


class FraudProfile:
    def __init__(self, window=5):
        self.window = max(1, int(window))
        self.feature_pool = []
        self.strategy_pool = []          # strategy names per pool row
        self.base_strategy_pool = []
        self.round_of_row = []           # which round each row came from
        self.familiar = set()            # accounts repeatedly attacked
        self.familiar_round = {}         # account id -> last seen round
        self.confidences = []            # model confidence per row
        self.rounds_seen = 0
        self.history = []                # one entry per round (for reporting)

    def update(self, feats, strategies, bases, familiar, confidences, round_idx):
        feats = np.asarray(feats, dtype=float)
        if feats.ndim == 1:
            feats = feats[None, :]
        self.feature_pool.append(feats)
        self.strategy_pool.extend(list(strategies))
        self.base_strategy_pool.extend(list(bases))
        self.round_of_row.extend([int(round_idx)] * len(feats))
        self.confidences.extend([float(c) for c in confidences])

        if familiar:
            for f in familiar:
                self.familiar.add(int(f))
                self.familiar_round[int(f)] = int(round_idx)

        # keep only the last ``window`` rounds of rows
        if len(self.feature_pool) > self.window:
            drop = self.feature_pool[0]
            keep = self.feature_pool[-self.window:]
            n_drop = len(drop)
            self.feature_pool = keep
            self.strategy_pool = self.strategy_pool[n_drop:]
            self.base_strategy_pool = self.base_strategy_pool[n_drop:]
            self.round_of_row = self.round_of_row[n_drop:]
            self.confidences = self.confidences[n_drop:]

        # cap the familiar set so it cannot grow without bound
        if len(self.familiar) > 4000:
            self.familiar = set(random.sample(sorted(self.familiar), 4000))
        self.rounds_seen += 1
        self.history.append({
            'round': int(round_idx),
            'missed': int(len(feats)),
            'familiar': int(len(set(familiar) if familiar else [])),
        })

    # ---- queries ----------------------------------------------------------
    def pool(self):
        if not self.feature_pool:
            return None
        return np.concatenate(self.feature_pool, axis=0)

    def stats(self):
        feats = self.pool()
        if feats is None:
            return None
        return {
            'mean': feats.mean(axis=0),
            'std': feats.std(axis=0) + 1e-6,
            'min': feats.min(axis=0),
            'max': feats.max(axis=0),
        }

    def diversity(self):
        feats = self.pool()
        return _pairwise_mean_distance(feats)

    def strategy_counts(self):
        from collections import Counter
        return dict(Counter(self.strategy_pool))

    def base_counts(self):
        from collections import Counter
        return dict(Counter(self.base_strategy_pool))

    def summary(self):
        feats = self.pool()
        return {
            'rows': 0 if feats is None else int(len(feats)),
            'rounds_seen': int(self.rounds_seen),
            'diversity': float(self.diversity()),
            'familiar': int(len(self.familiar)),
            'history': list(self.history),
            'strategies': self.base_counts(),
            'mean_conf': float(np.mean(self.confidences)) if self.confidences else -1.0,
        }
