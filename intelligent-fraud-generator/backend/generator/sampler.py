"""Probabilistic synthesizer for new fraud behaviour vectors.

Instead of a neural generator we can model the observed (missed) fraud with a
simple noise-injected resampling scheme:

  1. draw a seed row from the recent fraud pool,
  2. add Gaussian drift scaled by the pool's per-feature standard deviation,
  3. occasionally (50%) use the pool mean as the base instead of a seed row.

The result is clamped into [0, 1] so it stays a valid behaviour vector.
"""

import numpy as np


def sample_probabilistic(pool, n, rng, noise=1.0):
    """Sample ``n`` feature rows from a probabilistic model of ``pool``."""
    pool = np.asarray(pool, dtype=float)
    if pool.ndim == 1:
        pool = pool[None, :]
    if len(pool) == 0:
        return None
    d = pool.shape[1]
    mu = pool.mean(axis=0)
    sd = pool.std(axis=0) + 1e-6
    mn = pool.min(axis=0)
    mx = pool.max(axis=0)

    base = pool[rng.integers(0, len(pool), size=n)]
    drift = rng.normal(0.0, sd * noise, size=(n, d))
    gate = rng.random((n, d)) < 0.5
    out = np.where(gate, base + drift, mu + drift)
    out = np.clip(out, 0.0, 1.0)
    return out


def apply_diversity(feats, pool, diversity, rng):
    """Push samples further away from the seed distribution when asked."""
    feats = np.asarray(feats, dtype=float)
    pool = np.asarray(pool, dtype=float)
    if pool.ndim == 1:
        pool = pool[None, :]
    if diversity is None or float(diversity) <= 1.0 or len(pool) == 0:
        return feats
    sd = pool.std(axis=0) + 1e-6
    extra = rng.normal(0.0, sd * (float(diversity) - 1.0) * 0.5,
                       size=feats.shape)
    return np.clip(feats + extra, 0.0, 1.0)
