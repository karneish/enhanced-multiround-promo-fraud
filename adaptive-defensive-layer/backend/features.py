"""Feature engineering for the account graph.

Turns the current world state into a numeric feature matrix (one row per
account) mixing intrinsic behaviour with graph-derived signals:

  * referral_count      - how many referral edges touch the account
  * degree              - undirected referral-graph degree
  * clustering          - local clustering coefficient on the referral graph
  * shared_device       - how many *other* accounts use the same device
  * shared_ip           - how many *other* accounts come from the same IP
  * fraud_neighbor_ratio- fraction of neighbours already known to be fraud

``fraud_neighbor_ratio`` is computed using only the labels the system has
been *given* (supervised), never the hidden truth, so the detector never
cheats during training.
"""

import numpy as np

from .world import (
    FEATURE_NAMES, INTRINSIC_NAMES, GRAPH_NAMES, norm_count,
)


def _clustering_coefficients(neighbours, n):
    coeff = np.zeros(n)
    for i in range(n):
        nb = neighbours[i]
        k = len(nb)
        if k < 2:
            continue
        nblist = list(nb)
        edges = 0
        for j in range(k):
            a = nblist[j]
            nb_a = neighbours[a]
            for b in nblist[j + 1:]:
                if b in nb_a:
                    edges += 1
        coeff[i] = 2.0 * edges / (k * (k - 1))
    return coeff


def compute_features(world, supervised_mask):
    """Return an ``(n_accounts, len(FEATURE_NAMES))`` float matrix.

    Accounts soft-removed by the ADL (``blocked``) contribute nothing: they
    are dropped from every adjacency list and get an all-zero feature row, so
    removed fraud can never influence the detector or the graph statistics.
    """
    n = len(world.accounts)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)))

    active = world.active_mask()
    labels = np.array([a.label for a in world.accounts])
    known_fraud = np.logical_and(supervised_mask, labels == 1)

    # ---- adjacency (active accounts only) ----------------------------------
    referral_nbrs = [set() for _ in range(n)]
    all_nbrs = [set() for _ in range(n)]
    for s, d in world.referral:
        if not active[s] or not active[d]:
            continue
        referral_nbrs[s].add(d)
        referral_nbrs[d].add(s)
        all_nbrs[s].add(d)
        all_nbrs[d].add(s)
    for a, dev in world.device_edges:
        if not active[a]:
            continue
        for b in world.device_users[dev]:
            if b != a and active[b]:
                all_nbrs[a].add(b)
                all_nbrs[b].add(a)
    for a, ip in world.ip_edges:
        if not active[a]:
            continue
        for b in world.ip_users[ip]:
            if b != a and active[b]:
                all_nbrs[a].add(b)
                all_nbrs[b].add(a)

    degree = np.array([len(s) for s in referral_nbrs], dtype=float)
    clustering = _clustering_coefficients(referral_nbrs, n)

    shared_device = np.zeros(n)
    for a, dev in world.device_edges:
        if active[a]:
            shared_device[a] = len(world.device_users[dev]) - 1

    shared_ip = np.zeros(n)
    for a, ip in world.ip_edges:
        if active[a]:
            shared_ip[a] = len(world.ip_users[ip]) - 1

    fraud_ratio = np.zeros(n)
    for i in range(n):
        if not active[i]:
            continue
        nb = list(all_nbrs[i])
        if nb:
            fraud_ratio[i] = float(known_fraud[nb].mean())

    # ---- assemble matrix ---------------------------------------------------
    X = np.zeros((n, len(FEATURE_NAMES)))
    col = {name: j for j, name in enumerate(FEATURE_NAMES)}

    for i, acct in enumerate(world.accounts):
        if not active[i]:
            continue  # blocked accounts stay all-zero
        for j, name in enumerate(INTRINSIC_NAMES):
            X[i, j] = max(0.0, min(1.0, acct.attrs.get(name, 0.0)))

    X[:, col['referral_count']] = norm_count(degree, 'referral_count')
    X[:, col['degree']] = norm_count(degree, 'degree')
    X[:, col['clustering']] = clustering
    X[:, col['shared_device']] = norm_count(shared_device, 'shared_device')
    X[:, col['shared_ip']] = norm_count(shared_ip, 'shared_ip')
    X[:, col['fraud_neighbor_ratio']] = fraud_ratio

    return X


def intrinsic_matrix(world):
    """Return only the intrinsic columns (used by the fraud generator)."""
    X = compute_features(world, np.zeros(len(world.accounts), dtype=bool))
    return X[:, :len(INTRINSIC_NAMES)]
