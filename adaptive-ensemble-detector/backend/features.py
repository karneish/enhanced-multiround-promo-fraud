"""Feature engineering: world state -> numeric matrix."""
import numpy as np
from collections import Counter

INTRINSIC_NAMES = [
    'age', 'email_disposable', 'phone_verified', 'device_fresh',
    'ip_proxy', 'loc_entropy', 'login_night', 'amount_mean',
    'amount_std', 'txn_count', 'txn_freq',
]

N_INTRINSIC = len(INTRINSIC_NAMES)
N_GRAPH = 6
N_FEATURES = N_INTRINSIC + N_GRAPH
FEATURE_NAMES = INTRINSIC_NAMES + [
    'referral_count', 'degree', 'clustering', 'shared_device', 'shared_ip', 'fraud_neighbor_ratio',
]


def compute_features(world):
    n = world.node_count()
    active = world.active_mask()
    labels = world.label_array()
    accounts = world.accounts

    X = np.zeros((n, N_FEATURES), dtype=np.float32)

    for i, acct in enumerate(accounts):
        if not active[i]:
            continue
        attrs = acct.attrs or {}
        for j, name in enumerate(INTRINSIC_NAMES):
            X[i, j] = attrs.get(name, 0.0)

    adj = {i: set() for i in range(n)}
    referral_set = set()
    for src, dst in world.referral_edges:
        if src < n and dst < n:
            adj[src].add(dst)
            adj[dst].add(src)
            referral_set.add((src, dst))
            referral_set.add((dst, src))
    for src, dst in world.device_edges:
        if src < n and dst < n:
            adj[src].add(dst)
            adj[dst].add(src)
    for src, dst in world.ip_edges:
        if src < n and dst < n:
            adj[src].add(dst)
            adj[dst].add(src)

    device_map = {}
    for acct in accounts:
        if acct.device_id and acct.id < n:
            device_map.setdefault(acct.device_id, set()).add(acct.id)
    ip_map = {}
    for acct in accounts:
        if acct.ip_id and acct.id < n:
            ip_map.setdefault(acct.ip_id, set()).add(acct.id)

    for i in range(n):
        if not active[i]:
            continue
        neighbors = adj[i]
        ref_count = sum(1 for nb in neighbors if (i, nb) in referral_set)
        X[i, N_INTRINSIC] = min(ref_count / 10.0, 1.0)
        X[i, N_INTRINSIC + 1] = min(len(neighbors) / 20.0, 1.0)

        cluster = 0.0
        if len(neighbors) >= 2:
            nb_list = list(neighbors)
            possible = len(nb_list) * (len(nb_list) - 1) / 2
            actual = 0
            for a_idx in range(len(nb_list)):
                for b_idx in range(a_idx + 1, len(nb_list)):
                    if nb_list[b_idx] in adj.get(nb_list[a_idx], set()):
                        actual += 1
            if possible > 0:
                cluster = actual / possible
        X[i, N_INTRINSIC + 2] = min(cluster, 1.0)

        acct = accounts[i]
        same_device = len(device_map.get(acct.device_id, set()) - {i})
        same_ip = len(ip_map.get(acct.ip_id, set()) - {i})
        X[i, N_INTRINSIC + 3] = min(same_device / 10.0, 1.0)
        X[i, N_INTRINSIC + 4] = min(same_ip / 10.0, 1.0)

        fraud_nb = 0
        total_nb = len(neighbors) if neighbors else 0
        for nb in neighbors:
            if labels[nb] == 1 and active[nb]:
                fraud_nb += 1
        X[i, N_INTRINSIC + 5] = fraud_nb / max(total_nb, 1)

    return X, FEATURE_NAMES
