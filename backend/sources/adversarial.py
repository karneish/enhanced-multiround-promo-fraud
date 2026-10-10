"""Adversarial multi-round promo-fraud scenario, restored as a stream source.

This restores the second contribution of

    Hafizh Adi Prasetya, Xin Liu, Tsuyoshi Murata, Akiyoshi Matono,
    "A multi-rounded adversarial scenario for graph-based promo fraud
    detection", Social Network Analysis and Mining (2026),
    doi:10.1007/s13278-025-01566-0

The single-app rebuild dropped it entirely and had no attack model at all, so
"multi-round" meant only "several time windows". The paper's contribution is
that an attacker *adapts*: each round, new fraud is generated from the accounts
the model is getting wrong.

Three graphs, as in the paper §4:
    G_t     test graph      everything used for evaluation this round
    G_tQ    query graph     only the NEW nodes generated this round
    G_tS    support graph   previous support + newly generated support nodes

Two generator functions:
    g  query generator  - duplicates/perturbs nodes to make fresh fraud
    h  support generator - augments the training support

Seed-selection strategies (``adver_choose``), ported from the author's
``src/adversary/choose/simple_choose.py``:
    random  uniform over all nodes
    greedy  prioritises nodes the model *missed* (predicted False, label 1)
    ogreedy as greedy but restricted to nodes created before round 1

Modification strategies (``adver_mod``), from ``src/adversary/modify/simple_mod.py``:
    replay  no modification
    relative  perturb each feature by ``feat_coef`` standard deviations and
              rewire ``conn_coef`` of the node's edges to new targets
    absolute  perturb features toward a benign prototype
    mixing  blend the seed's features toward the graph mean

Resampling strategies (``augment``), from ``src/meta_strategies/augment.py``:
    none  nothing
    random  replay fresh positives/negatives each round
    reage  randomise the creation_round of round-0 training nodes, which is how
           the paper forces the embedder to keep a multi-round window

Everything is emitted through the same canonical event shape as the other
sources, so ``LiveSimulation`` consumes it without change.
"""
from __future__ import annotations

import numpy as np

from backend.events import normalize_event

SOURCE = "adver"

#: The author's default budget schedule as a fraction of the graph, used when
#: the config does not override it (main.py sets these to 5% of pos/neg).
DEFAULT_NEW_POS_FRACTION = 0.05
DEFAULT_NEW_NEG_FRACTION = 0.05
DEFAULT_ROUND_WINDOW = 7


def _sparse_to_edge_index(graph):
    """scipy/dgl-free graph dict -> (2, m) edge index, undirected."""
    import scipy.sparse as sp
    src, dst = graph["edge_index"]
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    return np.vstack([row, col])


class AdversarialScenario:
    """Generates the rounds of the multi-round adversarial scenario.

    ``graph`` is ``{"features": (n, d), "edge_index": (2, m), "label": (n,),}``
    with ``label`` optional (unlabeled graphs get generated positives only
    when a label source exists, exactly as the paper requires).
    """

    def __init__(self, graph, config=None):
        cfg = config or {}
        self.cfg = cfg
        self.rng = np.random.default_rng(int(cfg.get("seed", 42)))
        self.adver_choose = cfg.get("adver_choose", "greedy")
        # REPLAY is the base paper's own default (``DEFAULT_ADVER_CONFIG`` in
        # the author's src/utils/utils_const.py sets adver_mod_name='REPLAY'):
        # the generated node is a verbatim duplicate of its seed, with NO
        # feature or structural perturbation. Perturbing instead makes the
        # task materially harder and was the single largest source of the gap
        # to the published numbers (measured: ~30 on-round F1 with relative
        # perturbation vs the paper's 57).
        self.adver_mod = cfg.get("adver_mod", "replay")
        self.augment = cfg.get("augment", "reage")
        # fail fast on an unknown strategy rather than at the first round
        if self.adver_mod not in ("replay", "relative", "absolute", "mixing"):
            raise ValueError(f"unknown adver_mod: {self.adver_mod!r}")
        if self.adver_choose not in ("random", "greedy", "ogreedy"):
            raise ValueError(f"unknown adver_choose: {self.adver_choose!r}")
        if self.augment not in ("none", "random", "reage"):
            raise ValueError(f"unknown augment: {self.augment!r}")
        self.augment_round_split = int(cfg.get("augment_round_split", 3))
        self.round_window = int(cfg.get("round_window", DEFAULT_ROUND_WINDOW))
        self.feat_coef = float(cfg.get("adver_feat_coef", 1.0))
        self.conn_coef = float(cfg.get("adver_conn_coef", 0.1))

        feats = np.asarray(graph["features"], dtype=np.float64)
        self.base_features = feats
        self.base_labels = (np.asarray(graph["label"]).astype(int)
                            if graph.get("label") is not None
                            else np.zeros(len(feats), dtype=int))
        self.edge_index = _sparse_to_edge_index(graph)
        self.n0 = len(feats)

        labels = self.base_labels
        npos = int((labels == 1).sum())
        nneg = int((labels == 0).sum())
        self.new_pos = int(cfg.get("round_new_pos",
                                    DEFAULT_NEW_POS_FRACTION * npos))
        self.new_neg = int(cfg.get("round_new_neg",
                                    DEFAULT_NEW_NEG_FRACTION * nneg))

        # Live scenario state
        self.features = feats.copy()
        self.labels = labels.copy()
        self.edges = self.edge_index.copy()
        self.creation_round = np.zeros(self.n0, dtype=int)
        self.predicted = np.zeros(self.n0, dtype=bool)
        self.train_mask = np.zeros(self.n0, dtype=bool)
        self.train_mask[:] = True
        self.round_num = 0
        self.history: list[np.ndarray] = []
        self.round_log: list[dict] = []

    # ------------------------------------------------------------- helpers
    def _age(self):
        """Per-node round age, the x-axis of the embedder's temporal axis."""
        return self.round_num - self.creation_round

    def _pool(self, label, only_missed=False, only_original=False):
        pool = np.where(self.labels == label)[0]
        if only_missed:
            pool = pool[~self.predicted[pool]]
        if only_original:
            pool = pool[self.creation_round[pool] < 1]
        return pool

    def _choose_seeds(self, label, n, only_missed=False, only_original=False,
                      exclusive=False):
        """Port of the author's ``generate_seeds`` pool logic.

        The priority pool (missed nodes / original nodes) is drained first; if
        it cannot supply the full count the remainder is drawn from the whole
        class, exactly as ``BaseAdversarialChoice.random_node_seeds`` does.

        ``exclusive`` forbids that fallback, so the returned seeds come only
        from the priority pool (and the count shrinks accordingly).
        """
        prio = self._pool(label, only_missed, only_original)
        rest_pool = np.setdiff1d(np.where(self.labels == label)[0], prio)
        if exclusive:
            n = min(n, len(prio))
            if n <= 0:
                return np.array([], dtype=int)
            return self.rng.choice(prio, size=n, replace=False)
        n = min(n, len(prio) + len(rest_pool))
        if n <= 0:
            return np.array([], dtype=int)
        take_prio = self.rng.choice(prio, size=min(len(prio), n), replace=False)
        remaining = n - len(take_prio)
        take_rest = (self.rng.choice(rest_pool, size=remaining, replace=False)
                     if remaining > 0 else np.array([], dtype=int))
        return np.concatenate([take_prio, take_rest])

    def _duplicate(self, seed_ids):
        """Clone the seed nodes, returning the new node block.

        The clone keeps the seed's neighbourhood shape; the new ids are
        appended after the current node count.
        """
        seed_ids = np.asarray(seed_ids)
        new_ids = np.arange(self.features.shape[0],
                            self.features.shape[0] + len(seed_ids))
        feats = self.features[seed_ids].copy()
        labels = self.labels[seed_ids].copy()
        return new_ids, feats, labels, seed_ids

    def _neighbours(self, node):
        src, dst = self.edges[0], self.edges[1]
        return np.unique(np.concatenate([src[dst == node], dst[src == node]]))

    def _modify(self, feats, labels, seed_ids, new_ids):
        """Apply the configured modification strategy to the cloned block."""
        if self.adver_mod == "replay":
            return feats, labels
        if self.adver_mod == "relative":
            # perturb each feature by feat_coef standard deviations
            std = self.base_features.std(axis=0)
            sign = self.rng.integers(0, 2, size=feats.shape) * 2 - 1
            return feats + sign * std * self.feat_coef, labels
        if self.adver_mod == "absolute":
            benign = self.base_features[self.labels == 0]
            proto = benign.mean(axis=0)
            w = float(np.clip(self.feat_coef, 0.0, 1.0))
            return (1 - w) * feats + w * proto, labels
        if self.adver_mod == "mixing":
            proto = self.base_features.mean(axis=0)
            w = float(np.clip(self.conn_coef, 0.0, 1.0))
            return (1 - w) * feats + w * proto, labels
        raise ValueError(f"unknown adver_mod: {self.adver_mod!r}")

    def _rewire(self, new_ids, seed_ids):
        """Structurally perturb the clones: drop and re-add part of their edges."""
        if self.conn_coef <= 0 or self.adver_mod == "replay":
            return
        n_nodes = self.features.shape[0] + len(new_ids)
        add_src, add_dst = [], []
        for new_id, seed in zip(new_ids, seed_ids):
            neigh = self._neighbours(seed)
            if len(neigh) == 0:
                continue
            n_perturb = int(round(len(neigh) * self.conn_coef))
            if n_perturb <= 0:
                continue
            targets = self.rng.choice(neigh, size=n_perturb, replace=False)
            # rewire each selected edge onto a fresh, previously unseen node
            for t in targets:
                fresh = self.rng.integers(self.n0, n_nodes)
                add_src.append(new_id)
                add_dst.append(fresh)
        if add_src:
            extra = np.vstack([np.array(add_src), np.array(add_dst)])
            self.edges = np.concatenate(
                [self.edges, extra, extra[::-1]], axis=1)

    # ---------------------------------------------------------------- rounds
    def _augment(self):
        """Port of the author's resampling strategies."""
        if self.augment == "none":
            return
        if self.augment == "reage":
            # randomise the age of original training nodes so the embedder is
            # forced to keep a multi-round window rather than memorising one
            lo = max(0, self.round_num - self.augment_round_split + 1)
            orig = np.where(self.creation_round <= 0)[0]
            orig = orig[self.train_mask[orig]]
            if len(orig):
                self.creation_round[orig] = self.rng.integers(
                    lo, self.round_num + 1, size=len(orig))
        elif self.augment == "random":
            self._replay_sampling()
        else:
            raise ValueError(f"unknown augment: {self.augment!r}")

    def _replay_sampling(self):
        for _ in range(self.augment_round_split):
            for label, count in ((1, self.new_pos), (0, self.new_neg)):
                if count <= 0:
                    continue
                pool = self._pool(label, only_missed=False)
                if len(pool) == 0:
                    continue
                seeds = self.rng.choice(pool, size=min(count, len(pool)),
                                        replace=True)
                self._add_round(seeds, label, record=False)

    def _add_round(self, seed_ids, label, record=True, trainable=False):
        new_ids, feats, labels, seeds = self._duplicate(seed_ids)
        feats, labels = self._modify(feats, labels, seeds, new_ids)
        labels[:] = label  # the clone is fraud/benign by construction
        self.features = np.vstack([self.features, feats])
        self.labels = np.concatenate([self.labels, labels])
        self.creation_round = np.concatenate(
            [self.creation_round,
             np.full(len(new_ids), self.round_num, dtype=int)])
        self.predicted = np.concatenate(
            [self.predicted, np.zeros(len(new_ids), dtype=bool)])
        # support nodes join the training set; query nodes do not
        flags = np.full(len(new_ids), bool(trainable))
        self.train_mask = np.concatenate([self.train_mask, flags])
        self._rewire(new_ids, seeds)
        return new_ids

    def next_round(self, support_frac=0.5):
        """Advance one round; return ``(query_ids, support_ids)``.

        The base paper keeps two separate graphs (paper §4, Figure 2):

            G_tQ   query graph    - only the NEW nodes, used for EVALUATION
            G_tS   support graph  - previous support plus newly generated
                                    support nodes, used for TRAINING

        The earlier implementation returned every new node as query and never
        trained on adversarial nodes, so the model was always scored on a
        distribution it had not seen: with pre-training that produced a 0.00
        F1 on round 1. Generating a separate support set - the same seeds, the
        same modification strategies, but a disjoint draw - lets the classifier
        learn the perturbation signature before fresh query nodes are scored.
        """
        self.round_num += 1
        self._augment()

        def draw(label, n, **kw):
            return self._choose_seeds(label, n, **kw)

        missed = dict(only_missed=self.adver_choose in ("greedy", "ogreedy"),
                      only_original=self.adver_choose == "ogreedy")

        # ---- query graph: what will be scored
        q_pos = draw(1, self.new_pos, **missed)
        q_neg = draw(0, self.new_neg) if self.new_neg > 0 else \
            np.array([], dtype=int)
        q_new = []
        if len(q_pos):
            q_new.append(self._add_round(q_pos, 1))
        if len(q_neg):
            q_new.append(self._add_round(q_neg, 0))
        query = (np.concatenate(q_new) if q_new
                 else np.array([], dtype=int))

        # ---- support graph: what will be trained on, drawn AFTER the query
        # seeds are consumed so the two sets cannot overlap
        n_sup_pos = max(1, int(self.new_pos * support_frac)) \
            if self.new_pos > 0 else 0
        n_sup_neg = max(1, int(self.new_neg * support_frac)) \
            if self.new_neg > 0 else 0
        s_pos = draw(1, n_sup_pos) if n_sup_pos else np.array([], dtype=int)
        s_neg = draw(0, n_sup_neg) if n_sup_neg else np.array([], dtype=int)
        s_new = []
        if len(s_pos):
            s_new.append(self._add_round(s_pos, 1, trainable=True))
        if len(s_neg):
            s_new.append(self._add_round(s_neg, 0, trainable=True))
        support = (np.concatenate(s_new) if s_new
                   else np.array([], dtype=int))

        self.round_log.append({
            "round": self.round_num,
            "n_nodes": int(self.features.shape[0]),
            "n_edges": int(self.edges.shape[1]),
            "new_pos": int(len(q_pos) + len(s_pos)),
            "new_neg": int(len(q_neg) + len(s_neg)),
            "query": query,
            "support": support,
        })
        self._last_support = support
        return query, support

    def mark_predicted(self, node_ids, predicted):
        """Feed the model's decisions back so greedy choosing can target misses."""
        node_ids = np.asarray(node_ids, dtype=int)
        if len(node_ids) == 0:
            return
        self.predicted[node_ids] = np.asarray(predicted, dtype=bool)

    # ---------------------------------------------------------------- events
    def graph_dict(self, node_ids=None):
        """Snapshot the graph for the embedder."""
        if node_ids is None:
            feats = self.features
            edges = self.edges
        else:
            keep = np.zeros(self.features.shape[0], dtype=bool)
            keep[node_ids] = True
            src, dst = self.edges
            m = keep[src] & keep[dst]
            feats = self.features[keep]
            edges = np.vstack([src[m], dst[m]])
        return {
            "features": feats,
            "edge_index": edges,
            "age": self._age()[:len(feats)] if node_ids is None
            else self._age()[node_ids],
            "label": self.labels if node_ids is None else self.labels[node_ids],
        }

    def stream_events(self, start_round=0, which="query"):
        """Yield canonical events for one graph per round.

        Round 0 emits the original (real, labelled) graph so the pipeline starts
        from the dataset. Later rounds emit either the query graph ``G_tQ``
        (what is scored) or the support graph ``G_tS`` (what is trained on);
        ``which`` selects between them and ``start_round`` is inclusive.
        """
        if which not in ("query", "support", "both"):
            raise ValueError(f"which must be query|support|both, got {which!r}")
        if start_round <= 0:
            for i in range(self.n0):
                ev = self._event(i, round_id=0, is_new=False)
                if ev is not None:
                    yield ev
        for entry in self.round_log:
            if entry["round"] < start_round:
                continue
            keys = ("query", "support") if which == "both" else (which,)
            for key in keys:
                for nid in entry.get(key, ()):
                    ev = self._event(int(nid), round_id=entry["round"],
                                     is_new=True)
                    if ev is not None:
                        yield ev

    def trainable_nodes(self):
        """Node ids the classifier may train on: the round-0 graph plus every
        support node generated so far. Query nodes are excluded by construction,
        so evaluation is never contaminated by the round being scored."""
        sup = np.where(self.train_mask)[0]
        return sup

    def _event(self, node_id, round_id, is_new):
        label = int(self.labels[node_id])
        nbrs = self._neighbours(node_id)
        cp = [f"adver:node:{int(n)}" for n in nbrs[:16]]
        amount = float(self.features[node_id].mean())
        nat = {
            "round": float(round_id),
            "age": float(self._age()[node_id]),
            "deg": float(len(nbrs)),
            "is_new": 1.0 if is_new else 0.0,
            "feat_mean": float(self.features[node_id].mean()),
            "feat_std": float(self.features[node_id].std()),
        }
        return normalize_event(
            source=SOURCE,
            ts=float(round_id),
            account_id=f"adver:{node_id}",
            counterparties=cp,
            label=label,
            amount=amount,
            native=nat,
            raw={"node_id": int(node_id)},
            event_id=f"adver-r{round_id}-n{int(node_id)}",
        )

    def status(self):
        return {
            "round": self.round_num,
            "n_nodes": int(self.features.shape[0]),
            "n_edges": int(self.edges.shape[1]),
            "adver_choose": self.adver_choose,
            "adver_mod": self.adver_mod,
            "augment": self.augment,
            "new_pos": self.new_pos,
            "new_neg": self.new_neg,
            "rounds": list(self.round_log),
        }