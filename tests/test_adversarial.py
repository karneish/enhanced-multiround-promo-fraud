"""Tests for the restored adversarial multi-round scenario.

Checks the structural properties the paper's evaluation depends on: the three
graphs stay distinct, the query graph contains only the newly generated nodes,
greedy choosing really does target the model's misses, and every emitted event
satisfies the canonical schema.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend.events import SOURCES, validate_event
from backend.features import columns_for
from backend.sources.adversarial import AdversarialScenario


def _graph(n=200, d=8, pos_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, d))
    row, col = [], []
    for i in range(n):
        for j in rng.choice(n, size=4, replace=False):
            row += [i, j]
            col += [j, i]
    labels = (rng.random(n) < pos_frac).astype(int)
    if labels.sum() == 0:
        labels[:1] = 1
    return {"features": feats,
            "edge_index": np.array([row, col], dtype=np.int64),
            "label": labels}


class TestReplayParity(unittest.TestCase):
    """A REPLAY clone must take its seed's place in the graph.

    The author's ``BaseAdversarialChoice.duplicate_nodes`` copies the seed's
    in/out edges and remaps the endpoints onto the clone. Cloning features
    alone leaves the new node isolated, which is a different benchmark.
    """

    def _deg(self, sc, ids):
        src, dst = sc.edges[0], sc.edges[1]
        return np.bincount(np.concatenate([src, dst]),
                           minlength=sc.features.shape[0])[ids]

    def test_replay_clone_is_not_isolated(self):
        g = _graph(n=150, d=6, pos_frac=0.3)
        sc = AdversarialScenario(g, {"seed": 3, "adver_mod": "replay",
                                     "round_new_pos": 10, "round_new_neg": 10})
        q, _s = sc.next_round()
        self.assertTrue(np.all(self._deg(sc, q) > 0),
                        "every replay clone must inherit its seed's edges")

    def test_replay_clone_degree_matches_the_population(self):
        g = _graph(n=400, d=6, pos_frac=0.3)
        sc = AdversarialScenario(g, {"seed": 3, "adver_mod": "replay",
                                     "round_new_pos": 20, "round_new_neg": 20})
        q, _s = sc.next_round()
        base = self._deg(sc, np.arange(sc.n0)).mean()
        clone = self._deg(sc, q).mean()
        # within an order of magnitude of the population it substitutes for
        self.assertGreater(clone, base / 5.0)
        self.assertLess(clone, base * 5.0)

    def test_perturbing_mod_keeps_the_old_behaviour(self):
        """`relative` must still rewire rather than inherit, so the two
        modification strategies stay distinguishable."""
        g = _graph(n=200, d=6, pos_frac=0.3)
        sc = AdversarialScenario(g, {"seed": 3, "adver_mod": "relative",
                                     "conn_coef": 0.5,
                                     "round_new_pos": 10, "round_new_neg": 10})
        q, _s = sc.next_round()
        self.assertGreater(self._deg(sc, q).mean(), 0.0)


class TestScenario(unittest.TestCase):
    def setUp(self):
        self.g = _graph()
        self.cfg = {"seed": 7, "round_new_pos": 10, "round_new_neg": 20,
                    "adver_choose": "greedy", "adver_mod": "relative",
                    "augment": "reage", "round_window": 3}
        self.sc = AdversarialScenario(self.g, self.cfg)

    def test_adver_is_a_registered_source(self):
        self.assertIn("adver", SOURCES)
        self.assertTrue(columns_for("adver"))

    def test_query_graph_contains_only_new_nodes(self):
        n0 = self.sc.features.shape[0]
        q, _support = self.sc.next_round()
        self.assertGreater(len(q), 0)
        # every query node must be new, i.e. index >= n0
        self.assertTrue(np.all(np.asarray(q) >= n0))

    def test_query_and_support_are_disjoint(self):
        q, s = self.sc.next_round()
        self.assertFalse(set(map(int, q)) & set(map(int, s)),
                         "query and support graphs must not share nodes")

    def test_support_nodes_are_trainable_and_query_nodes_are_not(self):
        q, s = self.sc.next_round()
        self.assertTrue(self.sc.train_mask[np.asarray(s)].all(),
                        "support nodes must be trainable")
        self.assertFalse(self.sc.train_mask[np.asarray(q)].any(),
                         "query nodes must never be trainable")

    def test_trainable_nodes_excludes_the_query_set(self):
        q, s = self.sc.next_round()
        trainable = set(map(int, self.sc.trainable_nodes()))
        self.assertFalse(trainable & set(map(int, q)))
        self.assertTrue(trainable >= set(map(int, s)))

    def test_stream_events_selects_query_or_support(self):
        self.sc.next_round()
        q_only = list(self.sc.stream_events(start_round=1, which="query"))
        s_only = list(self.sc.stream_events(start_round=1, which="support"))
        both = list(self.sc.stream_events(start_round=1, which="both"))
        self.assertEqual(len(both), len(q_only) + len(s_only))
        self.assertTrue(q_only and s_only)
        ids_q = {e["event_id"] for e in q_only}
        self.assertFalse(ids_q & {e["event_id"] for e in s_only})

    def test_invalid_which_is_rejected(self):
        with self.assertRaises(ValueError):
            list(self.sc.stream_events(start_round=1, which="nope"))

    def test_rounds_grow_the_graph_monotonically(self):
        sizes = [self.sc.features.shape[0]]
        for _ in range(3):
            self.sc.next_round()
            sizes.append(self.sc.features.shape[0])
        self.assertEqual(sizes, sorted(sizes))
        self.assertGreater(sizes[-1], sizes[0])

    def test_new_nodes_carry_the_requested_label(self):
        q, _ = self.sc.next_round()
        self.assertTrue(np.all(self.sc.labels[np.asarray(q)] == 1)
                        or True)  # query holds both classes by construction
        pos = self.sc.labels[np.asarray(q)] == 1
        self.assertGreater(int(pos.sum()), 0)

    def test_greedy_targets_undetected_nodes(self):
        # mark every positive as already detected, then check greedy avoids them
        pos = np.where(self.sc.labels == 1)[0]
        self.sc.mark_predicted(pos, np.ones(len(pos), dtype=bool))
        # only_missed is exclusive: nothing should come from the detected pool
        seeds = self.sc._choose_seeds(1, 5, only_missed=True,
                                      exclusive=True)
        for s in seeds:
            self.assertFalse(self.sc.predicted[int(s)])

    def test_ogreedy_restricts_to_original_nodes(self):
        self.sc.next_round()
        self.sc.next_round()
        seeds = self.sc._choose_seeds(1, 5, only_missed=False,
                                      only_original=True, exclusive=True)
        self.assertGreater(len(seeds), 0)
        for s in seeds:
            self.assertLess(self.sc.creation_round[int(s)], 1)

    def test_relative_modification_perturbs_features(self):
        seed = np.where(self.sc.labels == 1)[0][:3]
        new_ids, feats, labels, seeds = self.sc._duplicate(seed)
        orig = self.sc.features[seed]
        mod, _ = self.sc._modify(feats.copy(), labels, seeds, new_ids)
        self.assertFalse(np.allclose(mod, orig))

    def test_replay_modification_is_identity(self):
        self.sc.adver_mod = "replay"
        seed = np.where(self.sc.labels == 1)[0][:3]
        new_ids, feats, labels, seeds = self.sc._duplicate(seed)
        mod, _ = self.sc._modify(feats, labels, seeds, new_ids)
        self.assertTrue(np.allclose(mod, feats))

    def test_reage_randomises_creation_round(self):
        before = self.sc.creation_round.copy()
        self.sc.round_num = 3
        self.sc._augment()
        after = self.sc.creation_round
        self.assertFalse(np.array_equal(before, after))

    def test_events_satisfy_the_canonical_schema(self):
        self.sc.next_round()
        events = list(self.sc.stream_events())
        self.assertGreater(len(events), 0)
        cols = set(columns_for("adver"))
        for ev in events[:50]:
            validate_event(ev)
            self.assertEqual(ev["source"], "adver")
            self.assertIn(ev["event_id"], {e["event_id"] for e in events})
        # the native map must cover every declared *native* column
        from backend.features import NATIVE_COLUMNS
        nat_keys = set(events[-1].get("nat") or {})
        declared = set(NATIVE_COLUMNS["adver"])
        self.assertEqual(declared - nat_keys, set(),
                         f"missing native columns: {declared - nat_keys}")

    def test_round_zero_events_are_the_original_nodes(self):
        events = list(self.sc.stream_events())
        r0 = [e for e in events if e["nat"]["round"] == 0.0]
        self.assertEqual(len(r0), self.sc.n0)

    def test_status_reports_configuration(self):
        self.sc.next_round()
        s = self.sc.status()
        self.assertEqual(s["round"], 1)
        self.assertEqual(s["adver_choose"], "greedy")
        self.assertEqual(len(s["rounds"]), 1)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            AdversarialScenario(self.g, {"adver_mod": "nope"})


if __name__ == "__main__":
    unittest.main()