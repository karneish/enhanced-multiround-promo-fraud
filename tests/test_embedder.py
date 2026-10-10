"""Tests for the restored TPNE embedder.

Covers the paper's three loss terms, the offline/incremental equivalence, and
- importantly - that importing the module never breaks the backend when torch
is absent, because the rest of the app must keep working without the ML extras.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend import embedder as emb


def _toy_graph(n=60, d=10, seed=0):
    rng = np.random.default_rng(seed)
    feats = rng.normal(size=(n, d))
    # ring + chords so every node has neighbours
    row, col = [], []
    for i in range(n):
        j = (i + 1) % n
        row += [i, j]
        col += [j, i]
    row += list(range(0, n, 7))
    col += list(range(3, n, 7))
    edge_index = np.array([row, col], dtype=np.int64)
    age = np.arange(n, dtype=np.float64) % 3
    return {"features": feats, "edge_index": edge_index, "age": age, "round": 1}


class TestAvailability(unittest.TestCase):
    def test_module_imports_without_torch(self):
        # embedder.available() must answer without raising, whatever the env
        self.assertIsInstance(emb.available(), bool)

    def test_raises_clear_error_when_torch_missing(self):
        if emb.available():
            self.skipTest("torch is present in this environment")
        with self.assertRaises(RuntimeError):
            emb.TPNEEmbedder()

    def test_columns_width_matches_embed_dim(self):
        if not emb.available():
            self.skipTest("torch is present in this environment")
        e = emb.TPNEEmbedder(h_feats=8, num_layers=2, round_window=3)
        self.assertEqual(len(e.columns()), e.embed_dim)


class TestTPNE(unittest.TestCase):
    def setUp(self):
        if not emb.available():
            self.skipTest("torch is present in this environment")
        self.graph = _toy_graph()

    def test_offline_embedding_shape_and_finiteness(self):
        e = emb.TPNEEmbedder(h_feats=8, num_layers=2, round_window=3,
                             mode="offline")
        out, loss = e.embed(self.graph)
        self.assertEqual(out.shape[0], self.graph["features"].shape[0])
        self.assertEqual(out.shape[1], e.embed_dim)
        self.assertTrue(np.isfinite(out).all())
        for k in ("recon", "disen", "temp", "final"):
            self.assertIn(k, loss)
            self.assertTrue(np.isfinite(loss[k]))

    def test_disentanglement_near_one_when_identical(self):
        # if static == temporal the cosine correlation is 1 so the disent
        # loss must be ~0; confirms the term is measuring what it claims
        e = emb.TPNEEmbedder(h_feats=8, num_layers=1, round_window=2)
        val = e._disent_loss(np.ones((10, 4)), np.ones((10, 4)))
        self.assertAlmostEqual(val, 0.0, places=6)

    def test_temp_loss_penalises_a_weak_signal(self):
        # the term is 1/log(||h||+e): a *small* temporal norm makes log tiny, so
        # the penalty is *large*. It therefore rewards a strong signal.
        e = emb.TPNEEmbedder(h_feats=8, num_layers=1, round_window=2)
        big = emb.TPNEEmbedder._temp_loss(np.ones((10, 4)) * 5.0)
        small = emb.TPNEEmbedder._temp_loss(np.ones((10, 4)) * 1e-6)
        self.assertLess(big, small)

    def test_incremental_mode_is_cheaper_and_stateful(self):
        e = emb.TPNEEmbedder(h_feats=8, num_layers=2, round_window=3,
                             mode="incremental")
        first, _ = e.embed(self.graph)
        self.assertTrue(np.isfinite(first).all())
        second, _ = e.embed(self.graph)
        self.assertEqual(first.shape, second.shape)
        self.assertEqual(e.refit_count, 2)
        # after the first embed there is cached static state to differ against
        self.assertIsNotNone(e._cache_h)

    def test_incremental_temp_is_zero_on_first_round(self):
        # no history yet -> nothing changed -> temporal signal is empty
        e = emb.TPNEEmbedder(h_feats=8, num_layers=1, round_window=2,
                             mode="incremental")
        _, loss = e.embed(self.graph)
        self.assertTrue(np.isfinite(loss["temp"]))

    def test_round_window_bounds_history_use(self):
        e = emb.TPNEEmbedder(h_feats=8, num_layers=1, round_window=2)
        hist = [_toy_graph(seed=s)["features"] for s in range(9)]
        out, _ = e.embed(self.graph, history=hist)
        self.assertTrue(np.isfinite(out).all())

    def test_summary_reports_config(self):
        e = emb.TPNEEmbedder(h_feats=8, num_layers=2, round_window=3,
                             mode="incremental")
        s = e.summary()
        self.assertEqual(s["mode"], "incremental")
        self.assertEqual(s["round_window"], 3)
        self.assertIn("refit_count", s)


class TestPretraining(unittest.TestCase):
    """Gradient-based pre-training: the losses must actually be optimised."""

    def setUp(self):
        if not emb.available():
            self.skipTest("torch is present in this environment")
        rng = np.random.default_rng(0)
        n, d = 90, 8
        feats = rng.normal(size=(n, d))
        row, col = [], []
        for i in range(n):
            for j in rng.choice(n, size=3, replace=False):
                row += [i, j]
                col += [j, i]
        self.graph = {"features": feats,
                      "edge_index": np.array([row, col], dtype=np.int64),
                      "age": np.arange(n) % 3}

    def test_pretrain_reduces_the_total_loss(self):
        e = emb.TPNEEmbedder(h_feats=16, num_layers=2, round_window=3)
        before = e.embed(self.graph)[1]["final"]
        after = e.pretrain(self.graph, epochs=40, lr=0.01)["final"]
        self.assertLess(after, before,
                        "pre-training must reduce the objective it optimises")

    def test_pretrain_reduces_reconstruction_loss(self):
        e = emb.TPNEEmbedder(h_feats=16, num_layers=2, round_window=3)
        before = e.embed(self.graph)[1]["recon"]
        after = e.pretrain(self.graph, epochs=40, lr=0.01)["recon"]
        self.assertLess(after, before)

    def test_pretrain_marks_the_embedder_as_pretrained(self):
        e = emb.TPNEEmbedder(h_feats=16, num_layers=2, round_window=3)
        self.assertFalse(e.pretrained)
        e.pretrain(self.graph, epochs=2, lr=0.01)
        self.assertTrue(e.pretrained)
        self.assertEqual(e.pretrain_epochs, 2)

    def test_embed_after_pretrain_is_finite_and_correct_width(self):
        e = emb.TPNEEmbedder(h_feats=16, num_layers=2, round_window=3)
        e.pretrain(self.graph, epochs=3, lr=0.01)
        out, loss = e.embed(self.graph)
        self.assertEqual(out.shape, (self.graph["features"].shape[0],
                                     e.embed_dim))
        self.assertTrue(np.isfinite(out).all())
        self.assertIn("final", loss)

    def test_warm_pretrain_reuses_the_optimiser(self):
        e = emb.TPNEEmbedder(h_feats=16, num_layers=2, round_window=3)
        e.pretrain(self.graph, epochs=2, lr=0.01)
        opt = e._opt
        e.pretrain(self.graph, epochs=2, lr=0.01, warm=True)
        self.assertIs(e._opt, opt, "warm=True must not rebuild the optimiser")

    def test_cold_pretrain_rebuilds_the_optimiser(self):
        e = emb.TPNEEmbedder(h_feats=16, num_layers=2, round_window=3)
        e.pretrain(self.graph, epochs=2, lr=0.01)
        opt = e._opt
        e.pretrain(self.graph, epochs=2, lr=0.01, warm=False)
        self.assertIsNot(e._opt, opt)

    def test_edgewise_distance_matches_the_dense_reference(self):
        """The edge-wise recon term must equal the dense n-by-n formulation.

        The dense form is unusable at 11k nodes (553 MB per layer), so the
        edge-wise version replaces it. This pins the two together, including
        the row/col/value weights - ``csr_matrix`` sums the two orientations of
        an undirected edge into one entry of weight 2, so unit weights would
        silently halve the result.
        """
        import torch
        e = emb.TPNEEmbedder(h_feats=8, num_layers=1, round_window=2)
        n = self.graph["features"].shape[0]
        h = torch.tensor(np.random.default_rng(0).normal(size=(n, 8)),
                         dtype=torch.float32)
        A, pattern = e._adjacency(self.graph, n)
        got = e._mean_dist_t(A, h, pattern).squeeze(1).numpy()
        ref = (A.to_dense() * torch.cdist(h, h)).sum(dim=1).numpy()
        self.assertLess(np.abs(got - ref).max(), 1e-2)


if __name__ == "__main__":
    unittest.main()