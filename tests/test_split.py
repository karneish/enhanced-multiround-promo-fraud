"""Guards the round-0 split used by ``upgraded_tpne.py``.

These tests exist because of a real bug. In ``--split masks`` mode the
training set was built as ``X0[:i_tr.max() + 1]`` and validation as
``slice(i_va.min(), i_va.max() + 1)``. The author's train/val masks select an
arbitrary non-contiguous subset of nodes, so ``i_tr.max()`` lands near the end
of the node range. Both slices therefore contained ~100% of the retention set:

    train_slice n test overlap : 5877 / 5879
    val_arr     n test overlap : 5879 / 5879

That is train-on-test, and it inflated the reported round-0 F1 (an
``--features embed`` run reported ~94 F1 where the honest figure is ~0.53).
The fix consumes the masks as index arrays. These tests pin that behaviour
down so the slice form cannot quietly come back.
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import upgraded_tpne  # noqa: E402

N_NODES = 11758


class _FakeGraph(dict):
    """Only ``graph['label']`` is consulted; node count comes from the dataset."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        super().__init__(label=(rng.random(N_NODES) < 0.218).astype(int))


class TestAuthorSplitDisjointness(unittest.TestCase):
    def test_partitions_are_pairwise_disjoint(self):
        i_tr, i_va, i_te = upgraded_tpne.author_split(_FakeGraph(7), seed=7)
        self.assertFalse(set(i_tr) & set(i_va))
        self.assertFalse(set(i_tr) & set(i_te))
        self.assertFalse(set(i_va) & set(i_te))

    def test_sizes_match_the_documented_protocol(self):
        i_tr, i_va, i_te = upgraded_tpne.author_split(_FakeGraph(7), seed=7)
        self.assertEqual(len(i_tr), 3527)      # 0.60 * 5879 pool
        self.assertEqual(len(i_va), 2352)      # 0.40 * pool
        self.assertEqual(len(i_te), 5879)      # complement of the pool
        self.assertEqual(len(i_tr) + len(i_va) + len(i_te), N_NODES)


class TestNaiveSliceWouldLeak(unittest.TestCase):
    """Documents why the slice form was unsafe, so the reason stays on record."""

    def test_reduction_to_a_range_swallows_the_retention_set(self):
        i_tr, i_va, i_te = upgraded_tpne.author_split(_FakeGraph(42), seed=42)
        slice_train = set(range(int(i_tr.max()) + 1))
        slice_val = set(range(int(i_va.min()), int(i_va.max()) + 1))
        test = set(i_te)
        self.assertGreater(len(slice_train & test) / len(test), 0.99)
        self.assertGreater(len(slice_val & test) / len(test), 0.99)

    def test_index_arrays_do_not(self):
        i_tr, i_va, i_te = upgraded_tpne.author_split(_FakeGraph(42), seed=42)
        self.assertFalse(set(i_tr) & set(i_te))
        self.assertFalse(set(i_va) & set(i_te))


class TestPipelineUsesIndexArrays(unittest.TestCase):
    def test_head_is_fit_on_train_indices(self):
        src = open(os.path.join(ROOT, "scripts", "upgraded_tpne.py")).read()
        self.assertIn("head.fit(X0[train_idx], y_all[train_idx])", src)
        self.assertNotIn("X0[:split]", src)

    def test_threshold_is_tuned_on_val_indices(self):
        src = open(os.path.join(ROOT, "scripts", "upgraded_tpne.py")).read()
        self.assertIn("head, X0[val_idx], y_all[val_idx])", src)
        self.assertNotIn("X0[val_slice]", src)


if __name__ == "__main__":
    unittest.main()
