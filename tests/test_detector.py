"""Regression tests for the adaptive ensemble.

Locks in the audit findings:
  * `balanced_window` grouped rows by class, so the trailing holdout slice
    could contain a single class and silently zero every F1
  * the weights were pinned to a uniform 1/3 fallback
  * the refit ran while holding the lock the scoring path needs, so a multi-
    second fit stalled the stream
  * `fit_needed` honoured only one of the two retrain cadences
"""
import threading
import time
import unittest

import numpy as np

from backend.detector import (HOLDOUT_FRACTION, MIN_HOLDOUT_PER_CLASS,
                              MODEL_NAMES, WEIGHT_FLOOR_FRACTION,
                              ThreeModelEnsemble)

FAST = {"seed": 42, "min_fit_labels": 10, "retrain_interval_events": 50,
        "retrain_interval_seconds": 1, "buffer_max_labels": 5000,
        "xgb_rounds": 20, "xgb_extra_rounds": 5, "hist_max_iter": 20,
        "hist_extra_iter": 5, "et_n_estimators": 20}


def separable(n=600, seed=0, n_feat=12):
    """Deterministic, learnable, mildly imbalanced data (not perfectly
    separable, so the per-model F1s differ and the weights can be compared)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_feat))
    signal = 1.7 * X[:, 0] + 1.1 * X[:, 1] + 0.6 * rng.normal(size=n)
    y = (signal > 0.75).astype(int)
    if y.sum() < 20 or y.sum() > n - 20:   # keep both classes healthy
        y[: n // 10] = 1
    return X, y


def train(ens, X, y):
    for i in range(len(y)):
        ens.add_labeled(X[i], int(y[i]))


class TestBalancedWindow(unittest.TestCase):
    def test_arrival_order_is_preserved(self):
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        X, y = separable(400)
        train(ens, X, y)
        Xw, yw = ens.balanced_window()
        # rows must still be in the order they arrived, not grouped by class
        self.assertEqual(len(yw), len(y))
        pos = np.where(yw == 1)[0]
        neg = np.where(yw == 0)[0]
        self.assertTrue(np.all(np.diff(pos) > 0))
        self.assertTrue(np.all(np.diff(neg) > 0))
        self.assertTrue(pos[-1] > neg[0], "a class-grouped window is the bug")

    def test_majority_class_is_downsampled(self):
        cfg = dict(FAST)
        ens = ThreeModelEnsemble("ieee", ["f"], cfg)
        X = np.zeros((900, 1))
        y = np.array([1] * 60 + [0] * 840)
        train(ens, X, y)
        _, yw = ens.balanced_window()
        n_pos, n_neg = int((yw == 1).sum()), int((yw == 0).sum())
        # contract: keep 2:1, but never fewer than 500 negatives
        self.assertEqual(n_pos, 60)
        self.assertLessEqual(n_neg, 500)
        self.assertEqual(n_neg, min(840, max(2 * 60, 500)))

    def test_downsample_actually_reduces_a_large_imbalance(self):
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        X = np.zeros((3000, 1))
        y = np.array([1] * 100 + [0] * 2900)
        train(ens, X, y)
        _, yw = ens.balanced_window()
        self.assertEqual(len(yw), 100 + 500)

    def test_single_class_window_is_returned_untouched(self):
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        train(ens, np.zeros((50, 1)), np.zeros(50, dtype=int))
        Xw, yw = ens.balanced_window()
        self.assertEqual(len(yw), 50)


class TestHoldout(unittest.TestCase):
    def test_trailing_slice_is_temporal_when_both_classes_present(self):
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        X, y = separable(600)
        train(ens, X, y)
        idx, mode, degen = ens._holdout_indices(y)
        self.assertEqual(mode, "temporal")
        self.assertFalse(degen)
        self.assertTrue(np.all(np.diff(idx) > 0), "holdout must be a contiguous tail")
        self.assertGreaterEqual(len(idx), int(len(y) * HOLDOUT_FRACTION) - 1)

    def test_bursty_window_falls_back_to_stratified(self):
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        y = np.array([0] * 200 + [1] * 100)      # all fraud at the tail
        idx, mode, degen = ens._holdout_indices(y)
        self.assertEqual(mode, "stratified")
        self.assertTrue(degen)
        self.assertGreaterEqual(int((y[idx] == 1).sum()), MIN_HOLDOUT_PER_CLASS)
        self.assertGreaterEqual(int((y[idx] == 0).sum()), MIN_HOLDOUT_PER_CLASS)

    def test_too_few_of_a_class_scores_on_everything(self):
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        y = np.array([0] * 200 + [1])
        idx, mode, degen = ens._holdout_indices(y)
        self.assertIsNone(idx)
        self.assertEqual(mode, "none")

    def test_f1_is_not_silently_zero_on_a_bursty_window(self):
        ens = ThreeModelEnsemble("ieee", ["f%d" % i for i in range(6)],
                                  dict(FAST, et_n_estimators=30))
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 6))
        # a burst of fraud filling the whole trailing 20% slice
        y = np.array([0] * 240 + [1] * 60)
        train(ens, X, y)
        snap = ens.fit()
        self.assertEqual(snap["holdout_mode"], "stratified")
        self.assertTrue(snap["holdout_degenerate"])
        for name in MODEL_NAMES:
            f1 = (snap["models"].get(name) or {}).get("f1")
            self.assertIsNotNone(f1, "%s produced no F1" % name)
            self.assertGreater(f1, 0.0, "%s F1 collapsed to 0" % name)


class TestWeights(unittest.TestCase):
    def test_weights_are_adaptive_and_floored(self):
        ens = ThreeModelEnsemble("ieee", ["f"], dict(FAST, et_n_estimators=60))
        X, y = separable(900)
        train(ens, X, y)
        snap = ens.fit()
        w = snap["weights"]
        self.assertEqual(set(w), set(MODEL_NAMES))
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        f1s = {n: (snap["models"].get(n) or {}).get("f1", 0.0)
               for n in MODEL_NAMES}
        best = max(f1s.values())
        for name, val in w.items():
            # the floor is relative to the best model, not an absolute constant
            self.assertGreaterEqual(val + 1e-9,
                                    WEIGHT_FLOOR_FRACTION * 0.5 / 3,
                                    "%s was floored out entirely" % name)
        self.assertGreater(len(set(round(v, 4) for v in w.values())), 1,
                           "weights collapsed to a uniform 1/3 split")
        # a clearly better model must get a clearly larger share
        order = sorted(f1s, key=lambda n: f1s[n])
        self.assertLessEqual(w[order[0]], w[order[-1]])
        if best > 0:
            ratio = w[order[-1]] / max(w[order[0]], 1e-12)
            self.assertGreater(ratio, 1.0, "weights do not track F1 at all")

    def test_weights_sum_to_one_when_a_model_has_zero_f1(self):
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        w = ens._weights_from({"XGBoost": {"f1": 0.0}, "ExtraTrees": {"f1": 1.0}},
                              ["XGBoost", "ExtraTrees"])
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        self.assertGreater(w["XGBoost"], 0.0, "a collapsed model was zeroed out")
        self.assertAlmostEqual(w["XGBoost"], WEIGHT_FLOOR_FRACTION /
                               (1.0 + WEIGHT_FLOOR_FRACTION), places=6)
        self.assertGreater(w["ExtraTrees"], 0.5)

    def test_weights_are_invariant_to_the_f1_scale(self):
        """Only the *ratios* between models may matter, so doubling every F1
        must not change the blend."""
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        names = ["XGBoost", "ExtraTrees"]
        small = ens._weights_from({"XGBoost": {"f1": 0.10},
                                   "ExtraTrees": {"f1": 0.20}}, names)
        large = ens._weights_from({"XGBoost": {"f1": 0.50},
                                   "ExtraTrees": {"f1": 1.00}}, names)
        self.assertAlmostEqual(small["XGBoost"], large["XGBoost"], places=9)

    def test_all_zero_f1_falls_back_to_an_even_blend(self):
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        names = ["XGBoost", "ExtraTrees", "HistGradientBoosting"]
        w = ens._weights_from({n: {"f1": 0.0} for n in names}, names)
        for name in names:
            self.assertAlmostEqual(w[name], 1.0 / 3, places=9)

    def test_no_available_models_gives_no_weights(self):
        self.assertIsNone(ThreeModelEnsemble("x", ["f"], FAST)._weights_from({}, []))


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.ens = ThreeModelEnsemble("ieee", ["f%d" % i for i in range(12)], FAST)
        X, y = separable(600)
        train(self.ens, X, y)
        self.ens.fit()
        self.X = X

    def test_batch_matches_row_by_row(self):
        rows = self.X[:25]
        batched = self.ens.predict_proba_batch(rows)
        singles = [self.ens.predict_proba(r) for r in rows]
        self.assertEqual(len(batched), len(singles))
        for b, s in zip(batched, singles):
            self.assertAlmostEqual(b, s, places=6)

    def test_single_row_shape_is_preserved(self):
        self.assertIsInstance(self.ens.predict_proba(self.X[0]), float)
        self.assertIsInstance(self.ens.predict_proba_batch(self.X[:1])[0], float)

    def test_probabilities_are_in_range(self):
        probs = self.ens.predict_proba_batch(self.X)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_batch_is_far_faster_than_row_by_row(self):
        rows = self.X[:200]
        t0 = time.time()
        self.ens.predict_proba_batch(rows)
        batched = time.time() - t0
        t0 = time.time()
        for r in rows:
            self.ens.predict_proba(r)
        singles = time.time() - t0
        self.assertLess(batched, singles,
                        "batching must not be slower than per-row scoring")

    def test_cold_start_prior_is_clamped(self):
        cold = ThreeModelEnsemble("ieee", ["f"], FAST)
        p = cold.predict_proba_batch(np.zeros((3, 1)))
        self.assertEqual(p, [cold._cold_start_proba()] * 3)
        self.assertTrue(all(0.10 <= v <= 0.5 for v in p))


class TestFitCadence(unittest.TestCase):
    def test_needs_enough_labels_first(self):
        ens = ThreeModelEnsemble("ieee", ["f"], dict(FAST, min_fit_labels=120))
        train(ens, np.zeros((50, 1)), np.zeros(50, dtype=int))
        self.assertFalse(ens.fit_needed(time.time()))

    def test_either_cadence_triggers(self):
        cfg = dict(FAST, retrain_interval_events=50, retrain_interval_seconds=60)
        ens = ThreeModelEnsemble("ieee", ["f"], cfg)
        X, y = separable(300)
        train(ens, X, y)
        base = 1_000_000.0
        self.assertTrue(ens.fit_needed(base))
        snap = ens.fit(now_ts=base)
        self.assertEqual(snap["last_fit_at"], base)
        ens.add_labeled(X[0], 1)
        # 1 buffered label, and the clock has barely moved: neither cadence
        self.assertFalse(ens.fit_needed(base + 1.0))
        # wall-clock cadence alone is enough
        self.assertTrue(ens.fit_needed(base + 61.0))
        # event cadence alone is enough
        for i in range(60):
            ens.add_labeled(X[i], int(y[i]))
        self.assertTrue(ens.fit_needed(base + 1.0))

    def test_nothing_buffered_means_no_refit(self):
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        X, y = separable(300)
        train(ens, X, y)
        ens.fit()
        self.assertFalse(ens.fit_needed(time.time() + 10_000))


class TestBackgroundRefit(unittest.TestCase):
    def test_refit_does_not_block_scoring(self):
        """The regression that mattered: a fit holding the ensemble lock made
        `predict` wait for the whole training run."""
        cfg = dict(FAST, xgb_rounds=250, hist_max_iter=250, et_n_estimators=250)
        ens = ThreeModelEnsemble("ieee", ["f%d" % i for i in range(12)], cfg)
        X, y = separable(1200)
        train(ens, X, y)

        done = threading.Event()

        def score():
            while not done.is_set():
                ens.predict_proba_batch(X[:64])
            done.set()

        worker = threading.Thread(target=score, daemon=True)
        worker.start()
        time.sleep(0.05)
        t0 = time.time()
        self.assertTrue(ens.request_refit())
        queued_latency = time.time() - t0
        done.set()
        worker.join(timeout=5)
        total = time.time() - t0
        self.assertLess(queued_latency, 0.5,
                        "request_refit() itself must return immediately")
        # wait for the fit to finish and compare against its own duration
        for _ in range(200):
            snap = ens.snapshot()
            if snap["refit_count"] and not snap["refit_in_progress"]:
                break
            time.sleep(0.1)
        snap = ens.snapshot()
        self.assertTrue(snap["refit_count"], "the background refit never ran")
        self.assertIsNotNone(snap["last_refit_seconds"])
        self.assertLess(total, snap["last_refit_seconds"] + 2.0,
                        "scoring was blocked for the whole fit")

    def test_only_one_refit_at_a_time(self):
        ens = ThreeModelEnsemble("ieee", ["f%d" % i for i in range(12)], FAST)
        X, y = separable(400)
        train(ens, X, y)
        first = ens.request_refit()
        second = ens.request_refit()
        self.assertTrue(first)
        self.assertFalse(second, "a second refit was queued behind the first")
        for _ in range(200):
            if not ens.snapshot()["refit_in_progress"]:
                break
            time.sleep(0.05)
        ens.shutdown()

    def test_models_are_replaced_not_mutated(self):
        ens = ThreeModelEnsemble("ieee", ["f%d" % i for i in range(12)], FAST)
        X, y = separable(400)
        train(ens, X, y)
        ens.fit()
        before = ens.deployed_models()
        for i in range(200):
            ens.add_labeled(X[i], int(y[i]))
        ens.fit()
        after = ens.deployed_models()
        for name in before:
            self.assertIsNot(before[name], after[name],
                             "%s was mutated in place; predictions in flight "
                             "would see a half-written model" % name)

    def test_snapshot_reports_a_usable_fit_breakdown(self):
        ens = ThreeModelEnsemble("ieee", ["f%d" % i for i in range(12)], FAST)
        X, y = separable(400)
        train(ens, X, y)
        snap = ens.fit()
        for name in MODEL_NAMES:
            self.assertIn(name, snap["last_fit_breakdown"])
            self.assertGreaterEqual(snap["last_fit_breakdown"][name], 0.0)

    def test_fit_errors_are_reported_not_raised(self):
        ens = ThreeModelEnsemble("ieee", ["f"], FAST)
        train(ens, np.zeros((200, 1)), np.zeros(200, dtype=int))   # single class
        snap = ens.fit()
        self.assertFalse(snap["fitted"])
        ens.shutdown()


if __name__ == "__main__":
    unittest.main()
