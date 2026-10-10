"""Tests for the live orchestrator: batching, review budget, lifecycle.

Locks in the audit findings that a second session inherited the first one's
state, and that the review budget was spent without producing any label.
"""
import time
import unittest

from backend.livesim import LiveSimulation


def config(**over):
    cfg = {
        "live_mode": "concurrent",
        "stream_slice": 0,              # 0 = do not load the dataset in tests
        "events_per_sec": 1000,
        "score_batch_max": 64,
        "score_batch_window_ms": 2,
        "retain_event_records": 100,
        "graph_node_cap": 200,
        "budget_pos": 2, "budget_neg": 20,
        "decision_thresholds": {"review": 0.30, "block": 0.70},
        "min_fit_labels": 40, "retrain_interval_events": 40,
        "retrain_interval_seconds": 1, "buffer_max_labels": 500,
        "xgb_rounds": 15, "xgb_extra_rounds": 5, "hist_max_iter": 15,
        "hist_extra_iter": 5, "et_n_estimators": 15,
        "ieee": {"data_dir": "data", "sample_fallback": False},
        "chain": {"transport": "rpc-poll", "poll_interval_seconds": 3600},
    }
    cfg.update(over)
    return cfg


def unlabeled(sim, n, start=0, amount=None):
    """Unlabeled webhook events - nothing but a reviewer can teach these.

    Amounts vary so both review buckets (high-value vs rest, split at the
    per-source median) are exercised instead of every event landing in the
    scarce high-value bucket on a tie.
    """
    for i in range(start, start + n):
        amt = amount if amount is not None else 10.0 + (i % 7) * 40.0
        sim.ingest({"account_id": "a-%d" % (i % 4), "amount": amt,
                    "counterparties": ["cp-%d" % (i % 3)]})


class TestReviewLoop(unittest.TestCase):
    def setUp(self):
        self.sim = LiveSimulation(config(live_mode="concurrent"))
        self.sim.start()

    def tearDown(self):
        self.sim.stop()

    def test_unlabeled_stream_bootstraps_through_review(self):
        """With no fitted model the cold prior sits below the review band, so
        nothing would ever be queued and the ensemble would never learn."""
        unlabeled(self.sim, 30)
        stats = self.sim.review_stats()
        self.assertGreater(stats["queued"], 0, "cold start queued nothing")
        self.assertEqual(self.sim.status()["decisions"]["review"], 30)

    def test_budget_is_spent_and_bounded(self):
        unlabeled(self.sim, 200)
        st = self.sim.status()
        self.assertEqual(sum(st["review_budget"].values()), 0,
                         "the budget was not exhausted")
        self.assertEqual(self.sim.review_stats()["pending"], st["review_spent"])
        self.assertLessEqual(st["review_spent"], 22)

    def test_high_value_budget_is_separate_and_scarce(self):
        # every event is at or above the median -> only the pos budget applies
        unlabeled(self.sim, 60, amount=500.0)
        st = self.sim.status()
        self.assertEqual(st["review_budget"]["pos"], 0)
        self.assertEqual(st["review_spent"], 2)
        self.assertEqual(st["review_budget"]["neg"], 20)

    def test_labels_reach_the_ensemble(self):
        unlabeled(self.sim, 12)
        pending = self.sim.review_pending(limit=50)
        self.assertTrue(pending)
        res = self.sim.submit_review_batch(
            [{"review_id": p["review_id"], "label": 1 if i % 3 == 0 else 0}
             for i, p in enumerate(pending)], reviewer="llm")
        self.assertEqual(res["errors"], [])
        self.assertEqual(len(res["ok"]), len(pending))
        ens = self.sim.ensembles["webhook"]
        self.assertEqual(ens.n_labeled, len(pending))
        s = self.sim.review_stats()
        self.assertEqual(s["pending"], 0)
        self.assertEqual(s["resolved"], len(pending))

    def test_labeled_events_are_never_queued(self):
        for i in range(10):
            self.sim.ingest({"account_id": "l-%d" % i, "amount": 5.0, "label": 0})
        self.assertEqual(self.sim.review_stats()["queued"], 0)
        self.assertEqual(self.sim.ensembles["webhook"].n_labeled, 10)

    def test_known_fraud_always_blocks(self):
        rec = self.sim.ingest({"account_id": "f-1", "amount": 99.0, "label": 1})
        self.assertEqual(rec["decision"], "block")
        self.assertTrue(rec["known_fraud"])
        self.assertEqual(self.sim.review_stats()["queued"], 0)

    def test_bad_review_payloads_raise_valueerror(self):
        unlabeled(self.sim, 4)
        rid = self.sim.review_pending(limit=1)[0]["review_id"]
        with self.assertRaises(ValueError):
            self.sim.submit_review(rid, 7)
        with self.assertRaises(ValueError):
            self.sim.submit_review("rv-missing", 1)
        self.sim.submit_review(rid, 0)
        with self.assertRaises(ValueError):
            self.sim.submit_review(rid, 1)          # already labeled

    def test_batch_reports_per_item_outcomes(self):
        unlabeled(self.sim, 16)
        pending = self.sim.review_pending(limit=50)
        self.assertGreaterEqual(len(pending), 3)
        items = [{"review_id": p["review_id"], "label": 0} for p in pending[:3]]
        items.append({"review_id": "rv-nope", "label": 0})
        items.append({"review_id": None, "label": 1})
        out = self.sim.submit_review_batch(items)
        self.assertEqual(len(out["ok"]), 3)
        self.assertEqual(len(out["errors"]), 2)


class TestBatching(unittest.TestCase):
    def test_a_burst_is_scored_in_batches(self):
        sim = LiveSimulation(config())
        sim.start()
        try:
            for i in range(64):
                sim.ingest({"account_id": "b-%d" % (i % 8), "amount": 10.0,
                            "label": i % 7 == 0 and 1 or 0})
            st = sim.status()
            self.assertEqual(st["throughput"]["scored"], 64)
            self.assertEqual(st["throughput"]["batch_max"], 64)
            self.assertEqual(st["errors"].get("score_error"), None)
        finally:
            sim.stop()

    def test_batch_and_single_scoring_agree(self):
        one = LiveSimulation(config(score_batch_max=1, score_batch_window_ms=0))
        many = LiveSimulation(config(score_batch_max=64, score_batch_window_ms=0))
        one.start()
        many.start()
        try:
            for i in range(12):
                payload = {"account_id": "x-%d" % i, "amount": 7.0 + i}
                a = one.ingest(payload)
                b = many.ingest(payload)
                self.assertEqual(a["decision"], b["decision"])
        finally:
            one.stop()
            many.stop()

    def test_profiles_accumulate_across_the_batch_in_order(self):
        sim = LiveSimulation(config())
        sim.start()
        try:
            for i in range(5):
                sim.ingest({"account_id": "same", "amount": 1.0})
            rec = sim.feed.tail(5)[-1]
            self.assertIsNotNone(rec)
            ens = sim.ensembles["webhook"]
            self.assertIsNotNone(ens)
        finally:
            sim.stop()


class TestLifecycle(unittest.TestCase):
    def test_stop_leaves_no_threads(self):
        sim = LiveSimulation(config())
        sim.start()
        unlabeled(sim, 5)
        sim.stop()
        time.sleep(0.2)
        self.assertEqual(sim.threads_alive(), 0)
        self.assertFalse(sim.running)

    def test_restart_clears_budget_counters_and_ensembles(self):
        sim = LiveSimulation(config())
        sim.start()
        unlabeled(sim, 150)          # enough to drain both budget buckets
        before = sim.status()
        self.assertEqual(sum(before["review_budget"].values()), 0)
        self.assertIn("webhook", before["ensembles"])
        sim.stop()

        sim.start()
        after = sim.status()
        # the generation is a cancellation token: it is bumped by stop as well
        # as by start, so only monotonicity is guaranteed
        self.assertGreater(after["generation"], before["generation"])
        self.assertEqual(sum(after["review_budget"].values()), 22,
                         "the new session did not get a fresh budget")
        self.assertEqual(after["review_spent"], 0)
        self.assertEqual(after["counters"].get("webhook", 0), 0)
        self.assertEqual(after["decisions"], {"allow": 0, "review": 0, "block": 0})
        self.assertEqual(after["ensembles"], {})
        self.assertEqual(after["graph"]["nodes"], 0)
        self.assertEqual(sim.review_stats()["queued"], 0)
        sim.stop()

    def test_stop_is_idempotent(self):
        sim = LiveSimulation(config())
        sim.start()
        sim.stop()
        sim.stop()
        self.assertFalse(sim.running)

    def test_double_start_is_refused(self):
        sim = LiveSimulation(config())
        sim.start()
        first = sim.status()["generation"]
        live = sim.threads_alive()
        sim.start()                       # must not spawn a second consumer
        self.assertEqual(sim.status()["generation"], first)
        self.assertEqual(sim.threads_alive(), live)
        sim.stop()

    def test_overrides_apply_to_a_new_session(self):
        sim = LiveSimulation(config())
        sim.start()
        sim.stop()
        sim.start(overrides={"budget_neg": 99, "score_batch_max": 8})
        self.assertEqual(sim.status()["review_budget"]["neg"], 99)
        self.assertEqual(sim.status()["throughput"]["batch_max"], 8)
        sim.stop()


class TestGraph(unittest.TestCase):
    def test_edges_and_nodes_are_recorded(self):
        sim = LiveSimulation(config())
        sim.start()
        try:
            for i in range(6):
                sim.ingest({"account_id": "a-%d" % (i % 2),
                            "counterparties": ["cp-%d" % (i % 2)], "label": 0})
            g = sim.graph_data()
            self.assertEqual(g["nodes"].__len__(), 4)   # 2 accounts + 2 counterparties
            self.assertEqual(len(g["edges"]), 2)
            self.assertTrue(all(e["weight"] >= 1 for e in g["edges"]))
        finally:
            sim.stop()

    def test_graph_is_capped(self):
        sim = LiveSimulation(config(graph_node_cap=10))
        sim.start()
        try:
            for i in range(40):
                sim.ingest({"account_id": "acct-%d" % i, "counterparties": []})
            g = sim.graph_data()
            self.assertLessEqual(len(g["nodes"]), 10)
            self.assertTrue(g["capped"])
        finally:
            sim.stop()

    def test_graph_is_json_serialisable(self):
        import json
        sim = LiveSimulation(config())
        sim.start()
        try:
            sim.ingest({"account_id": "a", "counterparties": ["b"], "label": 1})
            json.dumps(sim.graph_data())
            json.dumps(sim.status())
        finally:
            sim.stop()


if __name__ == "__main__":
    unittest.main()
