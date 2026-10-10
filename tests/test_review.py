"""Tests for the human/LLM review queue.

The loop only works if a queued candidate is (a) bounded, (b) carries the
feature vector that was actually scored, and (c) accepts exactly one label.
"""
import time
import unittest

from backend.review import ReviewQueue


def row(n=4, base=0.0):
    return [base + i for i in range(n)]


class TestQueueBasics(unittest.TestCase):
    def setUp(self):
        self.q = ReviewQueue(maxlen=5, expire_seconds=3600)

    def submit(self, i, **kw):
        return self.q.submit(source="chain", event_id="e%d" % i, x_row=row(base=i),
                             account_id="a%d" % i, amount=float(i), prob=0.4,
                             ts=time.time(), **kw)

    def test_submit_returns_a_pending_candidate(self):
        item = self.submit(1)
        self.assertTrue(item["review_id"].startswith("rv-"))
        self.assertIsNone(item["label"])
        self.assertEqual(self.q.stats()["pending"], 1)

    def test_candidate_keeps_the_scored_feature_row(self):
        self.submit(3, )
        rid = self.q.pending()[0]["review_id"]
        self.assertEqual(self.q.x_row(rid), [3.0, 4.0, 5.0, 6.0])
        # the public pending list must not leak the training row
        self.assertNotIn("x_row", self.q.pending()[0])

    def test_pending_is_oldest_first(self):
        for i in range(3):
            self.submit(i)
        self.assertEqual([p["event_id"] for p in self.q.pending()],
                         ["e0", "e1", "e2"])

    def test_resolve_records_the_label_once(self):
        self.submit(1)
        rid = self.q.pending()[0]["review_id"]
        meta, err = self.q.resolve(rid, 1, reviewer="me", note="obvious")
        self.assertIsNone(err)
        self.assertEqual(meta["label"], 1)
        self.assertEqual(meta["reviewer"], "me")
        self.assertEqual(meta["note"], "obvious")
        again, err2 = self.q.resolve(rid, 0)
        self.assertIsNotNone(err2)
        self.assertIn("already labeled", err2)

    def test_resolved_candidates_leave_the_pending_list(self):
        self.submit(1)
        rid = self.q.pending()[0]["review_id"]
        self.q.resolve(rid, 0)
        self.assertEqual(self.q.pending(), [])
        self.assertEqual(self.q.stats()["pending"], 0)
        self.assertEqual(self.q.stats()["resolved"], 1)

    def test_invalid_labels_are_rejected(self):
        self.submit(1)
        rid = self.q.pending()[0]["review_id"]
        for bad in (2, -1, "x", None, 1.5):
            meta, err = self.q.resolve(rid, bad)
            self.assertIsNone(meta)
            self.assertIn("label", err)
        # still unresolved and labelable
        meta, err = self.q.resolve(rid, 1)
        self.assertIsNone(err)
        self.assertEqual(meta["label"], 1)

    def test_unknown_id_is_rejected(self):
        meta, err = self.q.resolve("rv-nope", 1)
        self.assertIsNone(meta)
        self.assertIn("unknown review_id", err)

    def test_source_filter(self):
        self.q.submit(source="chain", event_id="c", x_row=row(), account_id="a",
                      amount=1.0, prob=0.4, ts=time.time())
        self.q.submit(source="webhook", event_id="w", x_row=row(), account_id="a",
                      amount=1.0, prob=0.4, ts=time.time())
        self.assertEqual(len(self.q.pending(source="chain")), 1)
        self.assertEqual(len(self.q.pending(source="webhook")), 1)
        self.assertEqual(len(self.q.pending()), 2)


class TestBounds(unittest.TestCase):
    def test_queue_is_bounded_and_evicts_oldest(self):
        q = ReviewQueue(maxlen=3, expire_seconds=3600)
        for i in range(6):
            q.submit(source="chain", event_id="e%d" % i, x_row=row(), account_id="a",
                     amount=0.0, prob=0.4, ts=time.time())
        pending = [p["event_id"] for p in q.pending()]
        self.assertEqual(pending, ["e3", "e4", "e5"])
        self.assertEqual(q.stats()["queued"], 6)
        self.assertEqual(q.stats()["held"], 3)

    def test_expired_candidates_are_dropped(self):
        q = ReviewQueue(maxlen=10, expire_seconds=0)   # 0 disables expiry
        q.submit(source="chain", event_id="a", x_row=row(), account_id="a",
                 amount=0.0, prob=0.4, ts=time.time())
        self.assertEqual(len(q.pending()), 1)
        q2 = ReviewQueue(maxlen=10, expire_seconds=1)
        item = q2.submit(source="chain", event_id="a", x_row=row(), account_id="a",
                         amount=0.0, prob=0.4, ts=time.time())
        q2._items[item["review_id"]]["meta"]["queued_at"] = time.time() - 10
        self.assertEqual(q2.pending(), [])
        self.assertEqual(q2.stats()["expired"], 1)

    def test_reconfigure_trims(self):
        q = ReviewQueue(maxlen=10, expire_seconds=3600)
        for i in range(8):
            q.submit(source="chain", event_id="e%d" % i, x_row=row(), account_id="a",
                     amount=0.0, prob=0.4, ts=time.time())
        q.configure(maxlen=3)
        self.assertEqual(len(q.pending()), 3)


class TestLifecycle(unittest.TestCase):
    def test_reset_clears_everything(self):
        q = ReviewQueue(maxlen=10, expire_seconds=3600)
        q.submit(source="chain", event_id="e", x_row=row(), account_id="a",
                 amount=0.0, prob=0.4, ts=time.time())
        rid = q.pending()[0]["review_id"]
        q.resolve(rid, 1)
        q.reset()
        s = q.stats()
        self.assertEqual(s["pending"], 0)
        self.assertEqual(s["queued"], 0)
        self.assertEqual(s["resolved"], 0)
        self.assertEqual(s["by_label"], {0: 0, 1: 0})

    def test_stats_shape(self):
        s = ReviewQueue().stats()
        for key in ("pending", "held", "queued", "resolved", "expired",
                    "rejected", "by_label", "queue_max", "expire_seconds"):
            self.assertIn(key, s)


if __name__ == "__main__":
    unittest.main()
