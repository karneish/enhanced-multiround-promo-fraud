"""Tests for the HTTP surface: review API, ingest, lifecycle, SSE contract.

`backend.app` keeps the simulation in a module global, so every test resets it
to make sure a session from an earlier test cannot leak into the next one.
"""
import json
import unittest

import backend.app as appmod
from backend.app import create_app


def config(**over):
    cfg = {
        "live_mode": "concurrent",
        "stream_slice": 0,
        "events_per_sec": 1000,
        "score_batch_max": 32,
        "score_batch_window_ms": 0,
        "retain_event_records": 100,
        "graph_node_cap": 100,
        "budget_pos": 3, "budget_neg": 30,
        "decision_thresholds": {"review": 0.30, "block": 0.70},
        "min_fit_labels": 40, "retrain_interval_events": 50,
        "retrain_interval_seconds": 5, "buffer_max_labels": 500,
        "xgb_rounds": 10, "xgb_extra_rounds": 5, "hist_max_iter": 10,
        "hist_extra_iter": 5, "et_n_estimators": 10,
        "ieee": {"data_dir": "data", "sample_fallback": False},
        "chain": {"transport": "rpc-poll", "poll_interval_seconds": 3600},
    }
    cfg.update(over)
    return cfg


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        appmod._sim = None
        self.app = create_app(config())
        self.client = self.app.test_client()
        self.sim = appmod._sim
        self.client.post("/api/live/start", json={})

    def tearDown(self):
        try:
            self.client.post("/api/live/stop")
        finally:
            appmod._sim = None

    def feed(self, n, **payload):
        out = []
        for i in range(n):
            body = {"account_id": "a-%d" % (i % 5), "amount": 5.0 + i * 7}
            body.update(payload)
            out.append(self.client.post("/api/events/ingest", json=body))
        return out


class TestBasics(ApiTestCase):
    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])

    def test_schema_reports_data_and_scam_list(self):
        data = self.client.get("/api/schema").get_json()
        self.assertIn("event_schema", data)
        self.assertIn("ieee_files", data["data"])
        self.assertIn("config", data)

    def test_start_stop_status(self):
        self.assertTrue(self.client.get("/api/live/status").get_json()["running"])
        self.assertFalse(self.client.post("/api/live/stop").get_json()["running"])
        self.assertTrue(self.client.post("/api/live/start", json={})
                        .get_json()["running"])

    def test_graph_endpoint(self):
        self.feed(4, counterparties=["cp"])
        data = self.client.get("/api/graph").get_json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("active", data)


class TestIngest(ApiTestCase):
    def test_ingest_returns_the_decision(self):
        r = self.client.post("/api/events/ingest",
                             json={"account_id": "a", "amount": 10.0})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertTrue(body["ok"])
        self.assertIn(body["decision"], ("allow", "review", "block"))
        self.assertIn("prob", body)

    def test_labeled_ingest_blocks_fraud(self):
        r = self.client.post("/api/events/ingest",
                             json={"account_id": "f", "amount": 10.0, "label": 1})
        self.assertEqual(r.get_json()["decision"], "block")

    def test_missing_body_is_400(self):
        r = self.client.post("/api/events/ingest", data="not json",
                             content_type="application/json")
        self.assertEqual(r.status_code, 400)
        self.assertFalse(r.get_json()["ok"])

    def test_invalid_event_is_400(self):
        r = self.client.post("/api/events/ingest", json={"amount": 5.0})
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_bad_label_is_400(self):
        r = self.client.post("/api/events/ingest",
                             json={"account_id": "a", "amount": 1.0, "label": 7})
        self.assertEqual(r.status_code, 400)
        self.assertIn("label", r.get_json()["error"])

    def test_supplied_label_hint_never_reaches_the_model(self):
        """A webhook label must not become a feature: `label_hint` is oracle-only,
        so a model trained on it would be reading the answer off the input."""
        from backend.features import ORACLE_COLUMNS, FeatureBuilder
        self.assertIn("label_hint", ORACLE_COLUMNS)
        r = self.client.post("/api/events/ingest", json={
            "account_id": "leaky", "amount": 1.0, "label": 1})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["decision"], "block")
        vec = FeatureBuilder().build({
            "source": "webhook", "ts": 1.0, "account_id": "leaky",
            "counterparties": [], "label": 1, "amount": 1.0,
            "nat": {"label_hint": 1.0, "txn_amt": 1.0, "n_attrs": 0.0}})
        self.assertNotIn("label_hint", vec)
        self.assertTrue(all(col not in ORACLE_COLUMNS for col in vec))


class TestReviewApi(ApiTestCase):
    def queue(self, **kw):
        r = self.client.get("/api/review/queue", query_string=kw)
        self.assertEqual(r.status_code, 200)
        return r.get_json()

    def test_queue_lists_pending_candidates(self):
        self.feed(20)
        data = self.queue()
        self.assertTrue(data["ok"])
        self.assertTrue(data["pending"])
        for item in data["pending"]:
            for key in ("review_id", "source", "account_id", "prob"):
                self.assertIn(key, item)
            self.assertNotIn("x_row", item)
        self.assertIn("stats", data)
        self.assertIn("budget", data)

    def test_queue_limit_is_clamped(self):
        self.feed(30)
        self.assertLessEqual(len(self.queue(limit=9999)["pending"]), 200)
        self.assertGreaterEqual(len(self.queue(limit=1)["pending"]), 1)

    def test_queue_source_filter(self):
        self.feed(20)
        data = self.queue(source="webhook")
        self.assertTrue(data["pending"])
        self.assertTrue(all(p["source"] == "webhook" for p in data["pending"]))

    def test_label_one_candidate(self):
        self.feed(20)
        item = self.queue()["pending"][0]
        r = self.client.post("/api/review", json={
            "review_id": item["review_id"], "label": 1,
            "reviewer": "tester", "note": "n"})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["review"]["label"], 1)
        self.assertEqual(body["review"]["reviewer"], "tester")
        self.assertEqual(body["ensemble"].get("n_labeled"), 1)
        self.assertEqual(body["ensemble"].get("source"), "webhook")
        self.assertNotIn(item["review_id"],
                         [p["review_id"] for p in self.queue()["pending"]])

    def test_double_label_is_400(self):
        self.feed(20)
        rid = self.queue()["pending"][0]["review_id"]
        self.client.post("/api/review", json={"review_id": rid, "label": 1})
        r = self.client.post("/api/review", json={"review_id": rid, "label": 0})
        self.assertEqual(r.status_code, 400)
        self.assertIn("already labeled", r.get_json()["error"])

    def test_unknown_id_is_400(self):
        r = self.client.post("/api/review",
                             json={"review_id": "rv-nope", "label": 1})
        self.assertEqual(r.status_code, 400)

    def test_missing_fields_are_400(self):
        self.assertEqual(self.client.post("/api/review", json={}).status_code, 400)
        self.assertEqual(self.client.post(
            "/api/review", json={"review_id": "x"}).status_code, 400)
        self.assertEqual(self.client.post(
            "/api/review", data="[]", content_type="application/json"
        ).status_code, 400)

    def test_fractional_label_is_400(self):
        self.feed(20)
        rid = self.queue()["pending"][0]["review_id"]
        r = self.client.post("/api/review",
                             json={"review_id": rid, "label": 0.5})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/review",
                             json={"review_id": rid, "label": True})
        self.assertEqual(r.status_code, 400)

    def test_batch_labels(self):
        self.feed(30)
        pending = self.queue(limit=5)["pending"]
        items = [{"review_id": p["review_id"], "label": i % 2}
                 for i, p in enumerate(pending)]
        r = self.client.post("/api/review/batch",
                             json={"items": items, "reviewer": "llm"})
        body = r.get_json()
        self.assertEqual(len(body["labeled"]), len(items))
        self.assertEqual(body["errors"], [])
        self.assertTrue(body["ok"])
        stats = self.sim.review_stats()
        self.assertEqual(stats["resolved"], len(items))
        self.assertEqual(stats["by_label"], {0: stats["by_label"][0],
                                             1: stats["by_label"][1]})

    def test_batch_reports_partial_failures(self):
        self.feed(30)
        pending = self.queue(limit=2)["pending"]
        items = [{"review_id": pending[0]["review_id"], "label": 1},
                 {"review_id": "rv-nope", "label": 1},
                 {"label": 1}]
        body = self.client.post("/api/review/batch", json={"items": items}).get_json()
        self.assertEqual(len(body["labeled"]), 1)
        self.assertEqual(len(body["errors"]), 2)
        self.assertFalse(body["ok"], "partial success must not claim full success")

    def test_batch_alias_and_validation(self):
        self.feed(20)
        pending = self.queue(limit=1)["pending"]
        r = self.client.post("/api/review/batch", json={
            "labels": [{"review_id": pending[0]["review_id"], "label": 0}]})
        self.assertEqual(len(r.get_json()["labeled"]), 1)
        self.assertEqual(self.client.post("/api/review/batch",
                                          json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/review/batch",
                                          json={"items": "x"}).status_code, 400)
        self.assertEqual(self.client.post("/api/review/batch", json={
            "items": [{"review_id": "x", "label": 1}] * 501}).status_code, 400)

    def test_labels_train_the_ensemble(self):
        self.feed(30)
        pending = self.queue(limit=10)["pending"]
        self.client.post("/api/review/batch", json={
            "items": [{"review_id": p["review_id"], "label": 1} for p in pending]})
        ens = self.sim.ensembles["webhook"]
        self.assertEqual(ens.n_labeled, len(pending))
        self.assertEqual(ens.snapshot()["buffered"], len(pending))


class TestStream(ApiTestCase):
    def test_sse_headers_and_shape(self):
        r = self.client.get("/api/stream", buffered=False)
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/event-stream", r.headers["Content-Type"])
        self.assertEqual(r.headers.get("X-Accel-Buffering"), "no")
        chunk = next(r.response)
        text = chunk.decode() if isinstance(chunk, bytes) else chunk
        self.assertIn("event: status", text)
        payload = text.split("data: ", 1)[1].split("\n", 1)[0]
        self.assertIn("running", json.loads(payload))
        r.response.close()

    def test_sse_emits_each_record_once(self):
        self.feed(3)
        cursor = appmod.FeedCursor()
        records = self.sim.feed.tail(1000)
        first = cursor.new_records(records)
        self.assertTrue(first)
        self.assertEqual(cursor.new_records(records), [],
                         "a second pass must not repeat records")


if __name__ == "__main__":
    unittest.main()
