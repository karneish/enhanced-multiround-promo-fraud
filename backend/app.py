"""Flask API + SSE feed for the real-time adaptive 3-model detector.

Endpoints
    GET  /health             liveness probe
    GET  /api/schema         configuration + data availability + event schema
    POST /api/live/start     {live_mode?, events_per_sec?, ...} start a session
    POST /api/live/stop
    GET  /api/live/status    running state + stats
    GET  /api/graph          account/counterparty graph snapshot
    POST /api/events/ingest  webhook: {account_id, amount?, counterparties?, label?, native?}
    GET  /api/review/queue   pending human/LLM review candidates
    POST /api/review         {review_id, label, reviewer?, note?} -> trains the source
    POST /api/review/batch   {items: [{review_id, label}], reviewer?}
    GET  /api/stream         Server-Sent Events feed (dashboard)
    GET  /api/               built React dashboard (frontend/dist), else hint JSON
"""
import json
import os
import time

from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS

from backend.config import load_config
from backend.events import event_schema_doc
from backend.livesim import LiveSimulation
from backend.sources.replay import available_ieee_files

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIST = os.path.join(ROOT, "frontend", "dist")

_sim_lock = None
_sim = None


class FeedCursor:
    """Per-SSE-connection cursor over a bounded deque feed."""

    def __init__(self):
        self._last = None

    def new_records(self, records, max_per_tick=40):
        """Records the client hasn't seen yet, bounded per tick.

        Identity comparison (not position) survives the ring buffer rolling over.
        On a cold start with a big backlog we advance a little each tick so a
        fresh client drains history gradually instead of all at once.
        """
        n = len(records)
        if not n:
            return []
        start = 0
        if self._last is not None:
            for i in range(n - 1, -1, -1):
                if records[i] is self._last:
                    start = i + 1
                    break
        cand = records[start:]
        if len(cand) > max_per_tick:
            self._last = cand[max_per_tick - 1]
            return cand[:max_per_tick]
        self._last = records[-1]
        return cand


def get_sim(config):
    global _sim
    if _sim is None:
        _sim = LiveSimulation(config)
    return _sim


def create_app(config=None) -> Flask:
    global _sim, _sim_lock
    cfg = config or load_config()

    app = Flask(__name__, static_folder=None)
    CORS(app)

    if _sim is None:
        _sim = LiveSimulation(cfg)
    sim = _sim

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "uptime": round(time.time() - app.config.get("started", time.time()), 2)})

    @app.get("/api/schema")
    def schema():
        return jsonify({
            "config": cfg,
            "event_schema": event_schema_doc(),
            "data": {
                "ieee_files": {k: bool(v) for k, v in available_ieee_files(cfg.get("ieee", {}).get("data_dir") or "data").items()},
                "sample_fallback": bool(cfg.get("ieee", {}).get("sample_fallback", True)),
                "scam_list": bool(os.path.isfile(cfg.get("chain", {}).get("scam_list_path", ""))),
            },
        })

    @app.post("/api/live/start")
    def live_start():
        body = request.get_json(silent=True) or {}
        sim.start(body)
        return jsonify(sim.status())

    @app.post("/api/live/stop")
    def live_stop():
        sim.stop()
        return jsonify(sim.status())

    @app.get("/api/live/status")
    def live_status():
        return jsonify(sim.status())

    @app.get("/api/graph")
    def graph():
        data = sim.graph_data()
        data["active"] = sim.running
        return jsonify(data)

    @app.post("/api/events/ingest")
    def ingest():
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify({"ok": False, "error": "expected JSON body"}), 400
        try:
            result = sim.ingest(payload)
            return jsonify({"ok": True, **result})
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception as exc:  # pipeline errors are surfaced to the client
            return jsonify({"ok": False, "error": str(exc)}), 500

    @app.get("/api/review/queue")
    def review_queue():
        """Pending review candidates, oldest first.

        This is the human/LLM labeling work list.  A reviewer (a person in the
        dashboard, or a script driving an LLM) reads each candidate and POSTs the
        ground-truth label back; nothing else in the pipeline can invent one.
        """
        limit = request.args.get("limit", default=50, type=int)
        source = request.args.get("source") or None
        limit = max(1, min(limit, 200))
        return jsonify({
            "ok": True,
            "running": sim.running,
            "pending": sim.review_pending(limit=limit, source=source),
            "stats": sim.review_stats(),
            "budget": sim.review_budget(),
        })

    @app.post("/api/review")
    def review_submit():
        """Attach one ground-truth label: {review_id, label, reviewer?, note?}."""
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"ok": False, "error": "expected JSON body"}), 400
        review_id = body.get("review_id")
        if not review_id:
            return jsonify({"ok": False, "error": "review_id is required"}), 400
        if "label" not in body:
            return jsonify({"ok": False, "error": "label is required (0 or 1)"}), 400
        try:
            out = sim.submit_review(review_id, body.get("label"),
                                    body.get("reviewer"), body.get("note"))
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500
        return jsonify({"ok": True, **out})

    @app.post("/api/review/batch")
    def review_batch():
        """Label many candidates at once: {items: [{review_id, label, note?}],
        reviewer?}.  Partial success is normal - each item reports its own
        outcome so one bad row cannot discard a reviewer's whole pass."""
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"ok": False, "error": "expected JSON body"}), 400
        items = body.get("items")
        if items is None and isinstance(body.get("labels"), list):
            # convenience: [{"review_id":..,"label":..}] passed as `labels`
            items = body["labels"]
        if not isinstance(items, list):
            return jsonify({"ok": False,
                            "error": "items must be a list of "
                                     "{review_id, label} objects"}), 400
        if len(items) > 500:
            return jsonify({"ok": False, "error": "at most 500 labels per call"}), 400
        out = sim.submit_review_batch(items, body.get("reviewer"))
        # `out["ok"]` is the per-item list of accepted labels; the response-level
        # `ok` flag has to win, so the list is published under `labeled`
        labeled = out.pop("ok")
        return jsonify({**out, "ok": not out["errors"], "labeled": labeled,
                        "stats": sim.review_stats()})

    @app.get("/api/stream")
    def stream():
        """SSE feed. Per-connection cursor: each client receives every record
        exactly once (no dupes when the endpoint is polled every 0.5s)."""
        def generate():
            cursor = FeedCursor()
            last_emit = 0.0
            while True:
                records = sim.feed.tail(1000)
                for rec in cursor.new_records(records):
                    yield f"event: message\ndata: {json.dumps(rec)}\n\n"
                status = sim.status()
                yield f"event: status\ndata: {json.dumps(status)}\n\n"
                now = time.time()
                if now - last_emit > 5.0:
                    yield ": keepalive\n\n"
                    last_emit = now
                time.sleep(0.5)

        return Response(generate(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    @app.get("/")
    def index():
        index_path = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.isfile(index_path):
            return Response(open(index_path, "rb").read(), mimetype="text/html")
        return jsonify({
            "message": "backend is running",
            "hint": "build the React dashboard (cd frontend && npm install && npm run build), or use /api/* directly",  # noqa: E501
        })

    @app.get("/assets/<path:asset>")
    def assets(asset):
        # send_from_directory infers a correct MIME type (text/javascript,
        # text/css, ...). Serving JS/CSS as application/octet-stream makes
        # browsers refuse to execute the module bundle -> blank dashboard.
        path = os.path.join("assets", asset)
        full = os.path.join(FRONTEND_DIST, path)
        if not os.path.isfile(full):
            return jsonify({"error": "not found"}), 404
        return send_file(full)

    app.config["started"] = time.time()
    return app


def run_server(host="127.0.0.1", port=5051):
    app = create_app()
    app.run(host=host, port=port, threaded=True, use_reloader=False)