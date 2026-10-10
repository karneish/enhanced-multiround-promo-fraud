"""End-to-end smoke test over real HTTP: start -> unlabeled queue -> label -> refit.

Complements tests/test_app.py (which drives the Flask test client) by
exercising a real socket, the real WSGI thread pool and the real SSE stream.
Run with:  .venv\Scripts\python.exe scripts\smoke_review_loop.py
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app import create_app  # noqa: E402
from backend.config import load_config  # noqa: E402
import backend.app as appmod  # noqa: E402

HOST = "127.0.0.1"
PORT = 5099
BASE = f"http://{HOST}:{PORT}"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as r:
        return json.loads(r.read().decode())


def post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def read_sse(seconds=4.0):
    """Collect SSE frames for a while; returns (frames, raw_text).

    Frames are (event_type, payload) so message and status records stay
    distinguishable -- the stream interleaves both plus keepalive comments.
    """
    frames, raw = [], []
    req = urllib.request.Request(BASE + "/api/stream")
    stop = threading.Event()

    def body():
        with urllib.request.urlopen(req, timeout=seconds + 10) as r:
            ctype = r.headers.get("Content-Type", "")
            check("sse content-type", "text/event-stream" in ctype, ctype)
            ev = None
            deadline = time.time() + seconds
            while time.time() < deadline and not stop.is_set():
                line = r.readline()
                if not line:
                    break
                text = line.decode(errors="replace")
                raw.append(text)
                if text.startswith("event: "):
                    ev = text[7:].strip()
                elif text.startswith("data: "):
                    try:
                        frames.append((ev, json.loads(text[6:].strip())))
                    except json.JSONDecodeError:
                        pass
                    ev = None

    t = threading.Thread(target=body, daemon=True)
    t.start()
    t.join(seconds + 5)
    stop.set()
    return frames, "".join(raw)


def main():
    cfg = load_config()
    cfg["port"] = PORT
    cfg["live_mode"] = "concurrent"
    cfg["stream_slice"] = 3000        # small, but a real CSV read
    cfg["events_per_sec"] = 400
    cfg["min_fit_labels"] = 30
    cfg["retrain_interval_events"] = 15
    cfg["retrain_interval_seconds"] = 2
    cfg["budget_pos"] = 6
    cfg["budget_neg"] = 40
    cfg["review"] = {"queue_max": 50, "expire_seconds": 600}
    cfg["chain"]["poll_interval_seconds"] = 3600     # do not hammer RPC

    appmod._sim = None
    app = create_app(cfg)
    threading.Thread(target=lambda: app.run(host=HOST, port=PORT, threaded=True,
                                            use_reloader=False), daemon=True).start()
    for _ in range(80):
        try:
            get("/health")
            break
        except Exception:
            time.sleep(0.25)
    else:
        print("server never came up")
        return 1

    print("1. health / schema / dashboard")
    check("health ok", get("/health")["ok"] is True)
    schema = get("/api/schema")
    check("schema reports the real IEEE-CIS CSV",
          schema["data"]["ieee_files"]["train_transaction"] is True,
          str(schema["data"]["ieee_files"]))
    idx = urllib.request.urlopen(BASE + "/", timeout=20)
    check("dashboard index served", idx.status == 200 and b"<" in idx.read()[:200])

    print("2. SSE delivers records before/while scoring")
    stop_flag = threading.Event()
    sse_result = {}

    def sse_thread():
        sse_result["frames"], sse_result["raw"] = read_sse(seconds=6.0)

    th = threading.Thread(target=sse_thread, daemon=True)
    th.start()
    time.sleep(0.4)
    status, started = post("/api/live/start", {"live_mode": "concurrent"})
    check("start returns running", started.get("running") is True)
    th.join(20)

    frames = sse_result.get("frames", [])
    raw = sse_result.get("raw", "")
    msgs = [p for t, p in frames if t == "message" and isinstance(p, dict)]
    stats = [p for t, p in frames if t == "status" and isinstance(p, dict)]
    check("sse produced status frames", len(stats) > 0,
          "%d status, %d message" % (len(stats), len(msgs)))
    check("sse frames are json", all("running" in f or "throughput" in f for f in stats))
    check("sse sent a keepalive", ": keepalive" in raw)
    check("sse carried message records", len(msgs) > 0, "%d messages" % len(msgs))
    ids = [m.get("event_id") for m in msgs if m.get("event_id")]
    check("no duplicate event_ids in sse", len(ids) == len(set(ids)),
          "%d ids, %d unique" % (len(ids), len(set(ids))))

    print("3. replay stream is scoring")
    time.sleep(6)
    st = get("/api/live/status")
    counters = st.get("counters", {})
    scored = sum(counters.values()) if isinstance(counters, dict) else 0
    check("events scored", scored > 0, f"counters={counters}")
    check("throughput reported", st.get("throughput", {}).get("rows_per_sec", 0) > 0,
          f"{st.get('throughput', {}).get('rows_per_sec')} rows/s")
    check("replay source loaded without error",
          not st.get("source_states", {}).get("replay", {}).get("error"),
          str(st.get("source_states", {}).get("replay")))
    check("no thread/process errors", not st.get("errors"), str(st.get("errors")))

    print("4. the review loop (unlabeled webhook events -> label -> model learns)")
    for i in range(60):
        post("/api/events/ingest",
             {"account_id": f"smoke-{i % 7}", "amount": 5.0 + (i % 11) * 9.0,
              "counterparties": [f"cp-{i % 3}"]})
    q = get("/api/review/queue?limit=50")
    pending = q.get("pending", [])
    by_source = {}
    for p in pending:
        by_source.setdefault(p.get("source"), []).append(p)
    mix = {k: len(v) for k, v in by_source.items()}
    check("unlabeled events were queued for review", len(pending) > 0,
          "pending=%d budget=%s sources=%s" % (len(pending), q.get("budget"), mix))
    check("queued items carry no training row",
          all("x_row" not in p for p in pending))
    check("queued ids are unique", len({p["review_id"] for p in pending}) == len(pending))

    wh = get("/api/review/queue?limit=50&source=webhook").get("pending", [])
    check("queue filters by source", len(wh) > 0 and all(p.get("source") == "webhook" for p in wh),
          "webhook=%d of %d" % (len(wh), len(pending)))
    pending = wh

    labels = [{"review_id": p["review_id"], "label": 1 if i % 4 == 0 else 0}
              for i, p in enumerate(pending)]
    code, res = post("/api/review/batch", {"items": labels, "reviewer": "smoke"})
    check("batch accepted every label", len(res.get("labeled", [])) == len(labels),
          f"{len(res.get('labeled', []))}/{len(labels)} ok={res.get('ok')}")
    check("batch flag is a bool, not a list", isinstance(res.get("ok"), bool),
          repr(res.get("ok"))[:40])

    after = get("/api/live/status")
    ens = after.get("ensembles", {}).get("webhook", {})
    check("labels reached the webhook ensemble",
          ens.get("n_labeled", 0) >= len(labels),
          "n_labeled=%s labels=%d" % (ens.get("n_labeled"), len(labels)))
    q2 = get("/api/review/queue?limit=50")
    still = {p["review_id"] for p in q2.get("pending", [])}
    check("labeled candidates left the queue",
          not (still & {l["review_id"] for l in labels}))

    print("5. validation is enforced over HTTP")
    code, _ = post("/api/review", {"review_id": labels[0]["review_id"], "label": 1})
    check("double label -> 400", code == 400, f"code={code}")
    code, _ = post("/api/review", {"review_id": "rv-nope", "label": 1})
    check("unknown id -> 400", code == 400, f"code={code}")
    code, _ = post("/api/events/ingest", {"account_id": "x", "amount": 1.0, "label": 7})
    check("bad ingest label -> 400", code == 400, f"code={code}")

    # the batch triage bar in the UI renders labeled[] + errors[], so a mixed
    # batch must partially succeed rather than discard the whole pass.
    # ingest more first: the previous batch drained the webhook candidates.
    for i in range(40, 100):
        post("/api/events/ingest",
             {"account_id": f"smoke2-{i % 5}", "amount": 3.0 + (i % 13) * 7.0,
              "counterparties": [f"cp-{i % 4}"]})
    fresh = []
    deadline = time.time() + 30
    while time.time() < deadline:
        fresh = get("/api/review/queue?limit=10&source=webhook").get("pending", [])
        if len(fresh) >= 2:
            break
        time.sleep(1)
    check("fresh webhook candidates available for the batch bar", len(fresh) >= 2,
          "%d pending" % len(fresh))
    if len(fresh) >= 2:
        mixed = [{"review_id": fresh[0]["review_id"], "label": 1},
                 {"review_id": "rv-does-not-exist", "label": 0},
                 {"review_id": fresh[1]["review_id"], "label": 0.5}]
        code, res = post("/api/review/batch", {"items": mixed, "reviewer": "smoke"})
        check("mixed batch keeps the good rows", len(res.get("labeled", [])) == 1,
              "labeled=%d" % len(res.get("labeled", [])))
        check("mixed batch reports per-item errors", len(res.get("errors", [])) == 2,
              str(res.get("errors"))[:120])
        check("mixed batch ok flag is False", res.get("ok") is False, repr(res.get("ok")))
        check("error entries carry review_id + error",
              all("review_id" in e and "error" in e for e in res.get("errors", [])))
        check("batch echoes stats for the budget bar", isinstance(res.get("stats"), dict))

    print("6. refit happened in the background")
    deadline = time.time() + 90
    fitted = False
    while time.time() < deadline:
        st = get("/api/live/status")
        if any(e.get("fitted") for e in st.get("ensembles", {}).values()):
            fitted = True
            break
        time.sleep(2)
    check("an ensemble refitted", fitted)
    st = get("/api/live/status")
    for src, e in (st.get("ensembles") or {}).items():
        if e.get("fitted"):
            w = e.get("weights") or {}
            print(f"      {src}: weights={ {k: round(v, 3) for k, v in w.items()} } "
                  f"f1={ {k: round((m or {}).get('f1', 0), 3) for k, m in (e.get('models') or {}).items()} } "
                  f"holdout={e.get('holdout_mode')}")
            check(f"{src} weights sum to 1",
                  abs(sum(w.values()) - 1.0) < 1e-6, str(w))
            check(f"{src} exposes per-model F1",
                  all((m or {}).get("f1") is not None
                      for m in (e.get("models") or {}).values()))

    print("7. graph + stop")
    g = get("/api/graph")
    cap = int(cfg.get("graph_node_cap", 2000))
    check("graph has nodes", len(g.get("nodes", [])) > 0, "%d nodes" % len(g.get("nodes", [])))
    check("graph respects the node cap", len(g.get("nodes", [])) <= cap,
          "%d <= %d" % (len(g.get("nodes", [])), cap))
    st = post("/api/live/stop", {})[1]
    check("stopped", st.get("running") is False)
    time.sleep(2)
    st = get("/api/live/status")
    check("status stops reporting a running session", st.get("running") is False)
    check("no threads left", appmod._sim.threads_alive() == 0,
          str(appmod._sim.threads_alive()))
    check("no leak flagged", not st.get("errors", {}).get("thread_leak"),
          str(st.get("errors")))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
        return 1
    print("all smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
