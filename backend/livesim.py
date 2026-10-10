"""LiveSimulation: the streaming orchestrator.

Producers (replay / chain / webhook) generate canonical events -> a single
consumer thread scores them in micro-batches with the per-source 3-model
ensemble, applies Allow/Review/Block decisions, spends the human-review budget
to queue candidates for labeling, warm-starts the models on a background thread
on cadence, and publishes everything to the SSE feed.

Two design notes that matter for real-time behaviour:

* **Scoring is batched.** Per-row ExtraTrees prediction costs ~135 ms of fixed
  overhead at 200 trees; a 256-row batch costs the same ~135 ms. The consumer
  therefore drains the queue into micro-batches (`score_batch_max`,
  `score_batch_window_ms`) and issues one `predict_proba_batch` per model.

* **Fitting is off-thread.** A first fit takes seconds and a warm refits takes
  seconds; running that on the consumer would stall the whole stream. Each
  ensemble owns a single-worker executor and swaps fitted models in atomically,
  so `predict` never blocks on `fit`.
"""
import collections
import queue
import threading
import time

import numpy as np

from backend.detector import ThreeModelEnsemble
from backend.features import FeatureBuilder, columns_for, decision_from_prob
from backend.labels import ScamAddressMatcher
from backend.review import ReviewQueue

DECISIONS = ("allow", "review", "block")
QUEUE_MAXSIZE = 20000


class StreamFeed:
    """Bounded, thread-safe ring buffer of JSON payloads for the SSE feed."""

    def __init__(self, maxlen=200):
        self._q = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, payload):
        with self._lock:
            self._q.append(payload)

    def tail(self, n=60):
        with self._lock:
            items = list(self._q)
        return items[-n:]

    def clear(self):
        with self._lock:
            self._q.clear()


class _AmountMedian:
    """Small per-source ring of recent amounts, used only to bucket a review
    candidate into the pos/neg budget.  Kept per source because IEEE amounts
    (dollars) and Ethereum amounts (ETH) are not comparable."""

    def __init__(self, size=200):
        self._vals = collections.deque(maxlen=size)
        self._sorted = []

    def add(self, amount):
        if amount is None:
            return
        self._vals.append(float(amount))

    def median(self):
        if not self._vals:
            return None
        self._sorted = sorted(self._vals)
        n = len(self._sorted)
        mid = n // 2
        if n % 2:
            return self._sorted[mid]
        return 0.5 * (self._sorted[mid - 1] + self._sorted[mid])

    def reset(self):
        self._vals.clear()
        self._sorted = []


class LiveSimulation:
    def __init__(self, config):
        self.config = dict(config)
        self.mode = self.config.get("live_mode", "concurrent")
        self.feed = StreamFeed(self.config.get("retain_event_records", 200))

        self._queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._running = False
        self._generation = 0
        self._start_lock = threading.Lock()
        self._ingest_lock = threading.Lock()

        self.builder = FeatureBuilder()
        self.matcher = ScamAddressMatcher(self.config.get("chain", {}).get("scam_list_path"))
        self._reviews = ReviewQueue()
        self.ensembles = {}
        self.sources = {}
        self._source_states = {}
        self._threads = []
        self._medians = {}
        self._batch_max = 0
        self._last_refit_count = {}

        self._counters = collections.Counter()
        self._errors = collections.Counter()
        self._decisions = {"allow": 0, "review": 0, "block": 0}
        self._review_budget = self._fresh_budget()
        self._review_spent = 0
        self._replay_finished = False
        self._replay_finish_at = None
        self._started_at = None
        self._cur_event_id = None
        self._scored = 0
        self._score_seconds = 0.0

        # Graph of everything seen: account/counterparty nodes + sharing edges.
        self._graph = {"nodes": {}, "edges": {}}
        self._graph_cap = int(self.config.get("graph_node_cap", 2000))
        self.graph_revision = 0

        self._configure()

    # ------------------------------------------------------------------ setup
    def _configure(self):
        review_cfg = self.config.get("review") or {}
        self._reviews.configure(maxlen=review_cfg.get("queue_max", 200),
                                expire_seconds=review_cfg.get("expire_seconds", 3600))
        self._batch_max = max(1, int(self.config.get("score_batch_max", 256)))
        self._batch_window = max(0.0,
                                 float(self.config.get("score_batch_window_ms", 25)) / 1000.0)
        self.feed = StreamFeed(self.config.get("retain_event_records", 200))

    def _fresh_budget(self):
        return {"pos": int(self.config.get("budget_pos", 10)),
                "neg": int(self.config.get("budget_neg", 200))}

    def _apply_overrides(self, overrides):
        if not overrides:
            return
        merged = dict(self.config)
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        self.config = merged
        self.mode = merged.get("live_mode", "concurrent")
        self._configure()

    # ------------------------------------------------------------------ state
    @property
    def running(self):
        return self._running

    @property
    def generation(self):
        return self._generation

    def _source_state(self, source):
        return self._source_states.setdefault(source, {
            "events": 0, "labeled": 0, "reviewed": 0,
            "finished": False, "error": None, "last_ts": None,
        })

    def _reset_source_state(self, source):
        self._source_states[source] = {
            "events": 0, "labeled": 0, "reviewed": 0,
            "finished": False, "error": None, "last_ts": None,
        }

    def _mirror_ieee_to_replay(self, source):
        """Keep the producer/UI-facing `replay` status dict in sync with the
        canonical `ieee` consumer counters (they are the same physical stream)."""
        if source != "ieee":
            return
        state = self._source_states.get("ieee") or {}
        repl = self._source_state("replay")
        repl["events"] = state.get("events", 0)
        repl["labeled"] = state.get("labeled", 0)
        repl["last_ts"] = state.get("last_ts")

    def _reset_session(self):
        """Clear everything a new session must not inherit.  Without this a
        second run starts with an already-spent review budget and last run's
        counters, so the dashboard's 'this session' numbers are cumulative."""
        for ens in self.ensembles.values():
            try:
                ens.shutdown()
            except Exception:
                pass
        self.ensembles = {}
        self.builder = FeatureBuilder()
        self._counters = collections.Counter()
        self._errors = collections.Counter()
        self._decisions = {"allow": 0, "review": 0, "block": 0}
        self._review_budget = self._fresh_budget()
        self._review_spent = 0
        self._replay_finished = False
        self._replay_finish_at = None
        self._cur_event_id = None
        self._scored = 0
        self._score_seconds = 0.0
        self._graph = {"nodes": {}, "edges": {}}
        self.graph_revision = 0
        self._medians = {}
        self._reviews.reset()
        self.feed.clear()
        for src in ("replay", "chain", "webhook", "ieee"):
            self._reset_source_state(src)

    # ------------------------------------------------------------------ start
    def start(self, overrides=None):
        with self._start_lock:
            if self._running:
                return self.status()
            self._apply_overrides(overrides)
            self._generation += 1
            gen = self._generation

            if self.mode not in ("replay-only", "chain-only", "handoff", "concurrent"):
                self.mode = "concurrent"
                self.config["live_mode"] = self.mode

            self._reset_session()
            self._stop.clear()
            self._queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
            self.sources = {}
            self._started_at = time.time()
            self._running = True

            needs_replay = self.mode in ("replay-only", "handoff", "concurrent")
            needs_chain = self.mode in ("chain-only", "concurrent")

            if needs_replay:
                from backend.sources.replay import IeeeReplaySource
                self.sources["replay"] = IeeeReplaySource(self.config)
            if needs_chain:
                from backend.sources.ethereum import EthereumSource
                # NOTE: EthereumSource.__init__ never probes the network - that
                # happens on the producer thread - so this stays instant.
                self.sources["chain"] = EthereumSource(self.config, self.matcher)

            if not self.sources:
                self._running = False
                self._generation += 1
                return self.status()

            consumer = threading.Thread(target=self._consumer_loop,
                                       name="livesim-consumer", daemon=True)
            consumer.start()
            self._threads = [consumer]

            if "replay" in self.sources:
                t = threading.Thread(target=self._run_replay, args=(gen,),
                                     name="producer-replay", daemon=True)
                t.start()
                self._threads.append(t)
            if "chain" in self.sources:
                t = threading.Thread(target=self._run_chain, args=(gen,),
                                     name="producer-chain", daemon=True)
                t.start()
                self._threads.append(t)
            if self.mode == "handoff":
                t = threading.Thread(target=self._run_handoff, args=(gen,),
                                     name="producer-handoff", daemon=True)
                t.start()
                self._threads.append(t)

            return self.status()

    # ------------------------------------------------------------------- stop
    def stop(self):
        with self._start_lock:
            if not self._running:
                return self.status()
            self._running = False
            self._generation += 1
            self._stop.set()
            alive = 0
            for t in self._threads:
                t.join(timeout=5)
                if t.is_alive():
                    alive += 1
            self._threads.clear()
            if alive:
                self._errors["thread_leak"] += 1
            return self.status()

    def threads_alive(self):
        return sum(1 for t in self._threads if t.is_alive())

    # ------------------------------------------------------------------ props
    def _stale(self, gen):
        """True once this producer's session has been superseded."""
        return self._stop.is_set() or gen != self._generation

    def _run_replay(self, gen):
        src = self.sources.get("replay")
        state = self._source_state("replay")
        try:
            result = src.load(cancel=lambda: self._stale(gen))
            if self._stale(gen):
                return
            if not result.get("loaded"):
                state["error"] = result.get("error", "replay not loaded")
                self.feed.push({"type": "source_error", "source": "replay",
                                "error": result.get("error"),
                                "hint": result.get("hint")})
                return
            state["sample"] = bool(result.get("sample"))
            self._source_state("ieee")["sample"] = bool(result.get("sample"))
            self.feed.push({"type": "source_status", "source": "replay",
                            "loaded": True, "rows": result.get("rows"),
                            "fraud": result.get("fraud"),
                            "path": result.get("path"),
                            "sample": bool(result.get("sample")),
                            "cancelled": bool(result.get("cancelled"))})
            for ev in src.stream(self._stop, cancel=lambda: self._stale(gen)):
                if self._stale(gen):
                    return
                if ev is None:
                    break
                self._queue.put(ev)
            if self._stale(gen):
                return
            state["finished"] = True
            self._replay_finished = True
            self._replay_finish_at = time.time()
            self.feed.push({"type": "event", "notice": "replay finished"})
        except Exception as exc:
            if self._stale(gen):
                return
            state["error"] = str(exc)
            self._errors["source_error"] += 1
            self.feed.push({"type": "source_error", "source": "replay", "error": str(exc)})

    def _run_handoff(self, gen):
        replay_state = self._source_state("replay")
        try:
            while not replay_state.get("finished") and not replay_state.get("error"):
                if self._stale(gen):
                    return
                if self._stop.wait(0.5):
                    return
            if self._stale(gen):
                return
            from backend.sources.ethereum import EthereumSource
            self.sources["chain"] = EthereumSource(self.config, self.matcher)
            self._run_chain(gen)
        except Exception as exc:
            if self._stale(gen):
                return
            self._source_state("chain")["error"] = str(exc)
            self._errors["source_error"] += 1

    def _run_chain(self, gen):
        src = self.sources.get("chain")
        if src is None:
            return
        state = self._source_state("chain")
        try:
            for ev in src.stream(self._stop):
                if self._stale(gen):
                    return
                if ev is None:
                    continue
                self._queue.put(ev)
        except Exception as exc:
            if self._stale(gen):
                return
            state["error"] = str(exc)
            state["finished"] = True
            self._errors["source_error"] += 1

    # --------------------------------------------------------------- consumer
    def _consumer_loop(self):
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            # opportunistically fill the batch so the per-model fixed cost is
            # amortised across many rows
            deadline = time.time() + self._batch_window
            while len(batch) < self._batch_max and time.time() < deadline:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            try:
                with self._ingest_lock:
                    self._process_batch(batch)
            except Exception as exc:
                self._errors["process_error"] += 1
                self.feed.push({"type": "process_error", "error": str(exc),
                                "event_ids": [e.get("event_id") for e in batch[:5]]})

    def ingest(self, payload):
        """Webhook entry point (called from the Flask request thread)."""
        from backend.sources.webhook import parse_payload
        ev = parse_payload(payload)
        with self._ingest_lock:
            results = self._process_batch([ev])
        return results[0]

    # -------------------------------------------------------------- processing
    def _ensemble(self, source, cols):
        ens = self.ensembles.get(source)
        if ens is None:
            ens = ThreeModelEnsemble(source, cols, self.config)
            self.ensembles[source] = ens
        return ens

    def _process_batch(self, batch):
        """Score a micro-batch of canonical events.

        Three passes: (1) build every feature vector from the profile state that
        precedes its own event, absorbing in order so the next event sees it;
        (2) one vectorised `predict_proba_batch` per source; (3) decisions,
        review-budget spending, graph, labels, feed records.
        """
        if not batch:
            return []
        started = time.time()

        prepared = []
        by_source = collections.defaultdict(list)
        for ev in batch:
            source = ev["source"]
            state = self._source_state(source)
            state["events"] += 1
            state["last_ts"] = ev.get("ts")
            self._counters[source] += 1
            self._cur_event_id = ev.get("event_id")
            self._mirror_ieee_to_replay(source)

            # ground truth: native labels always win; chain falls back to the
            # scam-address bridge; everything else stays null for the reviewer
            if ev.get("label") is None and source == "chain":
                ev["label"] = self.matcher.label_for(ev)

            vec = self.builder.build(ev)                 # profile state BEFORE absorb
            cols = columns_for(source)
            x_row = [vec[c] for c in cols]
            oracle = self.builder.oracle(ev)

            idx = len(prepared)
            prepared.append({
                "ev": ev, "source": source, "x_row": x_row, "oracle": oracle,
                "state": state,
            })
            by_source[source].append(idx)

            self.builder.absorb(ev)                      # now ingest into history
            self._amount_ring(source).add(ev.get("amount"))

        probs = {}
        fitted_by_source = {}
        for source, idxs in by_source.items():
            cols = columns_for(source)
            ens = self._ensemble(source, cols)
            fitted_by_source[source] = ens.fitted
            X = np.asarray([prepared[i]["x_row"] for i in idxs], dtype=float)
            try:
                scored = ens.predict_proba_batch(X)
            except Exception as exc:
                self._errors["score_error"] += 1
                self.feed.push({"type": "process_error", "error": str(exc),
                                "source": source})
                scored = [ens._cold_start_proba()] * len(idxs)
            for i, p in zip(idxs, scored):
                probs[i] = p

        thresholds = self.config.get("decision_thresholds",
                                     {"review": 0.30, "block": 0.70})
        records = []
        touched = set()
        for idx, item in enumerate(prepared):
            ev = item["ev"]
            source = item["source"]
            prob = probs.get(idx, 0.0)
            decision = decision_from_prob(prob, thresholds)
            cold_review = False
            if ev.get("label") == 1:
                decision = "block"      # known fraud always blocks
            elif ev.get("label") is None and not fitted_by_source.get(source, False):
                # An unlabeled stream with no fitted model can never reach the
                # review band on its own: the cold-start prior sits below
                # `review`, so the budget would never be spent and the ensemble
                # would stay frozen at cold start forever.  With no model there
                # is nothing to triage with, so the review budget *is* the
                # triage - queue until the budget runs out.
                decision = "review"
                cold_review = True

            self._decisions[decision] += 1
            reviewed = self._maybe_queue_review(ev, source, prob, decision,
                                                item["x_row"])

            self._record_graph(ev, prob, decision)

            if ev.get("label") is not None:
                ens = self.ensembles.get(source)
                if ens is not None:
                    ens.add_labeled(item["x_row"], ev["label"])
                    item["state"]["labeled"] += 1
                    self._mirror_ieee_to_replay(source)
                    touched.add(source)

            records.append({
                "type": "event",
                "event_id": ev.get("event_id"),
                "source": source,
                "ts": ev.get("ts"),
                "account_id": ev.get("account_id"),
                "amount": ev.get("amount"),
                "prob": round(prob, 4),
                "decision": decision,
                "label": ev.get("label"),
                "reviewed": reviewed,
                "cold_start_review": cold_review,
                "review_id": ev.get("review_id"),
                "known_fraud": bool(ev.get("label") == 1),
                "oracle": item["oracle"] or None,
            })

        for rec in records:
            self.feed.push(rec)

        for source in touched:
            ens = self.ensembles.get(source)
            if ens is not None:
                self._consider_refit(source, ens)

        self._scored += len(batch)
        self._score_seconds += time.time() - started
        return records

    def _maybe_queue_review(self, ev, source, prob, decision, x_row):
        """Spend review budget to queue an unlabeled Review decision.

        Spending the budget only *queues* the candidate: the label arrives via
        `POST /api/review` (dashboard click or an LLM script) and is what
        actually trains the model.  See backend/review.py.
        """
        if ev.get("label") is not None or decision != "review":
            return False

        amount = ev.get("amount") or 0.0
        median = self._amount_ring(source).median()
        key = "pos" if (median is not None and amount >= median) else "neg"
        if self._review_budget.get(key, 0) <= 0:
            return False

        self._review_budget[key] -= 1
        self._review_spent += 1
        candidate = self._reviews.submit(
            source=source,
            event_id=ev.get("event_id"),
            x_row=x_row,
            account_id=ev.get("account_id"),
            amount=amount,
            prob=prob,
            ts=ev.get("ts"),
            decided_prob=prob,
            extra={"counterparties": ev.get("counterparties") or [],
                   "oracle": self.builder.oracle(ev) or None},
        )
        ev["review_id"] = candidate["review_id"]
        self._source_state(source)["reviewed"] += 1
        return True

    def _amount_ring(self, source):
        ring = self._medians.get(source)
        if ring is None:
            ring = _AmountMedian()
            self._medians[source] = ring
        return ring

    # -------------------------------------------------------------- review API
    def review_pending(self, limit=50, source=None):
        return self._reviews.pending(limit=limit, source=source)

    def review_stats(self):
        return self._reviews.stats()

    def review_budget(self):
        return dict(self._review_budget)

    def submit_review(self, review_id, label, reviewer=None, note=None):
        """Attach a human/LLM ground-truth label and feed it to the ensemble."""
        meta, err = self._reviews.resolve(review_id, label, reviewer, note)
        if err:
            raise ValueError(err)
        x_row = self._reviews.x_row(review_id)
        source = meta["source"]
        cols = columns_for(source)
        refit = None
        with self._ingest_lock:
            ens = self._ensemble(source, cols)
            ens.add_labeled(x_row, meta["label"])
            state = self._source_state(source)
            state["labeled"] += 1
            state["reviewed"] = state.get("reviewed", 0) + 1
            self._mirror_ieee_to_replay(source)
            refit = self._consider_refit(source, ens)
            snapshot = ens.snapshot()
        self.feed.push({"type": "review_resolved", "source": source,
                        "review_id": review_id, "label": meta["label"],
                        "reviewer": meta["reviewer"],
                        "account_id": meta["account_id"],
                        "prob": meta["prob"]})
        return {"review": meta, "ensemble": snapshot, "refit": refit}

    def submit_review_batch(self, items, reviewer=None):
        out = {"ok": [], "errors": []}
        for it in items or []:
            if not isinstance(it, dict):
                out["errors"].append({"review_id": None, "error": "not an object"})
                continue
            rid = it.get("review_id")
            try:
                res = self.submit_review(rid, it.get("label"), reviewer,
                                         it.get("note"))
                out["ok"].append({"review_id": rid, "label": res["review"]["label"]})
            except ValueError as exc:
                out["errors"].append({"review_id": rid, "error": str(exc)})
            except Exception as exc:
                self._errors["review_error"] += 1
                out["errors"].append({"review_id": rid, "error": str(exc)})
        return out

    # -------------------------------------------------------------- graph view
    def _record_graph(self, ev, prob, decision):
        """Accumulate every account/counterparty node and every sharing edge
        (account <-> counterparty) in a bounded, query-ready graph."""
        acc = str(ev.get("account_id"))
        cps = ev.get("counterparties") or []
        nodes = self._graph["nodes"]
        edges = self._graph["edges"]

        def upsert(node_id, kind):
            node = nodes.get(node_id)
            if node is None:
                if len(nodes) >= self._graph_cap:
                    return None
                node = {
                    "id": node_id, "kind": kind,
                    "sources": set(), "n_events": 0,
                    "labels": {0: 0, 1: 0},
                    "last_decision": None, "last_prob": None,
                    "last_ts": None, "amount_total": 0.0,
                }
                nodes[node_id] = node
            return node

        acc_node = upsert(acc, "account")
        if acc_node is not None:
            acc_node["sources"].add(ev["source"])
            acc_node["n_events"] += 1
            acc_node["last_decision"] = decision
            acc_node["last_prob"] = prob
            acc_node["last_ts"] = ev.get("ts")
            acc_node["amount_total"] += float(ev.get("amount") or 0.0)
            if ev.get("label") in (0, 1):
                acc_node["labels"][ev["label"]] += 1

        for cp in cps:
            cp_s = str(cp)
            if cp_s == acc:
                continue
            cp_node = upsert(cp_s, "cp")
            if cp_node is None:
                continue
            cp_node["sources"].add(ev["source"])
            cp_node["n_events"] += 1
            if ev.get("label") in (0, 1):
                cp_node["labels"][ev["label"]] += 1

            if acc_node is not None:
                key = "|".join(sorted([acc, cp_s]))
                e = edges.get(key)
                if e is None:
                    if len(edges) >= 4 * self._graph_cap:
                        continue
                    e = {"a": acc, "b": cp_s, "weight": 0,
                         "last_ts": None, "sources": set()}
                    edges[key] = e
                e["weight"] += 1
                e["last_ts"] = ev.get("ts")
                e["sources"].add(ev["source"])

        self.graph_revision += 1

    def graph_data(self):
        """Serialized graph snapshot for GET /api/graph."""
        degree = collections.Counter()
        for e in self._graph["edges"].values():
            degree[e["a"]] += 1
            degree[e["b"]] += 1
        nodes = [
            {
                "id": n["id"],
                "kind": n["kind"],
                "sources": sorted(s for s in n["sources"] if s),
                "n_events": n["n_events"],
                "n_fraud": n["labels"][1],
                "n_known": n["labels"][0],
                "last_decision": n["last_decision"],
                "last_prob": n["last_prob"],
                "last_ts": n["last_ts"],
                "amount_total": n["amount_total"],
                "degree": degree.get(n["id"], 0),
            }
            for n in self._graph["nodes"].values()
        ]
        edges = [
            {
                "a": e["a"], "b": e["b"],
                "weight": e["weight"],
                "last_ts": e["last_ts"],
                "sources": sorted(s for s in e["sources"] if s),
            }
            for e in self._graph["edges"].values()
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "revision": self.graph_revision,
            "capped": len(self._graph["nodes"]) >= self._graph_cap,
        }

    # ------------------------------------------------------------------ refit
    def _consider_refit(self, source, ens):
        """Ask for a background warm-start refit.  Returns a small status dict
        so callers (the review API, refit error feed) can report it."""
        if not ens.fit_needed(time.time()):
            return None
        try:
            queued = ens.request_refit()
        except Exception as exc:
            self._errors["refit_error"] += 1
            self.feed.push({"type": "refit_error", "source": source, "error": str(exc)})
            return {"queued": False, "error": str(exc)}
        if not queued:
            return {"queued": False, "reason": "already in progress"}
        self.feed.push({"type": "refit", "source": source, "phase": "queued"})
        return {"queued": True, "source": source}

    def _poll_refits(self):
        """Called from the SSE loop so refit completions reach the dashboard even
        though the fit itself happens on a worker thread."""
        for source, ens in self.ensembles.items():
            snap = ens.snapshot()
            if snap.get("refit_in_progress"):
                continue
            if self._last_refit_count.get(source) != snap.get("refit_count"):
                self._last_refit_count[source] = snap.get("refit_count")
                self.feed.push({"type": "refit", "source": source, "phase": "done",
                                "models": snap.get("models"),
                                "weights": snap.get("weights"),
                                "holdout_mode": snap.get("holdout_mode"),
                                "seconds": snap.get("last_refit_seconds")})

    # ------------------------------------------------------------- status API
    def status(self):
        self._poll_refits()
        chain_state = self._source_state("chain")
        replay_state = self._source_state("replay")
        scored = self._scored
        return {
            "running": self._running,
            "mode": self.mode,
            "generation": self._generation,
            "started_at": self._started_at,
            "uptime_seconds": round(time.time() - self._started_at, 1) if self._started_at else 0,
            "counters": dict(self._counters),
            "decisions": dict(self._decisions),
            "errors": dict(self._errors),
            "review_budget": dict(self._review_budget),
            "review_spent": self._review_spent,
            "review": self._reviews.stats(),
            "replay_finished": self._replay_finished,
            "throughput": {
                "scored": scored,
                "score_seconds": round(self._score_seconds, 3),
                "rows_per_sec": round(scored / self._score_seconds, 1)
                if self._score_seconds > 0 else 0.0,
                "batch_max": self._batch_max,
            },
            "source_states": {
                "replay": dict(replay_state),
                "chain": dict(chain_state),
                "webhook": dict(self._source_state("webhook")),
            },
            "chain": {
                "last_block": getattr(self.sources.get("chain"), "last_block_number", None),
                "last_block_ts": getattr(self.sources.get("chain"), "last_block_ts", None),
                "last_error": getattr(self.sources.get("chain"), "last_error", None),
                "active_rpc_url": getattr(self.sources.get("chain"), "active_rpc_url", None),
                "endpoint_probed": getattr(self.sources.get("chain"), "endpoint_probed", False),
            },
            "ensembles": {
                name: ens.snapshot() for name, ens in self.ensembles.items()
            },
            "graph": {
                "nodes": len(self._graph["nodes"]),
                "edges": len(self._graph["edges"]),
                "revision": self.graph_revision,
                "accounts_tracked": self.builder.tracker.account_count(),
            },
            "scam_labels": {"available": self.matcher.available, "count": self.matcher.count},
        }
