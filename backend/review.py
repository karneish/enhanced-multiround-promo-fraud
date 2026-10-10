"""Human / LLM review queue - the labeling loop for unlabeled streams.

The replay source ships real `isFraud` labels, so it trains on its own.  The
Ethereum stream and the webhook source do not: a chain transaction is unlabeled
by nature.  Those streams only produce new labels if a human (or an LLM acting
as one) looks at the flagged events and says which are fraud.

That is what this module is for:

  1. the pipeline spots an **unlabeled** event in the Review band and spends
     review budget to *queue* it (this is the only thing the budget buys);
  2. a reviewer - a person clicking in the dashboard, or a script posting to
     `POST /api/review` - supplies a ground-truth label;
  3. the label is fed into that source's ensemble, so the next warm-start
     refit actually learns from it.

Without step 2 the loop spends budget and learns nothing, and the chain
ensemble can never leave cold start.

Candidates keep the *feature vector* that was scored, not the raw event: the
label returned here has to be trainable, and rebuilding the vector later would
re-derive profile features from a tracker that has already moved on.
"""
import collections
import threading
import time
import uuid


class ReviewQueue:
    """Bounded, thread-safe FIFO of pending review candidates."""

    def __init__(self, maxlen=200, expire_seconds=3600):
        self._items = collections.OrderedDict()
        self._lock = threading.Lock()
        self._maxlen = int(maxlen)
        self._expire = int(expire_seconds)
        # lifecycle counters (reset with the session)
        self.queued = 0
        self.resolved = 0
        self.expired = 0
        self.rejected = 0
        self.by_label = {0: 0, 1: 0}

    # ------------------------------------------------------------------ config
    def configure(self, maxlen=None, expire_seconds=None):
        with self._lock:
            if maxlen is not None:
                self._maxlen = int(maxlen)
            if expire_seconds is not None:
                self._expire = int(expire_seconds)
            self._evict_locked()

    def reset(self):
        with self._lock:
            self._items.clear()
            self.queued = 0
            self.resolved = 0
            self.expired = 0
            self.rejected = 0
            self.by_label = {0: 0, 1: 0}

    # ----------------------------------------------------------------- writing
    def submit(self, *, source, event_id, x_row, account_id, amount, prob,
               ts, decided_prob=None, extra=None):
        """Queue one candidate.  Returns the candidate dict (with `review_id`)."""
        rid = "rv-%s" % uuid.uuid4().hex[:12]
        item = {
            "review_id": rid,
            "source": source,
            "event_id": event_id,
            "account_id": account_id,
            "amount": amount,
            "prob": round(float(prob), 4),
            "decided_prob": round(float(decided_prob), 4) if decided_prob is not None else None,
            "ts": ts,
            "queued_at": time.time(),
            "label": None,
            "reviewer": None,
            "note": None,
            "resolved_at": None,
            "extra": dict(extra or {}),
        }
        with self._lock:
            self._items[rid] = {"meta": item, "x_row": list(map(float, x_row))}
            self.queued += 1
            self._evict_locked()
        return item

    def _evict_locked(self):
        now = time.time()
        if self._expire > 0:
            for rid, entry in list(self._items.items()):
                if now - entry["meta"]["queued_at"] > self._expire:
                    del self._items[rid]
                    self.expired += 1
        while len(self._items) > self._maxlen:
            _, dropped = self._items.popitem(last=False)
            if dropped["meta"]["label"] is None:
                self.expired += 1

    # ----------------------------------------------------------------- reading
    def get(self, review_id):
        with self._lock:
            entry = self._items.get(review_id)
            return entry["meta"] if entry else None

    def x_row(self, review_id):
        with self._lock:
            entry = self._items.get(review_id)
            return list(entry["x_row"]) if entry else None

    def pending(self, limit=50, source=None):
        """Oldest-first pending candidates, for the dashboard / a reviewer script."""
        with self._lock:
            self._evict_locked()
            out = []
            for entry in self._items.values():
                meta = entry["meta"]
                if meta["label"] is not None:
                    continue
                if source and meta["source"] != source:
                    continue
                out.append({k: v for k, v in meta.items() if k != "x_row"})
                if len(out) >= limit:
                    break
            return out

    def stats(self):
        with self._lock:
            self._evict_locked()
            return {
                "pending": sum(1 for e in self._items.values()
                               if e["meta"]["label"] is None),
                "held": len(self._items),
                "queued": self.queued,
                "resolved": self.resolved,
                "expired": self.expired,
                "rejected": self.rejected,
                "by_label": dict(self.by_label),
                "queue_max": self._maxlen,
                "expire_seconds": self._expire,
            }

    # ---------------------------------------------------------------- resolving
    def resolve(self, review_id, label, reviewer=None, note=None):
        """Attach a ground-truth label.  Returns (meta, error).

        The label must be exactly 0 or 1.  A non-integral value is rejected
        rather than truncated: `int(1.5) == 1` would otherwise file a
        half-confidence judgement as ground truth.
        """
        if isinstance(label, bool) or label is None:
            return None, f"label must be 0 or 1, got {label!r}"
        try:
            as_int = int(label)
        except (TypeError, ValueError):
            return None, f"label must be 0 or 1, got {label!r}"
        try:
            exact = float(as_int) == float(label)
        except (TypeError, ValueError):
            exact = False
        if not exact:
            return None, f"label must be 0 or 1, got {label!r}"
        if as_int not in (0, 1):
            return None, f"label must be 0 or 1, got {label!r}"
        with self._lock:
            entry = self._items.get(review_id)
            if entry is None:
                return None, f"unknown review_id: {review_id!r}"
            meta = entry["meta"]
            if meta["label"] is not None:
                return None, f"review_id {review_id!r} already labeled {meta['label']}"
            meta["label"] = as_int
            meta["reviewer"] = reviewer or "unknown"
            meta["note"] = note
            meta["resolved_at"] = time.time()
            self.resolved += 1
            self.by_label[as_int] = self.by_label.get(as_int, 0) + 1
            return {k: v for k, v in meta.items() if k != "x_row"}, None
