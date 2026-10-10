"""Feature building for streaming events.

Every source contributes a fixed-length numeric vector:

    vector = PROFILE_COLUMNS                   (account-history aggregates,
                                                computed from the ProfileTracker
                                                BEFORE the event is absorbed)
           + NATIVE_COLUMNS[source]            (source-specific numeric fields,
                                                carried in the event's `nat` map)

Each source therefore has its own ensemble with its own column list
(`columns_for(source)`).  The three models are written from scratch but keep
the hyper-parameter conventions documented in MODELS_REFERENCE.md.
"""
import bisect
import math

# ---------------------------------------------------------------------------
# Canonical profile columns (computed identically for every source)
# ---------------------------------------------------------------------------
PROFILE_COLUMNS = [
    "age_hours",              # time since account first seen (hours)
    "hours_since_last",       # time since the previous event from this account
    "n_events",               # lifetime event count for this account
    "events_60s",             # events in the trailing 60 seconds
    "events_1h",              # events in the trailing 1 hour
    "events_24h",             # events in the trailing 24 hours
    "total_amount",           # summed amount over the retained (24h) window
    "avg_amount",             # average amount over the trailing 24h window
    "max_amount",
    "amt_ratio",              # this amount / (previous average amount, floored)
    "n_unique_cp",            # distinct counterparties seen
    "cp_reuse_max",           # maximum count of any single counterparty
]

# ---------------------------------------------------------------------------
# Oracle-only fields
# ---------------------------------------------------------------------------
# Some `nat` fields encode the ground-truth label itself (a label bridge hit, a
# caller-supplied label).  They are kept on the event so the dashboard can show
# "known fraud", but they must never reach the model: a feature that is 1.0
# exactly when the label is 1.0 turns every reported metric into a lookup.
# `FeatureBuilder` strips these from the model vector.
ORACLE_COLUMNS = frozenset({"label_hint", "scam_match"})


# ---------------------------------------------------------------------------
# IEEE-CIS native columns (derived from the transaction rows in sources/replay.py)
# ---------------------------------------------------------------------------
IEEE_NATIVE_COLUMNS = [
    "amt", "log_amt", "hour", "dow",
    "dist1", "dist2",
    "c_sum", "c_mean", "c_max",       # C1..C14 velocity aggregates
    "d_sum",                          # D1..D15 time-decimal aggregates
    "num_nan_v",                      # missing V1..V339 indicators
    "product_cd_code",
    "has_email", "has_r_email",
]

# ---------------------------------------------------------------------------
# Chain native columns (Ethereum stream in sources/ethereum.py)
# ---------------------------------------------------------------------------
CHAIN_NATIVE_COLUMNS = [
    "value_eth", "log_value_eth",
    "gas_gwei", "tx_index",
]

# ---------------------------------------------------------------------------
# Webhook native columns (generic external producer)
# ---------------------------------------------------------------------------
WEBHOOK_NATIVE_COLUMNS = [
    "txn_amt", "n_attrs",
]

# ---------------------------------------------------------------------------
# Adversarial multi-round source (Prasetya et al. 2026 scenario, restored)
# ---------------------------------------------------------------------------
ADVER_NATIVE_COLUMNS = [
    "round",              # which generated round this event belongs to
    "age",                # rounds since this node was created
    "deg",                # current degree
    "is_new",             # 1.0 for a node generated this round
    "feat_mean",          # summary of the raw node feature vector
    "feat_std",
]

NATIVE_COLUMNS = {
    "ieee": IEEE_NATIVE_COLUMNS,
    "chain": CHAIN_NATIVE_COLUMNS,
    "webhook": WEBHOOK_NATIVE_COLUMNS,
    "adver": ADVER_NATIVE_COLUMNS,
}

#: The anonymised IEEE-CIS V1..V339 block. These are read by
#: ``sources/replay.py`` but are only passed to the model when
#: ``ieee.use_v_features`` is on; otherwise only ``num_nan_v`` (the count of
#: missing values in that block) is used.
V_FEATURE_COLUMNS = ["V%d" % i for i in range(1, 340)]


def ieee_columns(include_v=False):
    """IEEE feature columns, optionally including V1..V339.

    ``columns_for("ieee")`` returns the 26-column default so that existing
    behaviour, tests and the reported numbers are unchanged.
    """
    if include_v:
        return IEEE_NATIVE_COLUMNS + V_FEATURE_COLUMNS
    return IEEE_NATIVE_COLUMNS


def _as_float(val, default=0.0):
    """Best-effort numeric coercion that tolerates None, NaN and numpy scalars."""
    if val is None:
        return default
    if isinstance(val, bool):
        return 1.0 if val else 0.0
    try:
        out = float(val)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) else out


def columns_for(source: str) -> list:
    if source not in NATIVE_COLUMNS:
        raise ValueError(f"no feature schema for source: {source!r}")
    return PROFILE_COLUMNS + NATIVE_COLUMNS[source]


def decision_from_prob(prob: float, thresholds: dict) -> str:
    if prob >= thresholds.get("block", 0.70):
        return "block"
    if prob >= thresholds.get("review", 0.30):
        return "review"
    return "allow"


# ---------------------------------------------------------------------------
# Profile tracker: rolling per-account history (bounded to 24h of memory)
# ---------------------------------------------------------------------------
# Window lookups use bisect over parallel timestamp/amount lists, so the cost of
# building one event's profile is O(log n + window) rather than O(history). The
# lists are append-ordered and pruned from the front, so they stay sorted.
KEEP_WINDOW = 24 * 3600


class _AccountProfile:
    __slots__ = ("first_ts", "last_ts", "n_events", "total_amount",
                 "event_ts", "event_amt", "cp_counts")

    def __init__(self, ts, amount):
        self.first_ts = ts
        self.last_ts = ts
        self.n_events = 0
        self.total_amount = 0.0
        self.event_ts = []
        self.event_amt = []
        # counterparty -> [count, last_seen_ts]
        self.cp_counts = {}

    def prune(self, now_ts, keep_window=KEEP_WINDOW):
        cutoff = now_ts - keep_window
        k = 0
        ts_list = self.event_ts
        n = len(ts_list)
        # events are time-ordered; scan (rather than bisect) so that a
        # backwards timestamp also gets dropped instead of breaking the order
        while k < n and ts_list[k] < cutoff:
            k += 1
        if k:
            self.total_amount -= sum(self.event_amt[:k])
            del self.event_ts[:k]
            del self.event_amt[:k]
        if len(self.cp_counts) > 1:
            stale = [cp for cp, st in self.cp_counts.items() if st[1] < cutoff]
            for cp in stale:
                del self.cp_counts[cp]
        return k > 0

    def window(self, ts, window_s):
        """(count, sum_amounts, max_amount) over the trailing `window_s`."""
        cutoff = ts - window_s
        ts_list = self.event_ts
        amt_list = self.event_amt
        n = len(ts_list)
        i = bisect.bisect_left(ts_list, cutoff)
        if i >= n:
            return 0, 0.0, 0.0
        seg = amt_list[i:]
        return (n - i, float(sum(seg)), float(max(seg)) if seg else 0.0)

    def count_since(self, ts, window_s):
        cutoff = ts - window_s
        return len(self.event_ts) - bisect.bisect_left(self.event_ts, cutoff)

    def add_event(self, ts, amount, counterparties):
        self.event_ts.append(ts)
        self.event_amt.append(amount)
        cps = self.cp_counts
        for cp in counterparties:
            st = cps.get(cp)
            if st is None:
                cps[cp] = [1, ts]
            else:
                st[0] += 1
                st[1] = ts

    def cp_summary(self):
        if not self.cp_counts:
            return 0, 0
        reuse_max = 0
        for st in self.cp_counts.values():
            if st[0] > reuse_max:
                reuse_max = st[0]
        return len(self.cp_counts), reuse_max


class ProfileTracker:
    """Tracks per-(source, account) histories.  Only ever read/written from a
    single consumer thread, so no locking is required."""

    def __init__(self):
        self._profiles = {}
        self._default_profile = None

    def _get(self, source, account_id):
        return self._profiles.get((source, account_id))

    def _touch(self, source, account_id, ev):
        key = (source, account_id)
        prof = self._profiles.get(key)
        ts = ev["ts"]
        amount = ev.get("amount") or 0.0
        cps = ev.get("counterparties") or ()
        if prof is None:
            prof = _AccountProfile(ts, amount)
            self._profiles[key] = prof
        else:
            prof.prune(ts)
        prof.n_events += 1
        prof.total_amount += amount
        prof.last_ts = ts
        prof.add_event(ts, amount, cps)
        return prof

    def build(self, ev):
        """Compute the profile feature vector from account state BEFORE the
        event is absorbed into the tracker."""
        source = ev["source"]
        prof = self._get(source, ev["account_id"])
        ts = ev["ts"]
        amount = ev.get("amount") or 0.0

        if prof is None:
            return dict.fromkeys(PROFILE_COLUMNS, 0.0)

        # Age the profile out on the read path as well as on absorb: a build
        # after a long gap would otherwise report a `total_amount` that still
        # includes events from days ago (pruning only happened on the next
        # absorb).
        prof.prune(ts)

        e60 = prof.count_since(ts, 60.0)
        e1h = prof.count_since(ts, 3600.0)
        e24, sum24, max24 = prof.window(ts, KEEP_WINDOW)

        avg_amount = (sum24 / e24) if e24 else 0.0
        n_unique_cp, reuse_max = prof.cp_summary()

        age_hours = max(0.0, (ts - prof.first_ts) / 3600.0)
        since_last = max(0.0, (ts - prof.last_ts) / 3600.0)
        # ratio to the *previous* average; folding `amount` into the denominator
        # (as this once did) clamps the feature at 1.0 for every above-average
        # event, which is exactly the population the feature exists to flag.
        amt_ratio = (amount / avg_amount) if avg_amount > 1e-9 else 1.0

        return {
            "age_hours": age_hours,
            "hours_since_last": since_last,
            "n_events": float(prof.n_events),          # events absorbed BEFORE this one
            "events_60s": float(e60),
            "events_1h": float(e1h),
            "events_24h": float(e24),
            "total_amount": float(prof.total_amount),
            "avg_amount": float(avg_amount),
            "max_amount": float(max24),
            "amt_ratio": float(amt_ratio),
            "n_unique_cp": float(n_unique_cp),
            "cp_reuse_max": float(reuse_max),
        }

    def absorb(self, ev):
        self._touch(ev["source"], ev["account_id"], ev)

    def account_count(self):
        return len(self._profiles)


# ---------------------------------------------------------------------------
# FeatureBuilder
# ---------------------------------------------------------------------------
class FeatureBuilder:
    def __init__(self, extra_columns=None):
        self.tracker = ProfileTracker()
        #: optional per-source column extensions, e.g. {"ieee": ["V1", ...]}
        self._extra = {k: list(v) for k, v in (extra_columns or {}).items()}

    def build(self, ev):
        """Return a dict with exactly ``columns_for(ev['source'])`` keys.

        Oracle fields (``ORACLE_COLUMNS``) are read from ``nat`` for display but
        are never written into the returned vector. When the caller asks for
        extra source-native columns that the source did not supply (for
        example V1..V339 from a synthetic sample, which has no real V values),
        they default to 0.0 rather than raising, so an opt-in column list
        cannot break a run that does not populate it.
        """
        cols = self.columns_for_source(ev["source"])
        vec = self.tracker.build(ev)
        native = ev.get("nat") or {}
        for col in NATIVE_COLUMNS[ev["source"]]:
            if col in ORACLE_COLUMNS:
                continue
            vec[col] = _as_float(native.get(col))
        return {c: _as_float(vec.get(c)) for c in cols}

    def columns_for_source(self, source):
        """Column list for a source, honouring any configured extension.

        ``FeatureBuilder(extra_columns={"ieee": ["V1", ...]})`` appends those
        columns; the default stays exactly ``columns_for(source)``.
        """
        cols = PROFILE_COLUMNS + NATIVE_COLUMNS[source]
        extra = (self._extra or {}).get(source)
        if extra:
            cols = cols + [c for c in extra if c not in cols]
        return cols

    def oracle(self, ev):
        """The event's oracle-only `nat` fields (display only, never modelled)."""
        native = ev.get("nat") or {}
        return {c: _as_float(native.get(c)) for c in ORACLE_COLUMNS if c in native}

    def absorb(self, ev):
        self.tracker.absorb(ev)