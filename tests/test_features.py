"""Regression tests for the feature layer.

These lock in the bugs found in the audit:
  * oracle fields (`label_hint`, `scam_match`) reaching the model vector
  * `n_events` counting the event being built instead of the events before it
  * unbounded / wrong window lookups in the profile tracker
"""
import unittest

from backend.events import normalize_event
from backend.features import (ORACLE_COLUMNS, PROFILE_COLUMNS, FeatureBuilder,
                              columns_for, decision_from_prob)


def ev(ts, amount=10.0, account="a1", cps=("cp1",), label=None, nat=None,
       source="webhook"):
    return normalize_event(
        source=source, ts=ts, account_id=account, counterparties=list(cps),
        label=label, amount=amount, native=nat or {}, raw={}, event_id="e",
    )


class TestOracleLeakage(unittest.TestCase):
    def test_oracle_fields_are_excluded_from_the_vector(self):
        cols = columns_for("webhook")
        for name in ORACLE_COLUMNS:
            self.assertNotIn(name, cols, "%s must never be a model feature" % name)

    def test_build_ignores_oracle_values(self):
        b = FeatureBuilder()
        honest = ev(1000.0, nat={"txn_amt": 5.0, "label_hint": 0.0, "scam_match": 0.0})
        leaky = ev(1000.0, nat={"txn_amt": 5.0, "label_hint": 1.0, "scam_match": 1.0})
        self.assertEqual(b.build(honest), b.build(leaky),
                         "flipping an oracle field must not change the feature row")

    def test_oracle_is_still_reported_for_display(self):
        b = FeatureBuilder()
        e = ev(1000.0, nat={"txn_amt": 5.0, "scam_match": 1.0})
        self.assertEqual(b.oracle(e), {"scam_match": 1.0})

    def test_chain_oracle_marker_does_not_change_features(self):
        b1, b2 = FeatureBuilder(), FeatureBuilder()
        a = ev(1000.0, amount=3.0, source="chain",
               nat={"value_eth": 3.0, "gas_gwei": 5.0, "tx_index": 0.0})
        c = ev(1000.0, amount=3.0, source="chain",
               nat={"value_eth": 3.0, "gas_gwei": 5.0, "tx_index": 0.0,
                    "scam_match": 1.0})
        self.assertEqual(b1.build(a), b2.build(c))

    def test_vector_has_exactly_the_schema_keys(self):
        b = FeatureBuilder()
        for source in ("ieee", "chain", "webhook"):
            cols = columns_for(source)
            self.assertEqual(set(b.build(ev(1.0, source=source))), set(cols))
            self.assertEqual(set(cols[: len(PROFILE_COLUMNS)]), set(PROFILE_COLUMNS))


class TestProfileTracker(unittest.TestCase):
    def test_n_events_counts_prior_events_only(self):
        b = FeatureBuilder()
        self.assertEqual(b.build(ev(100.0))["n_events"], 0.0)
        b.absorb(ev(100.0))
        self.assertEqual(b.build(ev(200.0))["n_events"], 1.0)
        b.absorb(ev(200.0))
        self.assertEqual(b.build(ev(300.0))["n_events"], 2.0)

    def test_trailing_window_counts(self):
        b = FeatureBuilder()
        for ts in [0.0, 30.0, 45.0, 4000.0, 7200.0, 7460.0]:
            b.absorb(ev(ts, account="a1"))
        row = b.build(ev(7500.0, account="a1"))
        # the 7460 event is 40s old; 4000/7200 are inside 1h; all inside 24h
        self.assertEqual(row["events_60s"], 1.0)
        self.assertEqual(row["events_1h"], 3.0)
        self.assertEqual(row["events_24h"], 6.0)

    def test_window_boundaries_include_the_cutoff(self):
        b = FeatureBuilder()
        b.absorb(ev(0.0))       # exactly 3600s before the probe -> counted
        b.absorb(ev(0.001))     # 3599.999s before           -> counted
        # the trailing window is the closed interval [ts - window_s, ts]
        self.assertEqual(b.build(ev(3600.0))["events_1h"], 2.0)
        b2 = FeatureBuilder()
        b2.absorb(ev(0.0))
        # 3600.001s before the probe -> outside
        self.assertEqual(b2.build(ev(3600.001))["events_1h"], 0.0)

    def test_amount_aggregates_exclude_the_current_event(self):
        b = FeatureBuilder()
        b.absorb(ev(100.0, amount=10.0))
        b.absorb(ev(200.0, amount=30.0))
        row = b.build(ev(300.0, amount=100.0))
        self.assertEqual(row["total_amount"], 40.0)
        self.assertEqual(row["max_amount"], 30.0)
        self.assertEqual(row["avg_amount"], 20.0)
        self.assertGreater(row["amt_ratio"], 1.0)   # 100 vs a 20 average

    def test_history_is_pruned_after_the_keep_window(self):
        b = FeatureBuilder()
        b.absorb(ev(0.0, amount=10.0))
        b.absorb(ev(1.0, amount=10.0))
        later = 24 * 3600 + 100.0
        row = b.build(ev(later, account="a1"))
        self.assertEqual(row["events_24h"], 0.0)
        self.assertEqual(row["total_amount"], 0.0)
        # the account is still tracked, it has simply aged out
        self.assertEqual(b.tracker.account_count(), 1)

    def test_profiles_are_isolated_per_source_and_account(self):
        b = FeatureBuilder()
        b.absorb(ev(100.0, account="a1", source="webhook"))
        self.assertEqual(b.build(ev(150.0, account="a1", source="webhook"))["n_events"], 1.0)
        self.assertEqual(b.build(ev(150.0, account="a1", source="chain"))["n_events"], 0.0)
        self.assertEqual(b.build(ev(150.0, account="a2", source="webhook"))["n_events"], 0.0)

    def test_counterparty_reuse(self):
        b = FeatureBuilder()
        b.absorb(ev(100.0, cps=("cp1", "cp2")))
        b.absorb(ev(200.0, cps=("cp1",)))
        row = b.build(ev(300.0))
        self.assertEqual(row["n_unique_cp"], 2.0)
        self.assertEqual(row["cp_reuse_max"], 2.0)

    def test_hours_since_last_is_never_negative(self):
        b = FeatureBuilder()
        b.absorb(ev(5000.0))
        self.assertGreaterEqual(b.build(ev(10.0))["hours_since_last"], 0.0)


class TestDecisions(unittest.TestCase):
    def test_bands(self):
        th = {"review": 0.3, "block": 0.7}
        self.assertEqual(decision_from_prob(0.1, th), "allow")
        self.assertEqual(decision_from_prob(0.3, th), "review")
        self.assertEqual(decision_from_prob(0.69, th), "review")
        self.assertEqual(decision_from_prob(0.7, th), "block")


if __name__ == "__main__":
    unittest.main()
