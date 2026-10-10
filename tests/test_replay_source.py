"""Tests for the replay reader: load-time sort, slicing, typing, pacing.

The dataset is 651 MB, so the reader must never read more than the requested
window and must not degrade every numeric column to a Python object.
"""
import os
import tempfile
import threading
import time
import unittest

import numpy as np
import pandas as pd

from backend.sources.replay import IeeeReplaySource


IDENTITY_HEADER = ("TransactionID,isFraud,card*,addr*,P_emaildomain,P_phoneprefix,"
                   "DeviceInfo,cardholder_age,addr_,P_")

TXN_COLUMNS = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
               "ProductCD", "card4", "card6", "addr1", "P_emaildomain", "P_phoneprefix"]


def write_dataset(directory, n=200, step=100.0, fraud=None, shuffle=False):
    """Write a small IEEE-CS-like dataset.  `fraud` is a set of indices."""
    fraud = fraud or set()
    rows = []
    for i in range(n):
        rows.append({
            "TransactionID": 1000 + i,
            "isFraud": 1 if i in fraud else 0,
            "TransactionDT": 1_000_000.0 + i * step,
            "TransactionAmt": 10.0 + (i % 17),
            "ProductCD": "C" if i % 2 else "A",
            "card4": "123456789012%04d" % (i % 3),
            "card6": "411111" if i % 2 else "522005",
            "addr1": 100 + (i % 5),
            "P_emaildomain": "gmail.com" if i % 2 else "yahoo.com",
            "P_phoneprefix": "130" if i % 2 else "131",
        })
    frame = pd.DataFrame(rows, columns=TXN_COLUMNS)
    if shuffle:
        frame = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    txn = os.path.join(directory, "train_transaction.csv")
    frame.to_csv(txn, index=False)

    ident_rows = []
    for i in range(n):
        ident_rows.append({
            "TransactionID": 1000 + i,
            "isFraud": 1 if i in fraud else 0,
            "card1": "A" * 6,
            "card2": "B" * 6,
            "card3": "C" * 6,
            "card4": "123456789012%04d" % (i % 3),
            "card5": "D" * 6,
            "addr1": 100 + (i % 5),
            "addr2": "E" * 6,
            "P_emaildomain": "gmail.com" if i % 2 else "yahoo.com",
            "P_1": "abc",
            "P_2": "def",
            "P_3": "ghi",
            "P_4": "jkl",
            "P_5": "mno",
            "P_6": "pqr",
            "P_7": "stu",
            "P_8": "vwx",
            "P_9": "yz0",
            "P_10": "123",
            "DOB": 1990,
            "DOB_1": 1990,
            "DeviceInfo": "dev-a",
            "DeviceInfo1": "dev-b",
            "DeviceInfo2": "dev-c",
            "DeviceInfo3": "dev-d",
            "DeviceInfo4": "dev-e",
            "DeviceInfo5": "dev-f",
            "DeviceInfo6": "dev-g",
            "DeviceInfo7": "dev-h",
            "DeviceInfo8": "dev-i",
            "DeviceInfo9": "dev-j",
            "DeviceInfo10": "dev-k",
        })
    pd.DataFrame(ident_rows).to_csv(
        os.path.join(directory, "train_identity.csv"), index=False)
    return txn


class TempDataset(unittest.TestCase):
    fraud = None
    n = 200
    shuffle = False

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.txn = write_dataset(self.dir.name, n=self.n, fraud=self.fraud,
                                 shuffle=self.shuffle)
        self.cfg = {
            "stream_slice": 40,
            "stream_skip": 0,
            "events_per_sec": 0,
            "ieee": {"data_dir": self.dir.name, "sample_fallback": False},
        }

    def tearDown(self):
        self.dir.cleanup()


class TestLoad(TempDataset):
    fraud = {5, 6, 7}

    def loaded(self, **over):
        cfg = dict(self.cfg)
        cfg.update(over)
        src = IeeeReplaySource(cfg)
        status = src.load(cancel=lambda: False)
        self.assertTrue(status.get("loaded"), status)
        return src, src.frame

    def test_slice_is_time_ordered_and_typed(self):
        src, df = self.loaded()
        self.assertEqual(len(df), 40)
        dt = df["TransactionDT"].to_numpy()
        self.assertTrue(np.all(np.diff(dt) >= 0), "not sorted by time")
        self.assertEqual(dt[0], 1_000_000.0)
        # numeric columns must survive as numbers, not object/float-from-str
        self.assertTrue(pd.api.types.is_numeric_dtype(df["TransactionAmt"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(df["addr1"]))
        self.assertFalse(pd.api.types.is_object_dtype(df["TransactionAmt"]))

    def test_skip_then_slice(self):
        src, df = self.loaded(**{"ieee": {**self.cfg["ieee"],
                                         "replay_skip_first": 100}})
        self.assertEqual(len(df), 40)
        self.assertAlmostEqual(float(df["TransactionDT"].iloc[0]), 1_010_000.0, places=3)

    def test_full_load_when_slice_is_zero(self):
        src, df = self.loaded(stream_slice=0)
        self.assertEqual(len(df), self.n)

    def test_labels_survive(self):
        src, df = self.loaded()
        self.assertEqual(int(df["isFraud"].sum()), 3)

    def test_identity_is_joined_on_transaction_id(self):
        src, df = self.loaded()
        # the join is deliberately limited to the card columns the event
        # builder consumes; the other identity columns are never read
        for col in ("card1", "card2", "card3", "card5"):
            self.assertIn(col, df.columns)
            self.assertEqual(bool(df[col].notna().all()), True,
                             "%s did not survive the join" % col)
        self.assertNotIn("card4_x", df.columns)
        self.assertEqual(len(df), 40, "the join must not duplicate rows")

    def test_cancel_is_checked_while_loading(self):
        cfg = dict(self.cfg)
        src = IeeeReplaySource(cfg)
        status = src.load(cancel=lambda: True)
        self.assertFalse(status.get("loaded"))
        self.assertIsNone(src.frame)

    def test_missing_file_is_reported_not_fatal(self):
        cfg = dict(self.cfg)
        cfg["ieee"] = {**self.cfg["ieee"],
                       "data_dir": os.path.join(self.dir.name, "nope")}
        src = IeeeReplaySource(cfg)
        status = src.load(cancel=lambda: False)
        self.assertFalse(status.get("loaded"))
        self.assertIn("error", status)


class TestUnsortedInput(TempDataset):
    shuffle = True

    def test_out_of_order_csv_is_sorted_at_load_time(self):
        src = IeeeReplaySource(dict(self.cfg, stream_slice=0))
        status = src.load(cancel=lambda: False)
        self.assertTrue(status.get("loaded"), status)
        df = src.frame
        self.assertEqual(len(df), self.n)
        self.assertTrue(np.all(np.diff(df["TransactionDT"].to_numpy()) >= 0))


class TestStream(TempDataset):
    n = 60

    def source(self, **over):
        cfg = dict(self.cfg)
        cfg.update(over)
        src = IeeeReplaySource(cfg)
        status = src.load(cancel=lambda: False)
        self.assertTrue(status.get("loaded"), status)
        return src

    def test_stream_yields_events_then_a_none_sentinel(self):
        src = self.source()
        events = list(src.stream(threading.Event()))
        self.assertEqual(len(events), 41)
        self.assertTrue(all(e is not None for e in events[:-1]))
        self.assertEqual(events[-1], None)
        self.assertEqual(len([e for e in events if e is not None]), 40)

    def test_event_shape(self):
        src = self.source()
        ev = next(iter(src.stream(threading.Event())))
        for key in ("source", "account_id", "amount", "ts", "counterparties"):
            self.assertIn(key, ev)
        self.assertIsNotNone(ev["account_id"])
        self.assertGreater(ev["amount"], 0)
        self.assertEqual(ev["ts"], 1_000_000.0)

    def test_counterparties_are_namespaced(self):
        src = self.source()
        first = next(iter(src.stream(threading.Event())))
        self.assertTrue(first["counterparties"])
        for cp in first["counterparties"]:
            self.assertIn(":", cp)

    def test_stop_flag_ends_the_stream(self):
        src = self.source()
        stop = threading.Event()
        seen = []
        for ev in src.stream(stop):
            if ev is not None:
                seen.append(ev)
                if len(seen) == 3:
                    stop.set()
        self.assertEqual(len(seen), 3)

    def test_cancel_callable_ends_the_stream(self):
        src = self.source()
        seen = []
        for ev in src.stream(threading.Event(), cancel=lambda: len(seen) >= 2):
            if ev is not None:
                seen.append(ev)
        self.assertEqual(len(seen), 2)

    def test_unloaded_source_yields_only_the_sentinel(self):
        src = IeeeReplaySource(self.cfg)
        self.assertEqual(list(src.stream(threading.Event())), [None])


class TestPacer(unittest.TestCase):
    def test_fast_rates_are_not_limited_by_the_windows_timer(self):
        from backend.sources.replay import _Pacer
        for rate in (5.0, 20.0, 200.0, 2000.0):
            pacer = _Pacer(rate, threading.Event())
            t0 = time.perf_counter()
            for _ in range(5):
                pacer.wait()
            elapsed = time.perf_counter() - t0
            expected = 5.0 / rate
            # never faster than requested, and not orders of magnitude slower
            self.assertGreaterEqual(elapsed, expected * 0.9, "rate %.0f" % rate)
            self.assertLess(elapsed, expected + 1.0, "rate %.0f" % rate)

    def test_zero_rate_does_not_sleep(self):
        from backend.sources.replay import _Pacer
        pacer = _Pacer(0, threading.Event())
        t0 = time.perf_counter()
        for _ in range(1000):
            pacer.wait()
        self.assertLess(time.perf_counter() - t0, 0.5)

    def test_stop_flag_short_circuits(self):
        from backend.sources.replay import _Pacer
        stop = threading.Event()
        stop.set()
        pacer = _Pacer(1.0, stop)
        t0 = time.perf_counter()
        self.assertTrue(pacer.wait())
        self.assertLess(time.perf_counter() - t0, 0.5)


if __name__ == "__main__":
    unittest.main()
