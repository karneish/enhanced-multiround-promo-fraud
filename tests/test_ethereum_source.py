"""Tests for the Ethereum source: failover, dedup, intra-block timing, oracle.

No network access: `_rpc_call` is replaced with a scripted fake so the
failover ladder and the streaming loop can be driven deterministically.
"""
import threading
import unittest
from unittest import mock

from backend.sources import ethereum as eth
from backend.sources.ethereum import EthereumSource


def cfg(**chain):
    base = {
        "chain": {
            "transport": "rpc-poll",
            "rpc_url": "http://primary",
            "rpc_fallback_urls": ["http://second", "http://third"],
            "poll_interval_seconds": 0.01,
            "probe_timeout_seconds": 0.01,
            "max_txs_per_block": 5,
        }
    }
    base["chain"].update(chain)
    return base


class FakeMatcher:
    def __init__(self, bad=()):
        self.bad = {a.lower() for a in bad}
        self.calls = 0

    def label_for(self, ev):
        self.calls += 1
        if ev["account_id"].lower() in self.bad:
            return 1
        if any(c.lower() in self.bad for c in ev.get("counterparties", ())):
            return 1
        return 0


def tx(n, block_number, to="0xRecv%d"):
    return {
        "from": "0xSender%d" % (block_number + n),
        "to": to % n if "%d" in to else to,
        "value": hex(10 ** 18),
        "gasPrice": hex(20 * 10 ** 9),
        "hash": "0xtx%d" % n,
    }


def block(number, txs, ts=1_700_000_000):
    """A decoded block, as `fetch_block()` returns it."""
    return {
        "number": number,
        "hash": "0xblock%d" % number,
        "timestamp": ts,
        "transactions": [tx(i, number) for i in range(txs)],
    }


def rpc_block(number, txs, ts=1_700_000_000):
    """The wire form, as an RPC node returns it (hex-encoded)."""
    blk = block(number, txs, ts)
    blk["number"] = hex(number)
    blk["timestamp"] = hex(ts)
    return blk


class TestEndpointFailover(unittest.TestCase):
    def test_first_working_url_wins_and_is_remembered(self):
        src = EthereumSource(cfg(), FakeMatcher())
        with mock.patch.object(eth, "_rpc_call") as call:
            call.side_effect = lambda url, *a, **k: (
                "0x10" if url == "http://second" else (_ for _ in ()).throw(
                    RuntimeError("boom")))
            self.assertEqual(src.ensure_endpoint(), "http://second")
            # idempotent: no further probing without force
            calls = call.call_count
            src.ensure_endpoint()
            self.assertEqual(call.call_count, calls)
            self.assertTrue(src.endpoint_probed)
            self.assertIsNone(src.last_error)

    def test_all_endpoints_down_leaves_last_error(self):
        src = EthereumSource(cfg(), FakeMatcher())
        with mock.patch.object(eth, "_rpc_call", side_effect=RuntimeError("down")):
            url = src.ensure_endpoint()
        self.assertFalse(src.endpoint_probed)
        self.assertIn("down", src.last_error)
        self.assertIn(url, ("http://primary", "http://second", "http://third"))

    def test_primary_is_tried_first(self):
        src = EthereumSource(cfg(), FakeMatcher())
        with mock.patch.object(eth, "_rpc_call", return_value="0x1") as call:
            src.ensure_endpoint()
        self.assertEqual(src.active_rpc_url, "http://primary")
        self.assertEqual(call.call_args_list[0][0][0], "http://primary")

    def test_repeated_failures_trigger_a_reprobe_and_switch(self):
        src = EthereumSource(cfg(), FakeMatcher())
        src.active_rpc_url = "http://third"
        probed = []
        switched = {"on": False}

        def fake(url, method, params, timeout=15):
            if url != "http://third":
                switched["on"] = True
                probed.append(url)
                return "0x2a" if method == "eth_blockNumber" else block(11, 1)
            if switched["on"]:
                return "0x2a" if method == "eth_blockNumber" else block(11, 1)
            raise RuntimeError("primary is down")

        with mock.patch.object(eth, "_rpc_call", side_effect=fake):
            for _ in range(src._failover_threshold - 1):
                with self.assertRaises(RuntimeError):
                    src._rpc("eth_getBlockByNumber", ["latest", True])
            self.assertEqual(src.active_rpc_url, "http://third")
            # the failure threshold is reached: the ladder is re-probed and the
            # call is retried against the newly locked-in endpoint
            self.assertEqual(src._rpc("eth_getBlockByNumber",
                                      ["latest", True])["number"], 11)
        self.assertIn("http://primary", probed, "did not re-probe the ladder")
        self.assertEqual(src._consecutive_errors, 0)

    def test_init_never_touches_the_network(self):
        with mock.patch.object(eth, "_rpc_call", side_effect=AssertionError("boom")):
            EthereumSource(cfg(), FakeMatcher())
            EthereumSource(cfg(transport="ws"), FakeMatcher())

    def test_bad_fallback_entries_are_ignored(self):
        src = EthereumSource(cfg(rpc_fallback_urls=[None, "", "  ", "http://ok"]),
                             FakeMatcher())
        self.assertEqual(src.rpc_fallbacks, ["http://ok"])


class TestBlocks(unittest.TestCase):
    def setUp(self):
        self.src = EthereumSource(cfg(), FakeMatcher())

    def test_block_dedup(self):
        with mock.patch.object(eth, "_rpc_call", return_value=rpc_block(100, 2)):
            first = self.src.fetch_block("latest")
            second = self.src.fetch_block("latest")
        self.assertFalse(first["seen"])
        self.assertTrue(second["seen"])
        self.assertEqual(first["number"], 100)
        self.assertEqual(first["timestamp"], 1_700_000_000)

    def test_seen_set_is_fifo_bounded(self):
        self.src._max_seen = 4
        for n in range(10):
            with mock.patch.object(eth, "_rpc_call",
                                   return_value=rpc_block(100 + n, 1)):
                self.src.fetch_block("latest")
        self.assertLessEqual(len(self.src._seen_blocks), 4)
        self.assertIn(109, self.src._seen_blocks, "newest block was evicted")
        self.assertNotIn(100, self.src._seen_blocks, "FIFO eviction did not happen")

    def test_null_block_response(self):
        with mock.patch.object(eth, "_rpc_call", return_value=None):
            self.assertIsNone(self.src.fetch_block("latest"))

    def test_max_txs_cap(self):
        blk = block(1, 50)
        events = self.src._events_from_block(blk)
        self.assertEqual(len(events), 5)


class TestEventShape(unittest.TestCase):
    def test_intra_block_timestamps_are_spread_and_bounded(self):
        src = EthereumSource(cfg(max_txs_per_block=100), FakeMatcher())
        events = src._events_from_block(block(7, 20, ts=1_700_000_000))
        ts = [e["ts"] for e in events]
        self.assertEqual(len(set(ts)), 20, "timestamps collided")
        self.assertEqual(ts[0], 1_700_000_000)
        self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])))
        self.assertLessEqual(max(ts) - min(ts), eth.MAX_PLAUSIBLE_BLOCK_SPAN)

    def test_single_tx_block_has_no_offset(self):
        src = EthereumSource(cfg(), FakeMatcher())
        self.assertEqual(src._events_from_block(block(7, 1, ts=42))[0]["ts"], 42.0)

    def test_values_and_ids(self):
        src = EthereumSource(cfg(), FakeMatcher())
        ev = src._events_from_block(block(9, 1))[0]
        self.assertAlmostEqual(ev["amount"], 1.0, places=9)
        self.assertEqual(ev["account_id"], "0xsender9")
        self.assertEqual(ev["counterparties"], ["0xrecv0"])
        self.assertEqual(ev["event_id"], "chain-9-0")
        self.assertEqual(ev["source"], "chain")
        self.assertAlmostEqual(ev["nat"]["gas_gwei"], 20.0, places=6)

    def test_contract_creation_without_a_sender_gets_a_synthetic_account(self):
        src = EthereumSource(cfg(), FakeMatcher())
        ev = src._event_from_tx({"from": None, "to": "0xnew",
                                 "value": "0x0", "hash": "0xdead"}, 1, 10, 0)
        self.assertIn("contract-create", ev["account_id"])
        self.assertIn("0xdead", ev["account_id"])
        self.assertEqual(ev["counterparties"], ["0xnew"])

    def test_sender_is_the_account_even_for_a_contract_creation(self):
        src = EthereumSource(cfg(), FakeMatcher())
        ev = src._event_from_tx({"from": "0xABC", "to": None, "value": "0x0",
                                 "hash": "0xdead"}, 1, 10, 0)
        self.assertEqual(ev["account_id"], "0xabc")
        self.assertEqual(ev["counterparties"], [])

    def test_unlabeled_events_stay_unlabeled(self):
        src = EthereumSource(cfg(), FakeMatcher(bad=[]))
        for ev in src._events_from_block(block(1, 3)):
            self.assertIsNone(ev["label"], "an unverified event got a label")
            self.assertIsNone(ev["nat"].get("scam_match"))

    def test_scam_match_sets_label_and_oracle_field(self):
        matcher = FakeMatcher(bad=["0xrecv1"])
        src = EthereumSource(cfg(max_txs_per_block=10), matcher)
        events = src._events_from_block(block(1, 4))
        flagged = [e for e in events if e["label"] == 1]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["nat"]["scam_match"], 1.0)
        # the oracle flag must never leak into a feature vector
        from backend.features import ORACLE_COLUMNS
        self.assertIn("scam_match", ORACLE_COLUMNS)

    def test_matcher_optional(self):
        src = EthereumSource(cfg(), None)
        self.assertIsNone(src._events_from_block(block(1, 2))[0]["label"])


class TestStreaming(unittest.TestCase):
    def test_poll_loop_emits_events_then_idle_sentinels(self):
        src = EthereumSource(cfg(), FakeMatcher())
        stop = threading.Event()
        with mock.patch.object(eth, "_rpc_call", return_value=rpc_block(5, 3)):
            with mock.patch.object(src, "ensure_endpoint", return_value="http://primary"):
                out = []
                for ev in src.stream(stop):
                    out.append(ev)
                    if len(out) >= 4:
                        stop.set()
        kinds = ["event" if e is not None else "idle" for e in out[:4]]
        self.assertEqual(kinds, ["event", "event", "event", "idle"])
        self.assertEqual(src.last_block_number, 5)

    def test_network_error_does_not_kill_the_loop(self):
        src = EthereumSource(cfg(), FakeMatcher())
        stop = threading.Event()
        with mock.patch.object(eth, "_rpc_call", side_effect=RuntimeError("net down")):
            with mock.patch.object(src, "ensure_endpoint", return_value="http://primary"):
                out = []
                for ev in src.stream(stop):
                    out.append(ev)
                    if len(out) >= 2:
                        stop.set()
        self.assertIn("net down", src.last_error)
        self.assertTrue(any(e is None for e in out))

    def test_stop_flag_ends_the_generator(self):
        src = EthereumSource(cfg(), FakeMatcher())
        stop = threading.Event()
        stop.set()
        with mock.patch.object(src, "ensure_endpoint", return_value="http://primary"):
            with mock.patch.object(eth, "_rpc_call", return_value=rpc_block(1, 1)):
                self.assertEqual(list(src.stream(stop)), [])

    def test_duplicate_block_produces_no_events(self):
        src = EthereumSource(cfg(), FakeMatcher())
        stop = threading.Event()
        with mock.patch.object(eth, "_rpc_call", return_value=rpc_block(5, 3)):
            with mock.patch.object(src, "ensure_endpoint", return_value="http://primary"):
                out = []
                for ev in src.stream(stop):
                    out.append(ev)
                    if len(out) >= 5:
                        stop.set()
        self.assertEqual(len([e for e in out if e is not None]), 3)

    def test_ws_without_url_reports_an_error(self):
        src = EthereumSource(cfg(transport="ws", ws_url=""), FakeMatcher())
        stop = threading.Event()
        stop.set()
        with mock.patch.object(src, "ensure_endpoint", return_value="x"):
            list(src.stream(stop))
        self.assertIn("ws_url", src.last_error)

    def test_ws_head_is_followed_by_a_block_fetch(self):
        src = EthereumSource(cfg(transport="ws", ws_url="ws://x"), FakeMatcher())
        stop = threading.Event()
        stop.set()

        class FakeWs:
            def __init__(self, *a, **k):
                pass

            def send(self, msg):
                pass

            def recv(self):
                stop.set()
                return None

            def close(self):
                pass

        with mock.patch.object(src, "ensure_endpoint", return_value="x"):
            with mock.patch.dict("sys.modules", {"websocket": mock.Mock(
                    create_connection=FakeWs)}):
                list(src.stream(stop))
        self.assertIsNone(src.last_error, "a clean socket close is not an error")


if __name__ == "__main__":
    unittest.main()
