"""Ethereum live stream source.

Two transports:
  * `rpc-poll` (default, keyless): polls the latest block over plain JSON-RPC
    HTTP and emits each transaction as a canonical event.  The source is
    resilient: `chain.rpc_url` is used first, then every URL in
    `chain.rpc_fallback_urls` in order; the first endpoint that answers is
    locked in as the active one, and if it later fails repeatedly the source
    re-probes the list and switches automatically.
  * `ws` : subscribes to `eth_subscribe newHeads` over WebSocket (websocket-client)
    and then pulls each announced block, tracking announced heads deduplicated.

Endpoint probing happens on the **producer thread**, never in `__init__`: a cold
probe chain costs several seconds, and `POST /api/live/start` must return
immediately.

Ground truth: raw events are unlabeled; the ScamAddressMatcher (labels.py)
flags events touching known-bad addresses. Everything else flows into the
human-review queue.
"""
import math
import threading
import time

import requests

#: a plausible intra-block ordering offset, in seconds, so that the streaming
#: velocity features (`hours_since_last`, `events_60s`) are not degenerate.
#: Every transaction in a block otherwise shares the block timestamp.
TX_TIME_SPREAD = 0.25
TX_TIME_MAX = 12.0

#: a block longer than this on the reference chain is implausible; used only to
#: bound the synthetic intra-block offset.
MAX_PLAUSIBLE_BLOCK_SPAN = 30.0


def _hex_int(v):
    return int(v, 16)


def _rpc_call(url, method, params, timeout=15):
    resp = requests.post(
        url,
        json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload.get("result")


class EthereumSource:
    def __init__(self, config, scam_matcher):
        chain_cfg = config.get("chain", {})
        self.transport = (chain_cfg.get("transport") or "rpc-poll").lower()
        self.rpc_url = (chain_cfg.get("rpc_url") or "https://eth.drpc.org").strip()
        self.rpc_fallbacks = [
            u.strip() for u in (chain_cfg.get("rpc_fallback_urls") or [])
            if isinstance(u, str) and u.strip()
        ]
        self.ws_url = chain_cfg.get("ws_url") or ""
        self.poll_interval = float(chain_cfg.get("poll_interval_seconds", 5))
        self.probe_timeout = float(chain_cfg.get("probe_timeout_seconds", 6))
        self.max_txs = int(chain_cfg.get("max_txs_per_block", 300))
        self.scam_matcher = scam_matcher

        self._seen_blocks = set()
        self._seen_order = []          # insertion order, so eviction is FIFO
        self._max_seen = 200
        self.last_block_number = None
        self.last_block_ts = None
        self.last_error = None
        # start on the configured primary; `ensure_endpoint()` upgrades it to a
        # known-good URL from the producer thread
        self.active_rpc_url = self.rpc_url
        self.endpoint_probed = False
        self._consecutive_errors = 0
        self._failover_threshold = 3
        self._last_probe = 0.0
        self._probe_lock = threading.Lock()

    # ------------------------------------------------------- endpoint failover
    def _candidates(self):
        seen = set()
        ordered = [self.rpc_url] + self.rpc_fallbacks + [self.active_rpc_url]
        for u in ordered:
            if not u:
                continue
            if u not in seen:
                seen.add(u)
                yield u

    def ensure_endpoint(self, force=False):
        """Probe the candidate list and lock in the first that answers.

        Idempotent: after a successful probe it is a no-op unless `force`.
        Safe to call from the producer thread on every poll.
        """
        with self._probe_lock:
            if self.endpoint_probed and not force:
                return self.active_rpc_url
            if not force and (time.time() - self._last_probe) < 30.0:
                return self.active_rpc_url
            self._last_probe = time.time()
            for url in self._candidates():
                try:
                    result = _rpc_call(url, "eth_blockNumber", [],
                                       timeout=self.probe_timeout)
                    if result:
                        self.active_rpc_url = url
                        self.rpc_url = url
                        self._consecutive_errors = 0
                        self.last_error = None
                        self.endpoint_probed = True
                        return url
                except Exception as exc:
                    self.last_error = f"{url}: {exc}"
            self.endpoint_probed = False
            return self.active_rpc_url

    def _rpc(self, method, params):
        """Call the active RPC; on repeated failures re-probe and fail over."""
        try:
            result = _rpc_call(self.active_rpc_url or self.rpc_url, method, params)
            self._consecutive_errors = 0
            return result
        except Exception as exc:
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._failover_threshold:
                self.ensure_endpoint(force=True)
                if self.active_rpc_url:
                    result = _rpc_call(self.active_rpc_url, method, params)
                    self._consecutive_errors = 0
                    return result
            raise exc

    def fetch_block(self, block_tag="latest"):
        blk = self._rpc("eth_getBlockByNumber", [block_tag, True])
        if not blk:
            return None
        number = _hex_int(blk["number"])
        if number in self._seen_blocks:
            return {"seen": True, "number": number}
        self._seen_blocks.add(number)
        self._seen_order.append(number)
        while len(self._seen_order) > self._max_seen:
            self._seen_blocks.discard(self._seen_order.pop(0))
        return {
            "seen": False,
            "number": number,
            "hash": blk.get("hash"),
            "timestamp": _hex_int(blk.get("timestamp", "0x0")),
            "transactions": blk.get("transactions") or [],
        }

    # ------------------------------------------------------------ event build
    def _event_from_tx(self, tx, block_number, block_ts, tx_index, tx_count=1):
        from backend.events import normalize_event

        fr = (tx.get("from") or "").lower()
        to = (tx.get("to") or "").lower()
        value_eth = _hex_int(tx.get("value") or "0x0") / 1e18
        gas_gwei = _hex_int(tx.get("gasPrice") or "0x0") / 1e9

        counterparties = []
        if to:
            counterparties.append(to)
        account_id = fr if fr else f"0xcontract-create-{tx.get('hash', '')}"

        # Spread transactions inside the block over a plausible sub-block
        # interval.  Using the bare block timestamp for every transaction would
        # make the rolling-velocity features (the multi-round signal this whole
        # project is built on) constant within a block.
        if tx_count > 1:
            offset = min(MAX_PLAUSIBLE_BLOCK_SPAN,
                         tx_index * TX_TIME_SPREAD)
        else:
            offset = 0.0
        ts = float(block_ts) + offset

        native = {
            "value_eth": value_eth,
            "log_value_eth": math.log1p(value_eth),
            "gas_gwei": gas_gwei,
            "tx_index": float(tx_index),
        }
        ev = normalize_event(
            source="chain",
            ts=ts,
            account_id=account_id,
            counterparties=counterparties,
            label=None, amount=value_eth,
            native=native,
            raw={"block": block_number, "tx_hash": tx.get("hash", "")},
            event_id=f"chain-{block_number}-{tx_index}",
        )
        scam = self.scam_matcher.label_for(ev) if self.scam_matcher else None
        if scam == 1:
            ev["label"] = 1
            # `scam_match` is an ORACLE field (see features.ORACLE_COLUMNS): it
            # is kept on the event for the dashboard but never given to a model.
            ev["nat"]["scam_match"] = 1.0
        return ev

    def _events_from_block(self, blk):
        txs = (blk.get("transactions") or [])[: self.max_txs]
        n = len(txs)
        return [
            self._event_from_tx(tx, blk["number"], blk["timestamp"], i, n)
            for i, tx in enumerate(txs)
        ]

    # -------------------------------------------------------------- streaming
    def stream(self, stop_flag):
        """Generator: polls/subscribes and yields canonical chain events.
        Yields None on every idle poll so callers can act on liveness."""
        self.ensure_endpoint()
        if self.transport == "ws":
            yield from self._stream_ws(stop_flag)
        else:
            yield from self._stream_poll(stop_flag)

    def _record_block(self, blk):
        self.last_block_number = blk["number"]
        self.last_block_ts = blk["timestamp"]

    def _stream_poll(self, stop_flag):
        while not stop_flag.is_set():
            try:
                blk = self.fetch_block("latest")
                self.last_error = None
                if blk and not blk.get("seen"):
                    self._record_block(blk)
                    for ev in self._events_from_block(blk):
                        if stop_flag.is_set():
                            return
                        yield ev
            except Exception as exc:  # network hiccup: keep the loop alive
                self.last_error = str(exc)
                if stop_flag.wait(min(self.poll_interval, 2.0)):
                    return
            if stop_flag.wait(self.poll_interval):
                return
            yield None

    def _stream_ws(self, stop_flag):
        if not self.ws_url:
            self.last_error = "transport=ws but chain.ws_url is empty"
            stop_flag.wait()
            return
        import json

        import websocket

        while not stop_flag.is_set():
            try:
                ws = websocket.create_connection(self.ws_url, timeout=15)
                ws.send(json.dumps({"id": 1, "method": "eth_subscribe",
                                    "params": ["newHeads"]}))
                while not stop_flag.is_set():
                    msg = ws.recv()
                    if msg is None:
                        break
                    try:
                        parsed = json.loads(msg)
                    except Exception:
                        continue
                    if parsed.get("method") != "eth_subscribe":
                        continue
                    result = (parsed.get("params") or {}).get("result") or {}
                    head_number = _hex_int(result.get("number", "0x0"))
                    blk = self.fetch_block(hex(head_number))
                    if blk and not blk.get("seen"):
                        self._record_block(blk)
                        for ev in self._events_from_block(blk):
                            if stop_flag.is_set():
                                return
                            yield ev
                ws.close()
            except Exception as exc:
                self.last_error = str(exc)
                if stop_flag.wait(min(self.poll_interval, 5)):
                    return
            yield None
