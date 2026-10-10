"""Configuration for the real-time adaptive 3-model detector.

Sources of overrides (lowest to highest precedence):
  1. built-in defaults below
  2. a `config.local.json` file in the repo root (optional)
  3. per-request /api/live/start body (highest)
"""
import copy
import json
import os

DEFAULT_CONFIG = {
    "port": 5051,
    "host": "127.0.0.1",

    # live_mode:  replay-only | chain-only | handoff | concurrent
    #   - replay-only:  IEEE-CIS timed replay only
    #   - chain-only:   Ethereum live stream only
    #   - handoff:      replay first, then automatically switch to the chain stream
    #   - concurrent:   replay + chain together (source-tagged)
    "live_mode": "concurrent",

    # Streaming speed / volume
    "events_per_sec": 20.0,          # replay records pushed per second
    "stream_slice": 100000,          # max replay rows read/emitted after the skip offset
    "retain_event_records": 200,     # ring buffer shared with the dashboard feed

    # Scoring batch (real-time throughput).  The consumer drains the queue into
    # micro-batches of at most `score_batch_max` rows, waiting at most
    # `score_batch_window_ms` to fill one.  Per-row ExtraTrees prediction costs
    # ~135 ms of fixed overhead, so batching is worth ~1000x throughput.
    "score_batch_max": 256,
    "score_batch_window_ms": 25,

    # Streaming-learning cadence (per source)
    "min_fit_labels": 120,           # labeled events needed before the first model fit
    "retrain_interval_events": 500,  # relabeled events between warm-start refits
    "retrain_interval_seconds": 60,  # or the wall-clock guarantee between refits
    "buffer_max_labels": 3000,       # rolling labeled window kept per source

    # Model hyper-parameters (see MODELS_REFERENCE.md §3)
    "xgb_rounds": 200,
    "xgb_extra_rounds": 50,
    "hist_max_iter": 200,
    "hist_extra_iter": 30,
    "et_n_estimators": 200,

    # Allow / Review / Block bands (probability thresholds)
    "decision_thresholds": {"review": 0.30, "block": 0.70},

    # Human-review budget (gates how many unlabeled Review decisions may be
    # queued for a human/LLM to label; see `review` below and the /api/review
    # endpoints).
    "budget_pos": 10,
    "budget_neg": 200,
    "review": {
        "queue_max": 200,            # pending candidates held for a reviewer
        "expire_seconds": 3600,      # drop candidates nobody labelled in an hour
    },

    # Node graph of everything the stream has seen (bounded for the UI)
    "graph_node_cap": 2000,

    "seed": 42,

    "ieee": {
        "data_dir": "data",
        "train_transaction": "",     # auto-discovered if empty
        "train_identity": "",        #
        "final_fraud_file": "",      # optional pre-merged single-file variant
        "replay_skip_first": 0,      # warm-up rows to skip at the start of the replay
        "sample_fallback": True,     # deterministic labeled sample when no real CSVs
        "sample_size": 10000,        # rows generated for the sample fallback
        # Pass the anonymised V1..V339 columns through to the model instead of
        # keeping only a missing-value count. Off by default because it widens
        # the vector from 26 to ~365 columns; scripts/evaluate.py --v-features
        # measures whether it is worth it.
        "use_v_features": False,
    },

    "chain": {
        "transport": "rpc-poll",     # rpc-poll (keyless, HTTP) | ws (WebSocket subscribe)
        # Ordered by measured cold-start latency against the free public
        # gateways.  `ethereum-rpc.publicnode.com` and `cloudflare-eth.com` are
        # last: the first times out, the second answers with
        # {"code": -32046}.  Verified working: drpc, blastapi, flashbots, 1rpc.
        "rpc_url": "https://eth.drpc.org",
        "rpc_fallback_urls": [
            "https://eth-mainnet.public.blastapi.io",
            "https://rpc.flashbots.net",
            "https://1rpc.io/eth",
            "https://cloudflare-eth.com",
            "https://ethereum-rpc.publicnode.com",
        ],
        "ws_url": "",
        "poll_interval_seconds": 5,
        "probe_timeout_seconds": 6,
        "max_txs_per_block": 300,
        "scam_list_path": "data/scam_addresses.csv",
    },
}

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.local.json")


def deep_merge(base, override):
    out = copy.deepcopy(base)
    if not isinstance(override, dict):
        return out
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    cfg = deep_merge(DEFAULT_CONFIG, {})
    try:
        if os.path.isfile(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
                cfg = deep_merge(cfg, json.load(fh))
    except Exception:
        pass
    return cfg