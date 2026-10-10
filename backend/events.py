"""Canonical event schema shared by every source (replay, chain, webhook).

A canonical event is a plain dict:

    {
        "event_id":        "ieee-...",
        "source":          "ieee" | "chain" | "webhook",
        "ts":              float(epoch seconds),
        "account_id":      str,   unique paying/acting identity
        "counterparties":  [str], identities this event shares traits with
        "label":           0 | 1 | None,   ground truth when known
        "amount":          float,
        "nat":             { <source-native numeric features> },
        "raw":             { original fields for display },
    }
"""
import uuid

SOURCES = ("ieee", "chain", "webhook", "adver")

REQUIRED = ("event_id", "source", "ts", "account_id")


def normalize_event(*, source, ts, account_id,
                    counterparties=None, label=None, amount=None,
                    native=None, raw=None, event_id=None):
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source!r}")
    ev = {
        "event_id": event_id or f"{source}-{uuid.uuid4().hex[:12]}",
        "source": source,
        "ts": float(ts),
        "account_id": str(account_id),
        "counterparties": [str(c) for c in (counterparties or [])],
        "label": None if label is None else int(label),
        "amount": None if amount is None else float(amount),
        "nat": dict(native or {}),
        "raw": dict(raw or {}),
    }
    return ev


def validate_event(ev) -> dict:
    for key in REQUIRED:
        if key not in ev:
            raise ValueError(f"event missing required field: {key}")
    if not isinstance(ev.get("counterparties"), list):
        ev["counterparties"] = []
    if ev.get("label") not in (None, 0, 1):
        raise ValueError(f"label must be None, 0 or 1, got: {ev.get('label')!r}")
    return ev


def event_schema_doc() -> dict:
    """Human/UI readable description of the canonical event schema."""
    return {
        "required": list(REQUIRED),
        "optional": ["counterparties", "label", "amount", "nat", "raw"],
        "sources": list(SOURCES),
        "label": "0 = genuine, 1 = fraud, null = unknown (chain/webhook)",  # noqa: E501
        "nat": "dictionary of source-native numeric features (see real_features.py)",  # noqa: E501
    }