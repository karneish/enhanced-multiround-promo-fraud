"""Webhook producer: external systems POST canonical-adjacent events into the
detector.  The pipeline treats these exactly like any other source."""
import time
import uuid

from backend.events import normalize_event


def parse_payload(payload, source="webhook"):
    """Accept either a full canonical event or a light payload:

        {"account_id": ..., "amount": ..., "counterparties": [...],
         "label": 0|1|null, "native": {...}}

    Returns a canonical event dict.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    if "source" in payload and "event_id" in payload:
        from backend.events import validate_event
        return validate_event(payload)

    account_id = payload.get("account_id") or payload.get("account") or payload.get("id")
    if account_id is None:
        raise ValueError("webhook payload requires an account_id")

    native = dict(payload.get("native") or {})
    amt = payload.get("amount") or 0.0
    native.setdefault("txn_amt", float(amt))
    native.setdefault("n_attrs", float(len(payload.get("counterparties") or [])))
    native.setdefault("label_hint", 1.0 if payload.get("label") else 0.0)

    from backend.events import validate_event
    # validate the *normalised* event: a caller-supplied label outside {0, 1}
    # must be a 400, not a graph KeyError several layers downstream
    return validate_event(normalize_event(
        source=source,
        ts=float(payload.get("ts", payload.get("event_time", time.time()))),
        account_id=str(account_id),
        counterparties=payload.get("counterparties") or [],
        label=payload.get("label"),
        amount=amt,
        native=native,
        raw={k: v for k, v in payload.items() if k not in ("native",)},
        event_id="wh-" + uuid.uuid4().hex[:12],
    ))
