"""Label bridge for live (unlabeled) streams.

IEEE-CIS replay events carry real `isFraud` labels natively.
Ethereum events are unlabeled by default; a known-scam address list supplies
ground truth for the small tail of activity that touches a known-bad address.

Expected CSV layout (data/scam_addresses.csv):
    Address,Label,Type[,Source,...]
Columns are auto-detected (case-insensitive): address may be `Address`,
`address`, `addr`, `hash`, `account`; the label column is usually `Label` or
`label`.  Rows without a parseable address are ignored.
"""
import os

import pandas as pd


class ScamAddressMatcher:
    _ADDR_CANDIDATES = ("address", "addr", "hash", "wallet", "account", "id")
    _LABEL_CANDIDATES = ("label", "is_fraud", "type", "category")

    def __init__(self, path):
        self.path = path
        self._addresses = set()
        self._available = False
        self._count = 0
        self.load()

    def load(self):
        self._addresses = set()
        self._available = False
        self._count = 0
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            df = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        except Exception:
            return
        if df.empty:
            return
        cols = {c.lower().strip(): c for c in df.columns}
        addr_col = None
        for cand in self._ADDR_CANDIDATES:
            if cand in cols:
                addr_col = cols[cand]
                break
        if addr_col is None:
            addr_col = df.columns[0]
        raw = df[addr_col].astype(str).str.strip().str.lower()
        raw = raw[raw.str.startswith("0x") | raw.str.startswith(("bc1", "1", "3"))]
        self._addresses = set(raw.tolist())
        self._count = len(self._addresses)
        self._available = self._count > 0

    @property
    def available(self):
        return self._available

    @property
    def count(self):
        return self._count

    def is_scam(self, identifier):
        ident = str(identifier).strip().lower()
        return ident in self._addresses

    def label_for(self, ev):
        """Return 1 when the event's account or any counterparty is a known-scam
        address, otherwise None (unlabeled, falls back to the review loop)."""
        ids = [ev.get("account_id")] + list(ev.get("counterparties") or [])
        for ident in ids:
            if ident and self.is_scam(ident):
                return 1
        return None