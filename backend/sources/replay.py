"""IEEE-CIS timed replay source (real, labeled; replayed at a simulated rate).

Dataset: IEEE-CIS Fraud Detection (Kaggle / IEEE DataPort). Real Vesta credit
card transactions (~590k rows, ~3.5% fraud). Not live - the stream replays rows
in `TransactionDT` order at `events_per_sec`, which is where "real dataset" and
"real-time" meet.

Accepted layouts (auto-discovered under data/):
  * `train_transaction.csv` (+ optional `train_identity.csv`)
  * `final_fraud_dataset.csv`  (pre-merged single-file variant)
Run `python scripts/fetch_ieee.py` for download instructions / one-shot fetch.

Loading is *sliced and lazy*:
  * the CSV is read, sorted by `TransactionDT` and truncated to the configured
    window (`ieee.replay_skip_first` + `stream_slice`) **at load time**, so a
    590k-row file never materialises 590k canonical event dicts;
  * canonical events are then built chunk-by-chunk inside `stream()` and paced
    at `events_per_sec`.  Peak memory is the sliced frame plus one chunk, and
    `POST /api/live/start` returns before the first event exists.
"""
import math
import os
import time

import numpy as np
import pandas as pd

PRODUCT_CD_CODE = {"W": 0, "H": 1, "C": 2, "R": 3, "S": 4}

#: rows canonicalised per vectorised pass while streaming
CHUNK_ROWS = 4096

#: rows per raw CSV read pass; also the granularity at which a stop is honoured
READ_CHUNK_ROWS = 150000

#: columns the feature builder actually reads
_V_COLS = ["V%d" % i for i in range(1, 340)]
_C_COLS = ["C%d" % i for i in range(1, 15)]
_D_COLS = ["D%d" % i for i in range(1, 16)]
_ID_COLS = ["card1", "card2", "card3", "card4", "card5", "card6"]

#: explicit dtypes: skipping pandas' per-column inference is the difference
#: between a ~5 s and a multi-minute load of the 650 MB transaction file
_NUM_F64 = ["TransactionDT", "TransactionAmt", "isFraud", "addr1", "addr2"]
_NUM_F32 = ["dist1", "dist2"] + _C_COLS + _D_COLS + _V_COLS
_STR_COLS = (["TransactionID", "ProductCD", "P_emaildomain", "R_emaildomain",
              "DeviceType", "DeviceInfo"] + _ID_COLS)

#: (counterparty key, source column) - order fixes counterparty ordering
CP_SPECS = [
    ("addr1", "addr1"),
    ("addr2", "addr2"),
    ("p_emaildomain", "P_emaildomain"),
    ("r_emaildomain", "R_emaildomain"),
    ("devicetype", "DeviceType"),
    ("deviceinfo", "DeviceInfo"),
    ("productcd", "ProductCD"),
]


def _find_file(data_dir, *names):
    for name in names:
        full = os.path.join(data_dir, name)
        if os.path.isfile(full):
            return full
    if os.path.isdir(data_dir):
        for root, _dirs, files in os.walk(data_dir):
            for name in names:
                if name in files:
                    return os.path.join(root, name)
    return None


class _Pacer:
    """Deadline-based event pacing.

    `threading.Event.wait(tiny_timeout)` cannot actually sleep less than the OS
    timer granularity (~15 ms on Windows), so calling it once per event pins
    replay throughput at ~60 events/s no matter how high `events_per_sec` is.
    This pacer accumulates the schedule, sleeps only when it is genuinely more
    than `_MIN_SLEEP` behind, and resynchronises afterwards.
    """

    _MIN_SLEEP = 0.0005
    #: give up on catching up and rebase after drifting this far behind
    _MAX_CATCHUP = 0.5

    def __init__(self, events_per_sec, stop_flag):
        self.period = 1.0 / max(events_per_sec, 1e-9)
        self._stop = stop_flag
        self._next = time.time()

    def wait(self, n_events=1):
        # The ideal schedule is kept in absolute time so that the OS's sleep
        # overshoot is absorbed by the next deadline instead of accumulating:
        # `next = time.time()` after every sleep would turn a 10 ms overshoot
        # into a permanent ~17% slow-down of the replay rate.
        self._next += self.period * n_events
        now = time.time()
        delay = self._next - now
        if delay > self._MAX_CATCHUP:
            # genuinely behind (slow consumer, GC pause): rebase, do not
            # try to replay the missed schedule
            self._next = now
            return self._stop.is_set()
        if delay > self._MIN_SLEEP:
            self._stop.wait(min(delay, self._MAX_CATCHUP))
        return self._stop.is_set()


def available_ieee_files(data_dir):
    return {
        "train_transaction": _find_file(data_dir, "train_transaction.csv"),
        "train_identity": _find_file(data_dir, "train_identity.csv"),
        "final_fraud": _find_file(data_dir, "final_fraud_dataset.csv"),
    }


class IeeeReplaySource:
    def __init__(self, config):
        ieee_cfg = config.get("ieee", {})
        self.data_dir = ieee_cfg.get("data_dir") or "data"
        self.events_per_sec = float(config.get("events_per_sec", 20.0))
        self.slice = int(config.get("stream_slice", 100000))
        self.skip_first = int(ieee_cfg.get("replay_skip_first", 0))
        self.sample_fallback = bool(ieee_cfg.get("sample_fallback", True))
        self.sample_size = int(ieee_cfg.get("sample_size", 10000))
        # Pass V1..V339 through to the model (opt-in; see _build_events)
        self.use_v_features = bool(ieee_cfg.get("use_v_features", False))
        self.sample = False
        self.loaded_path = None
        self._frame = None
        self._rows = 0
        self._fraud = 0
        self.total_rows_in_file = 0

    # ------------------------------------------------------------------ load
    def load(self, cancel=None) -> dict:
        """Locate, parse, sort and slice the dataset.

        Returns a status dict.  `cancel` is polled while reading so a stop
        during a slow load does not have to wait for the whole CSV.
        """
        files = available_ieee_files(self.data_dir)
        frame = None
        if files["train_transaction"]:
            frame = self._read_split(files["train_transaction"],
                                     files["train_identity"], cancel)
        elif files["final_fraud"]:
            frame = self._read_merged(files["final_fraud"], cancel)
        elif self.sample_fallback:
            frame = self._build_sample()
        else:
            return {
                "loaded": False,
                "error": "IEEE-CIS file not found.",
                "searched": self.data_dir,
                "hint": "see scripts/fetch_ieee.py and data/README.md",
            }

        if frame is None or (not self.sample and frame.empty):
            return {
                "loaded": False,
                "error": "IEEE-CIS file was empty or unreadable.",
                "searched": self.data_dir,
                "path": self.loaded_path,
                "hint": "see scripts/fetch_ieee.py and data/README.md",
            }

        if not self.sample:
            frame = self._sort_and_slice(frame, cancel)
        self._frame = frame
        self._rows = int(len(frame))
        lab = (frame["isFraud"].to_numpy(dtype=np.float64)
               if "isFraud" in frame.columns else np.zeros(self._rows))
        self._fraud = int(np.count_nonzero(np.nan_to_num(lab) == 1))
        return {
            "loaded": True,
            "sample": self.sample,
            "path": self.loaded_path,
            "rows": self._rows,
            "fraud": self._fraud,
            "skip_first": self.skip_first,
            "slice": self.slice,
            "total_rows_in_file": self.total_rows_in_file,
        }

    def _read_merged(self, path, cancel=None):
        self.loaded_path = path
        return self._read_window(path, cancel)

    def _read_split(self, txn_path, identity_path, cancel=None):
        self.loaded_path = txn_path
        df = self._read_window(txn_path, cancel)
        if df is None or identity_path is None or not os.path.isfile(identity_path):
            return df
        return self._attach_identity(df, identity_path, cancel)

    # ---------------------------------------------------------- CSV ingestion
    @staticmethod
    def _plan_dtypes(header):
        """Map the file's actual header to the dtypes we want, dropping the ~20
        columns the pipeline never reads."""
        present = set(header)
        use, dtypes = [], {}
        for name in _NUM_F64:
            if name in present:
                use.append(name)
                dtypes[name] = "float64"
        for name in _NUM_F32:
            if name in present:
                use.append(name)
                dtypes[name] = "float32"
        for name in _STR_COLS:
            if name in present:
                use.append(name)
                dtypes[name] = "object"
        return use, dtypes

    def _read_window(self, path, cancel=None):
        """Read only as much of the CSV as the replay window needs.

        The IEEE-CIS transaction file is ordered by `TransactionID`, which is
        monotone in `TransactionDT`, so a *prefix* of the file is already in time
        order.  We read whole row-chunks until the window is satisfied, monitor
        monotonicity, and only fall back to reading the entire file if that
        assumption is ever violated.  With `stream_slice = 100k` this touches
        ~110 MB of a 650 MB file instead of all of it.
        """
        try:
            header = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception:
            return None
        use, dtypes = self._plan_dtypes(header)
        if "TransactionDT" not in use:
            return None
        need = None if not self.slice else self.skip_first + self.slice
        # read granularity follows the window: a 6k-row slice must not pay for a
        # 150k-row read pass
        read_chunk = READ_CHUNK_ROWS if not need else max(1000, min(READ_CHUNK_ROWS, need))

        frames = []
        got = 0
        prev_max = -np.inf
        ordered = True
        try:
            reader = pd.read_csv(path, usecols=use, dtype=dtypes,
                                 chunksize=read_chunk)
            try:
                for chunk in reader:
                    if cancel is not None and cancel():
                        return None
                    dt = chunk["TransactionDT"].to_numpy(dtype=np.float64)
                    if len(dt):
                        if dt[0] < prev_max:
                            ordered = False
                        prev_max = max(prev_max, float(np.nanmax(dt)))
                    frames.append(chunk)
                    got += len(chunk)
                    if need is not None and got >= need:
                        break
            finally:
                # breaking out of the chunk loop leaves the CSV handle open
                reader.close()
        except Exception:
            return None
        if cancel is not None and cancel():
            return None

        if not ordered:
            # time order is not the file order after all: correctness first
            frames = []
            try:
                for chunk in pd.read_csv(path, usecols=use, dtype=dtypes,
                                         chunksize=read_chunk):
                    if cancel is not None and cancel():
                        return None
                    frames.append(chunk)
            except Exception:
                return None
            self.total_rows_in_file = int(sum(len(f) for f in frames))
        else:
            self.total_rows_in_file = got
            if need is not None:
                self.total_rows_in_file = max(got, need)

        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return frame

    def _attach_identity(self, df, identity_path, cancel=None):
        """Left-join the card identity columns, reading the identity file in
        chunks so only the rows present in the window are kept.

        Returns the joined frame (or `df` unchanged when the identity file has
        nothing to contribute).
        """
        if "TransactionID" not in df.columns:
            return df
        want = set(df["TransactionID"].tolist())
        if not want:
            return df
        try:
            header = pd.read_csv(identity_path, nrows=0).columns.tolist()
        except Exception:
            return df
        use = [c for c in ["TransactionID"] + _ID_COLS if c in header]
        if len(use) < 2:
            return df
        # The two files may represent TransactionID differently (numeric vs
        # string), so read the identity side in the *window's* representation:
        # otherwise every join key mismatches and the identity features are
        # silently absent for the whole window.
        numeric = df["TransactionID"].dtype.kind in "fiu"
        id_dtype = "float64" if numeric else str
        dtypes = {c: (id_dtype if c == "TransactionID" else "object") for c in use}
        want_key = want if numeric else {str(v) for v in want}
        parts = []
        try:
            reader = pd.read_csv(identity_path, usecols=use, dtype=dtypes,
                                 chunksize=READ_CHUNK_ROWS)
            try:
                for chunk in reader:
                    if cancel is not None and cancel():
                        return df
                    sel = chunk[chunk["TransactionID"].isin(want_key)]
                    if not sel.empty:
                        parts.append(sel)
            finally:
                reader.close()
        except Exception:
            return df
        if not parts:
            return df
        idf = pd.concat(parts, ignore_index=True).drop_duplicates("TransactionID")
        if not numeric:
            df["TransactionID"] = df["TransactionID"].astype(str)
            idf["TransactionID"] = idf["TransactionID"].astype(str)
        for col in use:
            # keep the join key on the window side; drop the identity copy
            if col != "TransactionID" and col in df.columns:
                df.drop(columns=[col], inplace=True)
        return df.merge(idf, on="TransactionID", how="left")

    def _sort_and_slice(self, df, cancel=None):
        """Keep the configured replay window, ordered by transaction time.

        Slicing *after* the sort (not before) is what makes the window
        meaningful: the stream is always the first N rows in time order, so a
        longer session can raise `stream_slice` instead of replaying a
        different, non-deterministic subset.
        """
        if "TransactionDT" in df.columns:
            df = df.sort_values("TransactionDT", kind="mergesort")
        start = max(0, self.skip_first)
        end = None if not self.slice else start + self.slice
        if cancel is not None and cancel():
            return df.iloc[0:0]
        return df.iloc[start:end].reset_index(drop=True)

    # ------------------------------------------------- deterministic sample
    def _build_sample(self):
        """Fallback used ONLY when no real IEEE-CIS CSV is present.

        A deterministic labeled synthetic frame with the same column names, so
        every stage of the pipeline (training, adaptive weights, the review
        loop, the graph) can be exercised. Real CSVs placed in data/ always
        take precedence over this.
        """
        rng = np.random.default_rng(42)
        n = self.sample_size
        card_pool = [str(int(rng.integers(400000, 500000))) for _ in range(600)]
        domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
                   "aol.com", "proton.me", "mail.com"]
        devices = ["desktop", "mobile", "pos", "kiosk"]
        t0 = 800000000.0

        isfraud = rng.random(n) < 0.03
        base = np.exp(rng.normal(4.0, 0.85, size=n))
        amount = np.where(isfraud, base * (2.8 + rng.random(n) * 1.6), base)
        dt = (t0 + np.arange(n) * 15.0 + rng.normal(0.0, 20.0, size=n)
              + np.where(isfraud, rng.random(n) * 3600.0, 0.0))

        card = np.where(
            rng.random(n) < 0.6,
            [card_pool[int(v)] for v in rng.integers(0, len(card_pool), size=n)],
            [str(int(v)) for v in rng.integers(400000, 600000, size=n)])
        addr1 = np.array(["addr1:%d" % v
                      for v in rng.integers(100, 999, size=n)])
        p_email = np.where(rng.random(n) < 0.8,
                           ["p_email:" + domains[int(v)]
                            for v in rng.integers(0, len(domains), size=n)], "")
        r_email = np.where(rng.random(n) < 0.4,
                           ["r_email:" + domains[int(v)]
                            for v in rng.integers(0, len(domains), size=n)], "")
        device = np.where(rng.random(n) < 0.4,
                          ["device:" + devices[int(v)]
                           for v in rng.integers(0, len(devices), size=n)], "")

        cols = _C_COLS + _D_COLS + _V_COLS
        data = {
            "TransactionID": np.arange(1, n + 1),
            "TransactionDT": dt,
            "TransactionAmt": amount,
            "isFraud": isfraud.astype(np.int64),
            "dist1": np.abs(rng.normal(0.0, 40.0, size=n)),
            "dist2": np.abs(rng.normal(0.0, 60.0, size=n)),
            "c_sum": rng.normal(0.0, 5.0, size=n),
            "c_mean": rng.normal(0.0, 0.5, size=n),
            "c_max": np.abs(rng.normal(0.0, 3.0, size=n)),
            "d_sum": rng.normal(0.0, 4.0, size=n),
            "num_nan_v": rng.integers(0, 45, size=n).astype(float),
            "ProductCD": rng.integers(0, 6, size=n).astype(float),
            "P_emaildomain": p_email,
            "R_emaildomain": r_email,
            "DeviceType": device,
            "card1": card,
        }
        for c in cols:
            data.setdefault(c, 0.0)
        frame = pd.DataFrame(data)
        # keep the sample inside the same slice contract as a real file
        frame = self._sort_and_slice(frame)
        self.sample = True
        self.loaded_path = "__sample__"
        self.total_rows_in_file = int(n)
        return frame

    # --------------------------------------------- vectorised row -> event build
    def _build_events(self, df):
        """Build canonical events for one chunk of the sliced frame.

        Vectorised numpy/pandas extraction (a per-cell `Series.get` loop over
        590k rows was the startup bottleneck) and it is byte-for-byte
        equivalent to the per-row reference implementation.
        """
        n = len(df)
        if n == 0:
            return []

        def col(name, default=0.0):
            if name in df.columns:
                return df[name].to_numpy(dtype=np.float64)
            return np.full(n, default, dtype=np.float64)

        ts_arr = col("TransactionDT")
        amt_arr = np.nan_to_num(col("TransactionAmt"))
        lab_arr = np.nan_to_num(col("isFraud"))

        hour_arr = (ts_arr % 86400.0) // 3600.0
        dow_arr = (ts_arr // 86400.0) % 7.0
        log_amt_arr = np.log1p(amt_arr)

        dist1 = np.nan_to_num(col("dist1"))
        dist2 = np.nan_to_num(col("dist2"))
        c_sum, c_mean, c_max = col("c_sum"), col("c_mean"), col("c_max")
        d_sum = col("d_sum")
        nnan_v = col("num_nan_v")

        # Optional pass-through of the anonymised V1..V339 columns.
        #
        # The single-app rebuild kept only `num_nan_v` (a count of missing
        # values) and threw the other ~339 real features away, which is the
        # main reason the reported holdout ROC-AUC sat at 0.73 where published
        # work on the same dataset reaches ~0.95. They are opt-in because they
        # widen the vector from 26 to ~365 columns, which costs memory and
        # training time; see `ieee.use_v_features` in backend/config.py.
        v_arrays = {}
        if self.use_v_features:
            present = [c for c in _V_COLS if c in df.columns]
            for c in present:
                v_arrays[c] = np.nan_to_num(df[c].to_numpy(dtype=np.float64))

        if "ProductCD" in df.columns:
            raw_pc = df["ProductCD"].to_numpy(dtype=object)
        else:
            raw_pc = np.array([None] * n, dtype=object)
        product_codes = np.empty(n, dtype=object)
        for k in range(n):
            v = raw_pc[k]
            product_codes[k] = ("" if v is None
                                or (isinstance(v, float) and math.isnan(v))
                                else str(v).strip().upper())
        pcode = np.array([PRODUCT_CD_CODE.get(p, 5.0) for p in product_codes],
                         dtype=np.float64)

        has_email = (~df["P_emaildomain"].isna()).to_numpy() \
            if "P_emaildomain" in df.columns else np.zeros(n, dtype=bool)
        has_r_email = (~df["R_emaildomain"].isna()).to_numpy() \
            if "R_emaildomain" in df.columns else np.zeros(n, dtype=bool)

        def _strcol(name):
            if name not in df.columns:
                return np.array([""] * n, dtype=object)
            ser = df[name]
            out = ser.to_numpy(dtype=object)
            fixed = np.empty(n, dtype=object)
            for k in range(n):
                v = out[k]
                fixed[k] = "" if v is None or (isinstance(v, float) and math.isnan(v)) \
                    else str(v).strip()
            return fixed

        tid_col = _strcol("TransactionID")
        card_cols = {c: _strcol(c) for c in _ID_COLS}
        cp_arrays = [(key, _strcol(src)) for key, src in CP_SPECS
                     if src in df.columns]
        addr1 = df["addr1"] if "addr1" in df.columns else None
        addr2 = df["addr2"] if "addr2" in df.columns else None
        p_email_ser = df["P_emaildomain"] if "P_emaildomain" in df.columns else None
        r_email_ser = df["R_emaildomain"] if "R_emaildomain" in df.columns else None
        dev_type = df["DeviceType"] if "DeviceType" in df.columns else None
        dev_info = df["DeviceInfo"] if "DeviceInfo" in df.columns else None

        events = []
        append = events.append
        for i in range(n):
            ts = float(ts_arr[i])
            amt = float(amt_arr[i])
            label = int(lab_arr[i])
            tid = tid_col[i]
            c1, c2, c3 = (card_cols["card1"][i], card_cols["card2"][i],
                          card_cols["card3"][i])
            c5, c6 = card_cols["card5"][i], card_cols["card6"][i]
            account_id = "ieee:%s.%s.%s.%s" % (c1, c2, c5, c6)
            if account_id == "ieee:...":
                account_id = "ieee:t%s" % tid

            counterparties = []
            for _key, arr in cp_arrays:
                v = arr[i]
                if v and v.lower() != "nan":
                    counterparties.append("%s:%s" % (_key, v))

            product_code = str(product_codes[i]).upper()
            native = {
                "amt": amt,
                "log_amt": float(log_amt_arr[i]),
                "hour": float(hour_arr[i]),
                "dow": float(dow_arr[i]),
                "dist1": float(dist1[i]),
                "dist2": float(dist2[i]),
                "c_sum": float(c_sum[i]),
                "c_mean": float(c_mean[i]),
                "c_max": float(c_max[i]),
                "d_sum": float(d_sum[i]),
                "num_nan_v": float(nnan_v[i]),
                "product_cd_code": float(pcode[i]),
                "has_email": 1.0 if has_email[i] else 0.0,
                "has_r_email": 1.0 if has_r_email[i] else 0.0,
            }
            if v_arrays:
                for c, arr in v_arrays.items():
                    native[c] = float(arr[i])
            raw = {
                "transact_id": str(tid),
                "card1": c1, "card2": c2, "card3": c3,
                "card4": card_cols["card4"][i], "card5": c5, "card6": c6,
                "addr1": _rawv(addr1, i),
                "addr2": _rawv(addr2, i),
                "p_email": _rawv(p_email_ser, i),
                "r_email": _rawv(r_email_ser, i),
                "device_type": _rawv(dev_type, i),
                "product_cd": product_code,
            }
            append({
                "event_id": "ieee-%s" % tid,
                "source": "ieee",
                "ts": ts,
                "account_id": account_id,
                "counterparties": counterparties,
                "label": label,
                "amount": amt,
                "nat": native,
                "raw": raw,
            })
        return events

    # -------------------------------------------------------------- streaming
    @property
    def frame(self):
        """The loaded, time-ordered replay window, or None before `load()`."""
        return self._frame

    def stream(self, stop_flag, cancel=None):
        """Yield canonical events in time order at `events_per_sec`.

        Events are canonicalised one chunk at a time, so the cost is spread
        across the session instead of paid up front.  `cancel` is an optional
        callable that supersedes a mere stop_flag (session generation change).
        Yields None once the window is exhausted, then returns: the caller
        decides what to do with an exhausted window.
        """
        if self._frame is None:
            yield None
            return
        dead = (lambda: stop_flag.is_set()
                or (cancel is not None and cancel()))
        pacer = _Pacer(self.events_per_sec, stop_flag)
        for start in range(0, len(self._frame), CHUNK_ROWS):
            if dead():
                return
            events = self._build_events(self._frame.iloc[start:start + CHUNK_ROWS])
            for pos, ev in enumerate(events):
                if dead():
                    return
                yield ev
                if pos + 1 < len(events) and pacer.wait(1):
                    return
        yield None


def _rawv(ser, i):
    """Raw cell value for a column (mirrors _get(..., '') on the cell)."""
    if ser is None:
        return ""
    v = ser.iloc[i]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return v
