"""The adaptive 3-model ensemble detector.

Models (the project's canonical trio, see MODELS_REFERENCE.md):
  * XGBoost                - gradient-boosted trees, warm-started by resuming
                             the underlying booster (`xgb_model=`) on each refit
  * HistGradientBoosting   - histogram-based GBDT.  Grown by raising `max_iter`
                             each refit.  It is re-created rather than mutated
                             in place: sklearn's `warm_start` only reuses trees
                             when `fit` is called on the *same* estimator, and
                             an in-place refit would race the scoring path.
  * ExtraTrees             - maximally-randomized bagging; refit on the rolling
                             labeled buffer (no warm-start in sklearn)

Each source keeps its own per-source ensemble + rolling labeled window.
Refits run on a dedicated single-worker thread and are swapped in atomically,
so scoring never blocks on training.

Interface used by livesim.py:
    ens = ThreeModelEnsemble(source, columns, config)
    ens.add_labeled(X_row: numpy array, y: int) -> None
    ens.fit_needed(now_ts) -> bool
    ens.request_refit() -> bool            # async; returns False if busy
    ens.fit() -> dict                      # synchronous (tests / offline eval)
    prob = ens.predict_proba(X_row) -> float
    probs = ens.predict_proba_batch(X) -> list[float]
    decision = ens.decide(prob, thresholds) -> str
"""
import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MODEL_NAMES = ("XGBoost", "HistGradientBoosting", "ExtraTrees")

#: share of the window held back to score the models on
HOLDOUT_FRACTION = 0.2
#: a holdout slice needs at least this many rows of each class to score F1
MIN_HOLDOUT_PER_CLASS = 2
#: A collapsed model keeps at most this fraction of the *best* model's weight.
#: Relative, not absolute: an absolute floor (f1 + 0.01) dominates the signal
#: once F1s sit in the realistic 0.1-0.2 band and silently reduces the blend to
#: uniform averaging, and its effect depends on the arbitrary F1 scale.
WEIGHT_FLOOR_FRACTION = 0.05

_warmup_started = False
_warmup_lock = threading.Lock()


def _warm_imports():
    """Import the model libraries once, ahead of any fit.

    `import xgboost` costs several seconds on a cold Windows filesystem (large
    DLL + OpenMP init).  Paying it lazily inside the first refit inflates the
    first-fit latency and, on a machine where the import itself is the bulk of
    the fit, makes a warm refit look far slower than it is.  It is done on a
    throwaway daemon thread - never on the refit executor, so a real refit can
    never queue behind it.
    """
    global _warmup_started
    with _warmup_lock:
        if _warmup_started:
            return
        _warmup_started = True

    def _load():
        for mod in ("xgboost", "sklearn.ensemble"):
            try:
                importlib.import_module(mod)
            except Exception:
                pass

    t = threading.Thread(target=_load, name="model-import-warmup", daemon=True)
    t.start()


def _f1_precision_recall(y_true, y_pred):
    tp = int(np.count_nonzero((y_pred == 1) & (y_true == 1)))
    tn = int(np.count_nonzero((y_pred == 0) & (y_true == 0)))
    fp = int(np.count_nonzero((y_pred == 1) & (y_true == 0)))
    fn = int(np.count_nonzero((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}



def _class_weight(cfg):
    """Resolve the ensemble members' `class_weight`.

    ``"balanced"`` (the historical default) maximises recall at the cost of
    precision. A numeric weight below 1 trades some recall for precision, which
    is usually the better trade for F1 on a rare-class problem. Set
    ``class_weight: null`` in the config to disable re-weighting entirely.
    """
    w = cfg.get("class_weight", "balanced")
    if w is None or (isinstance(w, str) and w.lower() in ("none", "off")):
        return None
    return w


def _class_weight_numeric(cfg):
    """Numeric form for estimators that reject a per-class dict."""
    w = cfg.get("class_weight", "balanced")
    n = cfg.get("class_weight_numeric")
    if n is not None:
        return float(n)
    if isinstance(w, dict):
        return float(w.get(1, 1.0))
    if w is None or (isinstance(w, str) and w.lower() in ("none", "off")):
        return None
    return w


class ThreeModelEnsemble:
    def __init__(self, source, columns, config):
        self.source = source
        self.columns = list(columns)
        self.cfg = config
        self._seed = int(config.get("seed", 42))

        # model handles, replaced atomically on every refit
        self._xgb_booster = None
        self._xgb_rounds_done = 0
        self._hist = None
        self._hist_max_iter = 0
        self._et = None

        # Rolling labeled window (lists; converted to arrays on fit)
        self._X = []
        self._y = []
        self._buffered = 0          # labels added since the last fit

        self._fitted = False
        self._weights = None        # {name: weight}
        self._last_fit_at = 0.0
        self._metrics = {}          # {name: {f1, precision, recall, ...}}
        self._ensemble_metrics = {}
        self._holdout_mode = None   # "temporal" | "stratified" | None
        self._holdout_degenerate = False

        self._refit_in_progress = False
        self._last_refit_seconds = None
        self._refit_count = 0
        self._refit_errors = 0

        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix=f"refit-{source}")
        self._last_fit_breakdown = {}
        _warm_imports()

    def shutdown(self):
        """Release the refit worker (called on session reset)."""
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ state
    @property
    def fitted(self):
        return self._fitted

    @property
    def n_labeled(self):
        return len(self._y)

    @property
    def refit_in_progress(self):
        return self._refit_in_progress

    def has_both_classes(self):
        if not self._y:
            return False
        seen = set(self._y)
        return 0 in seen and 1 in seen

    def balanced_window(self):
        """Best-effort down-sample the majority class so every refit stays fast
        and both classes are always represented.

        Crucially the returned rows keep **arrival order** (the index array is
        sorted).  Callers slice a *trailing* holdout out of this, and if the
        window were grouped by class that slice would contain one class only -
        which silently zeroes every F1 and pins the adaptive weights to a
        uniform fallback.
        """
        y = np.asarray(self._y, dtype=int)
        if len(self._X) == 0:
            return np.zeros((0, len(self.columns))), y
        X = np.vstack(self._X)
        if len(y) == 0:
            return X, y

        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            return X, y

        target_neg = min(len(neg), max(2 * len(pos), 500))
        if target_neg < len(neg):
            rng = np.random.default_rng(self._seed)
            neg_keep = neg[rng.choice(len(neg), size=target_neg, replace=False)]
        else:
            neg_keep = neg
        idx = np.sort(np.concatenate([pos, neg_keep]))
        return X[idx], y[idx]

    # -------------------------------------------------------------- streaming
    def add_labeled(self, X_row, y):
        with self._lock:
            cap = int(self.cfg.get("buffer_max_labels", 3000))
            self._X.append(np.asarray(X_row, dtype=float).reshape(-1))
            self._y.append(int(y))
            if len(self._X) > cap:
                del self._X[: len(self._X) - cap]
                del self._y[: len(self._y) - cap]
            self._buffered += 1

    def fit_needed(self, now_ts):
        """Refit when EITHER the event cadence OR the wall clock has elapsed."""
        if len(self._y) < int(self.cfg.get("min_fit_labels", 120)):
            return False
        if not self._fitted:
            return self._buffered > 0
        if self._buffered <= 0:
            return False
        by_events = self._buffered >= int(self.cfg.get("retrain_interval_events", 500))
        by_time = (now_ts - self._last_fit_at) >= int(
            self.cfg.get("retrain_interval_seconds", 60))
        return by_events or by_time

    # ------------------------------------------------------------------ fit
    def request_refit(self, now_ts=None):
        """Kick a background refit.  Returns False if one is already queued/running."""
        if not self.fit_needed(time.time() if now_ts is None else now_ts):
            return False
        with self._lock:
            if self._refit_in_progress:
                return False
            self._refit_in_progress = True
        self._executor.submit(self._fit_worker, now_ts)
        return True

    def _fit_worker(self, now_ts=None):
        started = time.time()
        try:
            snap = self._do_fit(now_ts)
            snap["ok"] = True
        except Exception as exc:  # surfaced via the snapshot
            with self._lock:
                self._refit_errors += 1
            snap = {"ok": False, "error": str(exc)}
        finally:
            with self._lock:
                self._refit_in_progress = False
                self._refit_count += 1
                self._last_refit_seconds = round(time.time() - started, 3)
        return snap

    def fit(self, now_ts=None):
        """Synchronous fit (used by the offline evaluation and by tests)."""
        return self._do_fit(now_ts)

    def _do_fit(self, now_ts=None):
        """Warm-start (or first-fit) every model on the balanced rolling window
        and recompute the ADS-style weights from a holdout slice.

        The deployed models are trained on the **whole** window (a streaming
        learner should use all data it has); the holdout is scored afterwards.
        Because XGBoost/HistGB resume trees fitted on earlier rounds of the same
        window, the resulting metrics are labelled optimistic in the snapshot -
        use `scripts/evaluate.py` for clean, reportable numbers.

        Concurrency contract: the lock is held only to *snapshot* the inputs and,
        at the very end, to *swap* the fitted references in. Training itself -
        the multi-second part - happens with no lock held, so `predict` on the
        consumer thread never waits for a fit. Training builds new estimators
        rather than mutating the live ones, so a prediction already in flight
        against the old references stays valid.
        """
        with self._lock:
            X, y = self.balanced_window()
            if len(y) < 4:
                return self.snapshot()
            self._buffered = 0
            self._last_fit_at = time.time() if now_ts is None else now_ts
            warm_from = {
                "booster": self._xgb_booster,
                "rounds_done": self._xgb_rounds_done,
                "hist_max_iter": self._hist_max_iter,
            }

        has_pos = int((y == 1).sum()) > 0
        has_neg = int((y == 0).sum()) > 0

        # ---- train into fresh objects, outside the lock ----
        new_models = {}
        breakdown = {}
        if has_pos and has_neg:
            t0 = time.time()
            booster, rounds = self._fit_xgb(X, y, warm_from["booster"])
            new_models["XGBoost"] = booster
            new_models["_xgb_rounds"] = warm_from["rounds_done"] + rounds
            breakdown["XGBoost"] = round(time.time() - t0, 3)
            t0 = time.time()
            new_models["HistGradientBoosting"] = self._fit_hist(
                X, y, warm_from["hist_max_iter"])
            breakdown["HistGradientBoosting"] = round(time.time() - t0, 3)
            t0 = time.time()
            new_models["ExtraTrees"] = self._fit_et(X, y)
            breakdown["ExtraTrees"] = round(time.time() - t0, 3)
        # else: single-class window - keep the previous models rather than
        # fitting a degenerate one; the cold-start prior still applies.

        if new_models:
            live = {k: v for k, v in new_models.items() if not k.startswith("_")}
        else:
            with self._lock:
                live = self._models_locked()
        t_eval = time.time()
        te_idx, holdout_mode, degenerate = self._holdout_indices(y)
        Xte, yte = (X[te_idx], y[te_idx]) if te_idx is not None else (X, y)
        preds, metrics = self._evaluate(live, Xte, yte)
        available = [m for m in MODEL_NAMES if m in preds]
        weights = self._weights_from(metrics, available)
        ensemble_metrics = self._score_ensemble(yte, preds, available, weights)
        if breakdown:
            breakdown["holdout_and_scoring"] = round(time.time() - t_eval, 3)

        with self._lock:
            if new_models:
                self._xgb_booster = new_models.get("XGBoost", self._xgb_booster)
                self._hist = new_models.get("HistGradientBoosting", self._hist)
                self._et = new_models.get("ExtraTrees", self._et)
                self._xgb_rounds_done = new_models.get("_xgb_rounds",
                                                      self._xgb_rounds_done)
                if "HistGradientBoosting" in new_models:
                    self._hist_max_iter = new_models["HistGradientBoosting"].max_iter
            # Only claim `fitted` when something was actually trained: a
            # single-class window keeps the previous models (or stays cold) and
            # must not be reported as fitted.
            if new_models:
                self._fitted = True
            self._metrics = metrics
            self._weights = weights
            self._ensemble_metrics = ensemble_metrics
            self._holdout_mode = holdout_mode
            self._holdout_degenerate = degenerate
            if breakdown:
                self._last_fit_breakdown = breakdown
            return self.snapshot()

    def _holdout_indices(self, y):
        """`(indices, mode, degenerate)` for the holdout rows, or
        `(None, mode, degenerate)` to score on the whole window.

        Primary strategy is the *trailing* slice (a genuine time split, since
        the window is arrival-ordered).  If that slice is single-class - which
        happens on short or bursty windows - fall back to a stratified sample
        so F1 is still computable, and say so in the snapshot.
        """
        y = np.asarray(y, dtype=int)
        n = len(y)
        if n < 4:
            return None, None, False
        cut = max(1, int(n * (1.0 - HOLDOUT_FRACTION)))
        tail = np.arange(cut, n)
        n_pos = int((y[tail] == 1).sum())
        n_neg = int((y[tail] == 0).sum())
        if n_pos >= MIN_HOLDOUT_PER_CLASS and n_neg >= MIN_HOLDOUT_PER_CLASS:
            return tail, "temporal", False

        n_pos_all = int((y == 1).sum())
        n_neg_all = int((y == 0).sum())
        if n_pos_all < MIN_HOLDOUT_PER_CLASS or n_neg_all < MIN_HOLDOUT_PER_CLASS:
            return None, "none", True
        want = max(2, int(n * HOLDOUT_FRACTION))
        per_class = max(MIN_HOLDOUT_PER_CLASS, want // 2)
        rng = np.random.default_rng(self._seed + 1)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        take_pos = rng.choice(pos, size=min(per_class, len(pos)), replace=False)
        take_neg = rng.choice(neg, size=min(per_class, len(neg)), replace=False)
        return np.sort(np.concatenate([take_pos, take_neg])), "stratified", True

    # --------------------------------------------------------- model training
    def _fit_xgb(self, X, y, base):
        """Returns `(new_booster, rounds_added)`; `base` is never mutated."""
        import xgboost as xgb

        params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": 6,
            "eta": 0.1,
            "eval_metric": "logloss",
            "seed": self._seed,
        }
        # `scale_pos_weight` defaults to the automatic neg/pos ratio, which on
        # a ~20%-prevalence problem drives recall above 90% at the cost of
        # precision. Setting `scale_pos_weight` in the config pins it (values
        # below 1 trade recall for precision); `scale_pos_weight: null`
        # disables re-weighting.
        spw = self.cfg.get("scale_pos_weight", None)
        if spw is None:
            params["scale_pos_weight"] = max(1.0, float((y == 0).sum())
                                             / max(1.0, float((y == 1).sum())))
        else:
            params["scale_pos_weight"] = float(spw)

        dtrain = xgb.DMatrix(X, label=y)
        rounds = (int(self.cfg.get("xgb_rounds", 200)) if base is None
                  else int(self.cfg.get("xgb_extra_rounds", 50)))
        rounds = max(rounds, 1)
        return xgb.train(params, dtrain, num_boost_round=rounds, xgb_model=base), rounds

    def _fit_hist(self, X, y, prev_max_iter):
        from sklearn.ensemble import HistGradientBoostingClassifier

        if prev_max_iter:
            max_iter = min(prev_max_iter + int(self.cfg.get("hist_extra_iter", 30)), 800)
        else:
            max_iter = int(self.cfg.get("hist_max_iter", 200))
        model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=0.1,
            max_depth=6,
            # "balanced" up-weights the minority class so aggressively that
            # recall exceeds 90% while precision collapses, which caps F1 on a
            # ~20%-prevalence problem. Configurable so the operating point can
            # be set on a validation split rather than fixed by assumption.
            class_weight=_class_weight(self.cfg),
            warm_start=True,
            random_state=self._seed,
        )
        model.fit(X, y)
        return model

    def _fit_et(self, X, y):
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(
            n_estimators=int(self.cfg.get("et_n_estimators", 200)),
            min_samples_split=5,
            class_weight=_class_weight_numeric(self.cfg),
            random_state=self._seed,
            n_jobs=-1,
        )
        model.fit(X, y)
        return model

    # ---------------------------------------------------------- evaluation
    def _score_model(self, y_true, probs, threshold=0.5):
        pred = (np.asarray(probs, dtype=float) >= threshold).astype(int)
        return _f1_precision_recall(np.asarray(y_true, dtype=int), pred)

    def _evaluate(self, models, Xte, yte):
        """Score every model in `models` on the supplied holdout arrays."""
        preds = self._model_probs(models, Xte)
        if len(yte) == 0:
            return preds, {}
        metrics = {name: self._score_model(yte, p) for name, p in preds.items()}
        return preds, metrics

    def _score_ensemble(self, yte, preds, available, weights=None):
        """Score the weighted blend.  Uses the *same* holdout arrays that the
        per-model metrics were computed on - re-deriving them from the rolling
        buffer would compare rows from different windows."""
        if not available or len(yte) == 0:
            return {}
        blend = self._blend(preds, available, weights)
        return self._score_model(yte, blend)

    def _blend(self, probs_by_model, available, weights=None):
        """Weighted mean of the available models' probabilities."""
        n = len(next(iter(probs_by_model.values())))
        w = weights if weights is not None else (self._weights or {})
        wsum = 0.0
        total = np.zeros(n, dtype=float)
        for name in available:
            wi = w.get(name, 0.0)
            if wi <= 0:
                continue
            total += wi * np.asarray(probs_by_model[name], dtype=float)
            wsum += wi
        if wsum <= 0:
            return np.mean([np.asarray(probs_by_model[m], dtype=float)
                            for m in available], axis=0)
        return total / wsum

    def _weights_from(self, metrics, available):
        """Adaptive (ADS-style) weights from per-model holdout F1.

        Proportional to F1, with a floor expressed *relative to the best model*
        so a collapsed model is bounded without flattening the blend when the
        models are merely similar.  Pure function of its inputs so `_do_fit` can
        compute the new weights before taking the lock to swap anything in.
        """
        if not available:
            return None
        scores = {name: max((metrics.get(name) or {}).get("f1", 0.0), 0.0)
                  for name in available}
        best = max(scores.values()) if scores else 0.0
        if best <= 0.0:
            # nothing to distinguish: fall back to an even blend
            share = 1.0 / len(available)
            return {name: share for name in available}
        floor = WEIGHT_FLOOR_FRACTION * best
        total = sum(max(v, floor) for v in scores.values())
        return {name: max(scores[name], floor) / total for name in available}

    # ------------------------------------------------------------ predict
    def predict_proba(self, X_row):
        """Weighted adaptive blend of the three models' fraud probability."""
        return self.predict_proba_batch(np.asarray(X_row, dtype=float).reshape(1, -1))[0]

    def predict_proba_batch(self, X):
        """Vectorised blend over a whole micro-batch.

        Batching is not an optimisation detail here, it is the difference
        between a real-time system and a slideshow: per-row ExtraTrees
        prediction costs ~135 ms of fixed overhead at 200 trees, while a batch
        of 500 costs the same ~135 ms in total.
        """
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        if single:
            X = X.reshape(1, -1)
        n = X.shape[0]
        if n == 0:
            return []

        # Grab the deployed references and the weights, then release the lock:
        # the (much slower) predict calls run against objects that a concurrent
        # refit is not allowed to mutate - it always trains replacements.
        with self._lock:
            fitted = self._fitted
            weights = dict(self._weights or {})
            models = self._models_locked()

        if not fitted or not models:
            out = [self._cold_start_proba()] * n
            return out[0] if single else out

        probs = self._model_probs(models, X)
        avail = [m for m in probs if m in models]
        if not avail or sum(weights.get(m, 0.0) for m in avail) <= 0:
            out = [self._cold_start_proba()] * n
            return out[0] if single else out

        out = np.clip(self._blend(probs, avail, weights), 0.0, 1.0)
        if single:
            return float(out[0])
        return [float(v) for v in out]

    def _models_locked(self):
        """(name -> estimator) for the deployed models.  Caller holds the lock."""
        out = {}
        if self._xgb_booster is not None:
            out["XGBoost"] = self._xgb_booster
        if self._hist is not None:
            out["HistGradientBoosting"] = self._hist
        if self._et is not None:
            out["ExtraTrees"] = self._et
        return out

    def deployed_models(self):
        """The fitted estimators currently deployed, keyed by model name.

        Public accessor for offline evaluation and tests, which need to score a
        clean holdout per model and must not reach into the private fields.
        """
        with self._lock:
            return self._models_locked()

    def _model_probs(self, models, X):
        """Per-model probabilities for a batch against the given model refs.
        A model that raises is simply dropped from the blend."""
        probs = {}
        booster = models.get("XGBoost")
        if booster is not None:
            try:
                import xgboost as xgb
                probs["XGBoost"] = np.asarray(
                    booster.predict(xgb.DMatrix(X)), dtype=float)
            except Exception:
                pass
        hist = models.get("HistGradientBoosting")
        if hist is not None:
            try:
                probs["HistGradientBoosting"] = np.asarray(
                    hist.predict_proba(X)[:, 1], dtype=float)
            except Exception:
                pass
        et = models.get("ExtraTrees")
        if et is not None:
            try:
                probs["ExtraTrees"] = np.asarray(
                    et.predict_proba(X)[:, 1], dtype=float)
            except Exception:
                pass
        return probs

    def _cold_start_proba(self):
        n = len(self._y)
        prior = 0.15 if n == 0 else float(sum(self._y) / n)
        return max(0.10, min(prior, 0.5))

    def decide(self, prob, thresholds):
        from backend.features import decision_from_prob
        return decision_from_prob(prob, thresholds)

    # ------------------------------------------------------------ snapshot
    def snapshot(self):
        with self._lock:
            return {
                "source": self.source,
                "fitted": self._fitted,
                "n_labeled": self.n_labeled,
                "buffered": self._buffered,
                "weights": dict(self._weights or {}),
                "models": {m: self._metrics.get(m) for m in MODEL_NAMES},
                "ensemble": self._ensemble_metrics,
                "holdout_mode": self._holdout_mode,
                "holdout_degenerate": self._holdout_degenerate,
                "metrics_note": "streaming holdout (optimistic: warm-start trees "
                                "carry earlier rounds of this window) - use "
                                "scripts/evaluate.py for clean numbers",
                "refit_in_progress": self._refit_in_progress,
                "refit_count": self._refit_count,
                "refit_errors": self._refit_errors,
                "last_refit_seconds": self._last_refit_seconds,
                "last_fit_breakdown": dict(self._last_fit_breakdown),
                "last_fit_at": self._last_fit_at or None,
                "xgb_rounds_done": self._xgb_rounds_done,
                "hist_max_iter": self._hist_max_iter,
            }
