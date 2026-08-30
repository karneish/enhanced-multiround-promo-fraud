"""Adaptive Multi-Model Ensemble Detector with ADS scoring."""
import warnings
import numpy as np
from collections import deque

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight

_XGB_VERSION = tuple(int(x) for x in xgb.__version__.split('.')[:2])


class AdaptiveDetectorScore:
    def __init__(self, model_names, history_window=5, ema_alpha=0.7, recall_importance=1.0):
        self.model_names = list(model_names)
        self.history_window = history_window
        self.ema_alpha = ema_alpha
        self.recall_importance = recall_importance
        self.history = {name: deque(maxlen=history_window) for name in self.model_names}
        self.ema_scores = {name: 0.0 for name in self.model_names}

    def update(self, model_name, f1, recall):
        self.history[model_name].append({'f1': f1, 'recall': recall})
        if len(self.history[model_name]) == 1:
            self.ema_scores[model_name] = f1
        else:
            self.ema_scores[model_name] = (
                self.ema_alpha * f1 + (1 - self.ema_alpha) * self.ema_scores[model_name]
            )

    def _f1_score(self, name):
        h = self.history[name]
        return h[-1]['f1'] if h else 0.0

    def _recall_score(self, name):
        h = self.history[name]
        return h[-1]['recall'] * self.recall_importance if h else 0.0

    def _stability(self, name):
        h = self.history[name]
        if len(h) < 2:
            return 1.0
        vals = [x['f1'] for x in h]
        mean = np.mean(vals)
        std = np.std(vals)
        return 1.0 - min(std / (mean + 1e-10), 1.0)

    def _historical(self, name):
        return self.ema_scores[name]

    def compute_score(self, name):
        return (
            0.25 * self._f1_score(name) +
            0.25 * self._recall_score(name) +
            0.25 * self._stability(name) +
            0.25 * self._historical(name)
        )

    def compute_all_scores(self):
        return {name: self.compute_score(name) for name in self.model_names}

    def compute_weights(self):
        scores = self.compute_all_scores()
        total = sum(scores.values())
        if total == 0:
            n = len(self.model_names)
            return {name: 1.0 / n for name in self.model_names}
        return {name: s / total for name, s in scores.items()}


class EnsembleDetector:
    def __init__(self, model_names=None, history_window=5, ema_alpha=0.7):
        self.model_names = model_names or [
            'XGBoost', 'HistGradientBoosting', 'ExtraTrees'
        ]
        self.classifiers = {}
        self.fitted = {}
        self.ads = AdaptiveDetectorScore(
            self.model_names, history_window=history_window, ema_alpha=ema_alpha
        )
        self._init_classifiers()

    def _init_classifiers(self):
        for name in self.model_names:
            if name == 'XGBoost':
                self.classifiers[name] = None
            elif name == 'RandomForest':
                self.classifiers[name] = RandomForestClassifier(
                    n_estimators=300, max_depth=None, min_samples_split=5,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
            elif name == 'ExtraTrees':
                self.classifiers[name] = ExtraTreesClassifier(
                    n_estimators=300, max_depth=None, min_samples_split=5,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
            elif name == 'HistGradientBoosting':
                self.classifiers[name] = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.1, max_depth=6, random_state=42
                )
            elif name == 'LogisticRegression':
                self.classifiers[name] = LogisticRegression(
                    max_iter=5000, solver='lbfgs', class_weight='balanced',
                    random_state=42, tol=1e-4, C=1.0
                )
            self.fitted[name] = False

    def train(self, X_train, y_train, X_val, y_val, ce_weight=1.0):
        sample_weights = compute_sample_weight('balanced', y_train)
        results = {}
        for name in self.model_names:
            try:
                if name == 'XGBoost':
                    params = {
                        "objective": "binary:logistic",
                        "scale_pos_weight": ce_weight,
                        "tree_method": "hist",
                        "max_depth": 6,
                    }
                    if _XGB_VERSION >= (2, 0):
                        params["device"] = "cpu"
                    dtrain = xgb.DMatrix(X_train, y_train)
                    dval = xgb.DMatrix(X_val, y_val)
                    existing = self.classifiers['XGBoost'] if self.fitted.get('XGBoost') else None
                    self.classifiers['XGBoost'] = xgb.train(
                        params, dtrain,
                        num_boost_round=500, early_stopping_rounds=100,
                        evals=[(dtrain, 'Train'), (dval, 'Eval')],
                        verbose_eval=False, xgb_model=existing,
                    )
                    self.fitted['XGBoost'] = True
                else:
                    clf = self.classifiers[name]
                    try:
                        clf.fit(X_train, y_train, sample_weight=sample_weights)
                    except TypeError:
                        clf.fit(X_train, y_train)
                    self.fitted[name] = True

                val_preds = self._predict_proba_single(name, X_val)
                if val_preds is not None and len(val_preds) == len(y_val):
                    val_binary = (val_preds > 0.5).astype(int)
                    f1 = f1_score(y_val, val_binary, average='macro', zero_division=0)
                    rec = recall_score(y_val, val_binary, zero_division=0)
                    self.ads.update(name, f1, rec)
                    results[name] = {'f1': float(f1), 'recall': float(rec)}
            except Exception as exc:
                warnings.warn(f'Model {name} training failed: {exc}', stacklevel=2)
        return results

    def _predict_proba_single(self, name, X):
        if not self.fitted.get(name):
            return None
        if name == 'XGBoost' and self.classifiers[name] is not None:
            return self.classifiers[name].predict(xgb.DMatrix(X))
        elif self.classifiers.get(name) is not None:
            try:
                proba = self.classifiers[name].predict_proba(X)
            except Exception:
                return None
            proba = np.asarray(proba)
            if proba.ndim == 1:
                return proba
            if proba.shape[1] == 1:
                return proba[:, 0]
            return proba[:, 1]
        return None

    def predict_proba(self, X):
        all_probs = {}
        for name in self.model_names:
            try:
                probs = self._predict_proba_single(name, X)
                if probs is not None:
                    probs = np.asarray(probs).flatten()
                    if len(probs) == X.shape[0]:
                        all_probs[name] = probs
            except Exception:
                pass
        if not all_probs:
            return np.zeros(X.shape[0])
        weights = self.ads.compute_weights()
        ensemble = np.zeros(X.shape[0])
        total_w = 0.0
        for name, probs in all_probs.items():
            w = weights.get(name, 0.0)
            ensemble += w * probs
            total_w += w
        if total_w > 0:
            ensemble /= total_w
        return ensemble

    def evaluate(self, X, y):
        probs = self.predict_proba(X)
        binary = (probs > 0.5).astype(int)
        tp = int(((y == 1) & (binary == 1)).sum())
        fp = int(((y == 0) & (binary == 1)).sum())
        tn = int(((y == 0) & (binary == 0)).sum())
        fn = int(((y == 1) & (binary == 0)).sum())
        rec = recall_score(y, binary, zero_division=0)
        prec = precision_score(y, binary, zero_division=0)
        f1 = f1_score(y, binary, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(y, probs)
        except (ValueError, Exception):
            auc = 0.5
        return {
            'rec': float(rec), 'prec': float(prec), 'f1': float(f1), 'auc': float(auc),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'threshold': 0.5,
        }

    def evaluate_per_model(self, X, y):
        results = {}
        for name in self.model_names:
            probs = self._predict_proba_single(name, X)
            if probs is None or len(probs) != len(y):
                continue
            binary = (probs > 0.5).astype(int)
            try:
                auc = roc_auc_score(y, probs)
            except (ValueError, Exception):
                auc = 0.5
            results[name] = {
                'f1': float(f1_score(y, binary, average='macro', zero_division=0)),
                'recall': float(recall_score(y, binary, zero_division=0)),
                'precision': float(precision_score(y, binary, zero_division=0)),
                'auc': float(auc),
            }
        return results

    def get_state(self):
        return {
            'weights': self.ads.compute_weights(),
            'scores': self.ads.compute_all_scores(),
            'individual_f1': {n: [h['f1'] for h in self.ads.history[n]] for n in self.model_names},
            'individual_recall': {n: [h['recall'] for h in self.ads.history[n]] for n in self.model_names},
        }
