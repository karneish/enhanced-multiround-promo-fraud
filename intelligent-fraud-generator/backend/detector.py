"""Gradient-boosted fraud detector.

A thin wrapper around XGBoost that:

  * balances the classes with ``scale_pos_weight``,
  * picks the decision threshold from a validation split using macro-F1,
  * exposes a sklearn-style ``train`` / ``predict`` interface.
"""

import numpy as np
import xgboost as xgb

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def best_threshold(labels, scores):
    """Grid-search the probability threshold maximising macro-F1."""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if np.unique(labels).size < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        preds = (scores > t).astype(int)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def evaluate(labels, probs, threshold=None):
    """Compute recall / precision / macro-F1 / AUC for fraud class."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    if threshold is None:
        threshold = best_threshold(labels, probs)
    preds = (probs > threshold).astype(int)
    rec = recall_score(labels, preds, zero_division=0)
    prec = precision_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    auc = roc_auc_score(labels, probs) if np.unique(labels).size > 1 else -1.0
    tp = int(((labels == 1) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    return {
        'rec': float(rec), 'prec': float(prec), 'f1': float(f1), 'auc': float(auc),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'threshold': float(threshold),
    }


class FraudDetector:
    def __init__(self, seed=0):
        self.seed = seed
        self.model = None
        self.threshold = 0.5

    def train(self, X, y, X_val=None, y_val=None):
        y = np.asarray(y, dtype=int)
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        if pos < 2 or neg < 2:
            raise ValueError('need at least 2 samples of each class to train')

        model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.12,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=1.0,
            scale_pos_weight=neg / pos,
            random_state=self.seed,
            tree_method='hist',
            n_jobs=2,
            eval_metric='auc',
            verbosity=0,
        )
        model.fit(X, y)
        self.model = model

        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_scores = model.predict_proba(np.asarray(X_val))[:, 1]
            self.threshold = best_threshold(y_val, val_scores)
        else:
            self.threshold = 0.5
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError('detector not trained')
        return self.model.predict_proba(np.asarray(X))[:, 1]
