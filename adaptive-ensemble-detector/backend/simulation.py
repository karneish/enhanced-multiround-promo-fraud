"""Multi-round adaptive ensemble simulation engine."""
import time
import math
import random
import threading
import traceback
import numpy as np

from .world import World, FRAUD_TEMPLATES, INTRINSIC_NAMES
from .features import compute_features, N_FEATURES
from .adaptive_detector import EnsembleDetector

DEFAULTS = {
    'rounds': 5,
    'seed': 42,
    'base_accounts': 500,
    'initial_fraud': 60,
    'genuine_per_round': 45,
    'fraud_per_round': 30,
    'supervised_ratio': 0.25,
    'budget_pos': 6,
    'budget_neg': 15,
    'describe': 'Adaptive Ensemble Detection',
}


def normalize_config(cfg):
    c = dict(DEFAULTS)
    for k, v in (cfg or {}).items():
        if k in DEFAULTS and v is not None:
            try:
                c[k] = type(DEFAULTS[k])(v)
            except (ValueError, TypeError):
                c[k] = DEFAULTS[k]
    if 'describe' in cfg:
        c['describe'] = str(cfg['describe'])[:120]
    return c


class Simulation:
    def __init__(self, cfg):
        self.cfg = normalize_config(cfg)
        self.id = f'{int(time.time() * 1000)}_{random.randint(1000, 9999)}'
        self.state = 'pending'
        self.events = []
        self.round_results = []
        self.report = None
        self._thread = None
        self.started_at = time.time()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.state = 'stopped'

    def _emit(self, ev):
        self.events.append(ev)

    def _log(self, text, t=0):
        self._emit({'type': 'log', 'text': text, 't': round(t, 2)})

    def _run(self):
        try:
            self.state = 'running'
            t0 = time.time()
            c = self.cfg
            self._log(f'[done] Starting adaptive ensemble simulation: {c["describe"]}')
            self._log(f'[done] Config: {c["rounds"]} rounds, {c["base_accounts"]} base, {c["initial_fraud"]} fraud, seed={c["seed"]}')

            world = World(seed=c['seed'])
            world.build_initial(c['base_accounts'], c['initial_fraud'], c['seed'])
            detector = EnsembleDetector()

            supervised_mask = world.supervised_mask(c['supervised_ratio'])

            genuine_indices = np.where((~supervised_mask) & (world.label_array() == 0))[0]
            if len(genuine_indices) > 0:
                n_genuine_label = min(int(len(genuine_indices) * c['supervised_ratio']), len(genuine_indices))
                rng_gen = np.random.RandomState(c['seed'] + 999)
                gen_label_idx = rng_gen.choice(genuine_indices, n_genuine_label, replace=False)
                for idx in gen_label_idx:
                    supervised_mask[idx] = True

            self._log(f'[done] Initial supervised: {int(supervised_mask.sum())} accounts')

            for round_idx in range(c['rounds']):
                if self.state == 'stopped':
                    self._log('[warn] Simulation stopped by user')
                    break

                self._log(f'[done] === ROUND {round_idx} ===')
                round_t0 = time.time()

                X, feat_names = compute_features(world)
                labels = world.label_array()
                active = world.active_mask()

                train_mask = active & supervised_mask
                val_mask = active & ~supervised_mask

                if train_mask.sum() < 10 or val_mask.sum() < 5:
                    self._log('[warn] Not enough labelled data, skipping round training')
                    continue

                X_train, y_train = X[train_mask], labels[train_mask].astype(int)
                X_val, y_val = X[val_mask], labels[val_mask].astype(int)

                pos_count = int(y_train.sum())
                neg_count = len(y_train) - pos_count
                ce_weight = neg_count / max(pos_count, 1)
                self._log(f'  Training: {len(X_train)} samples ({pos_count} pos, {neg_count} neg), CE weight={ce_weight:.2f}')

                if pos_count == 0 or neg_count == 0:
                    self._log('[warn] Only one class in training data, skipping round training')
                    continue
                if len(y_val) == 0 or len(np.unique(y_val)) < 2:
                    self._log('[warn] Only one class in validation data, skipping round training')
                    continue

                try:
                    train_results = detector.train(X_train, y_train, X_val, y_val, ce_weight)
                except Exception as exc:
                    self._log(f'[warn] Training failed: {exc}, skipping round')
                    continue
                for name, res in train_results.items():
                    self._log(f'    {name}: F1={res["f1"]:.4f} Recall={res["recall"]:.4f}')

                try:
                    ensemble_eval = detector.evaluate(X[active], labels[active].astype(int))
                    per_model = detector.evaluate_per_model(X[active], labels[active].astype(int))
                except Exception as exc:
                    self._log(f'[warn] Evaluation failed: {exc}, using defaults')
                    ensemble_eval = {'rec': 0.0, 'prec': 0.0, 'f1': 0.0, 'auc': 0.5, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0, 'threshold': 0.5}
                    per_model = {}
                state = detector.get_state()

                self._log(f'  Ensemble: F1={ensemble_eval["f1"]:.4f} AUC={ensemble_eval["auc"]:.4f} REC={ensemble_eval["rec"]:.4f} PRE={ensemble_eval["prec"]:.4f}')

                probs_all = detector.predict_proba(X)
                missed_fraud = []
                for i in range(len(world.accounts)):
                    if active[i] and labels[i] == 1 and probs_all[i] < 0.5:
                        missed_fraud.append({
                            'id': world.accounts[i].id,
                            'strategy': world.accounts[i].strategy,
                            'prob': float(probs_all[i]),
                            'attrs': {k: round(v, 3) for k, v in (world.accounts[i].attrs or {}).items()},
                        })

                self._log(f'  Missed fraud: {len(missed_fraud)} accounts (of {int(labels[active].sum())} total fraud)')

                new_fraud_features = self._generate_evolved_fraud(missed_fraud, c, world.rng)
                genuine_added = world.add_round_accounts(
                    round_idx, c['genuine_per_round'], c['fraud_per_round'],
                    fraud_features=new_fraud_features, fraud_strategy='evolved'
                )
                self._log(f'  Added {c["genuine_per_round"]} genuine + {len(genuine_added)} evolved fraud accounts')

                n_total = len(world.accounts)
                if len(supervised_mask) < n_total:
                    extra = np.zeros(n_total - len(supervised_mask), dtype=bool)
                    supervised_mask = np.concatenate([supervised_mask, extra])

                new_fraud_mask = world.round_mask(round_idx) & (world.label_array() == 1)
                reveal_pos = min(c['budget_pos'], int(new_fraud_mask.sum()))
                if reveal_pos > 0:
                    new_fraud_idx = np.where(new_fraud_mask)[0]
                    reveal_idx = np.random.RandomState(c['seed'] + round_idx).choice(
                        new_fraud_idx, reveal_pos, replace=False
                    )
                    for idx in reveal_idx:
                        supervised_mask[idx] = True
                new_genuine_mask = world.round_mask(round_idx) & (world.label_array() == 0)
                reveal_neg = min(c['budget_neg'], int(new_genuine_mask.sum()))
                if reveal_neg > 0:
                    new_genuine_idx = np.where(new_genuine_mask)[0]
                    reveal_neg_idx = np.random.RandomState(c['seed'] + round_idx + 1000).choice(
                        new_genuine_idx, reveal_neg, replace=False
                    )
                    for idx in reveal_neg_idx:
                        supervised_mask[idx] = True
                self._log(f'  Revealed {reveal_pos} fraud + {reveal_neg} genuine as budget (total supervised: {int(supervised_mask.sum())})')

                round_time = time.time() - round_t0
                round_result = {
                    'round': round_idx,
                    'num_nodes': int(active.sum()),
                    'num_edges': world.edge_count(),
                    'ensemble': ensemble_eval,
                    'per_model': per_model,
                    'weights': state['weights'],
                    'scores': state['scores'],
                    'individual_f1': {n: list(v) for n, v in state['individual_f1'].items()},
                    'individual_recall': {n: list(v) for n, v in state['individual_recall'].items()},
                    'missed_fraud': len(missed_fraud),
                    'new_fraud_generated': len(new_fraud_features),
                    'total_fraud': int(labels[active].sum()),
                    'total_genuine': int((labels[active] == 0).sum()),
                    'supervised_count': int(supervised_mask.sum()),
                    'time': round(round_time, 2),
                    'fraud_strategies': world.strategy_counts(new_fraud_mask),
                    'label_counts': world.label_counts(),
                }
                self.round_results.append(round_result)
                self._emit({'type': 'round_result', **round_result})
                self._log(f'[done] Round {round_idx} complete in {round_time:.1f}s')

            self.state = 'done'
            self._log(f'[done] Simulation finished. {len(self.round_results)} rounds completed.')

            self.report = {
                'config': c,
                'rounds': self.round_results,
                'total_time': round(time.time() - t0, 2),
            }
            self._emit({'type': 'state', 'state': 'done', 'finished': True})

        except Exception as e:
            self.state = 'error'
            self._log(f'[error] {str(e)}')
            self._log(f'[error] {traceback.format_exc()}')
            self._emit({'type': 'state', 'state': 'error', 'finished': True, 'error': str(e)})

    def _generate_evolved_fraud(self, missed, cfg, rng):
        if not missed:
            features = []
            templates = list(FRAUD_TEMPLATES.values())
            for _ in range(cfg['fraud_per_round']):
                tmpl = rng.choice(templates)
                noise = tmpl['noise']
                row = [max(0, min(1, m + rng.gauss(0, noise))) for m in tmpl['means']]
                features.append(row)
            return features

        features = []
        for _ in range(cfg['fraud_per_round']):
            seed = rng.choice(missed)
            base_attrs = list(seed['attrs'].values()) if seed['attrs'] else [rng.random() for _ in range(len(INTRINSIC_NAMES))]
            if len(base_attrs) < len(INTRINSIC_NAMES):
                base_attrs = list(base_attrs) + [0.5] * (len(INTRINSIC_NAMES) - len(base_attrs))

            new_row = []
            for j in range(len(INTRINSIC_NAMES)):
                mode = rng.random()
                if mode < 0.5:
                    val = base_attrs[j] + rng.gauss(0, 0.12)
                elif mode < 0.8:
                    vals = [m['attrs'].get(INTRINSIC_NAMES[j], 0.5) for m in missed if m['attrs']]
                    mean_v = float(np.mean(vals)) if vals else 0.5
                    val = mean_v + rng.gauss(0, 0.08)
                else:
                    val = rng.random()
                new_row.append(max(0.0, min(1.0, val)))
            features.append(new_row)
        return features
