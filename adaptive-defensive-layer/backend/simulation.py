"""Multi-round adversarial simulation engine with an Adaptive Defensive Layer.

Full ecosystem loop (attacker - detector - defense)::

    build world -> train detector -> predict fraud probability
        -> ADL: risk score -> Allow / Review / Block
        -> blocked fraud is removed from the graph (never reaches the attacker)
        -> only SURVIVING fraud feeds the intelligent generator
        -> generator synthesises evolved fraud + fresh genuine users
        -> thresholds adapt (escape vs false-block trade-off)
        -> retrain detector -> repeat

Every round is logged (detector metrics + generator diagnostics + defence
metrics) and streamed to the frontend through an in-memory event list.
"""

import threading
import time
import traceback

import numpy as np
from sklearn.model_selection import train_test_split

from .world import World, INITIAL_STRATEGIES, INTRINSIC_NAMES
from .features import compute_features
from .detector import FraudDetector, evaluate
from .adl import AdaptiveDefense
from .generator.engine import IntelligentFraudGenerator, ReplayGenerator

DEFAULTS = {
    'rounds': 8,
    'seed': 7,
    'base_accounts': 500,
    'initial_fraud': 60,
    'genuine_per_round': 45,
    'fraud_per_round': 30,
    'generator_mode': 'intelligent',     # 'intelligent' | 'replay'
    'gen_type': 'GAN',                   # 'GAN' | 'PROB'
    'gan_epochs': 120,
    'gan_noise_dim': 12,
    'gan_hidden': 32,
    'diversity': 1.0,
    'conn_coef': 0.6,
    'ring_ratio': 0.5,
    'profile_window': 5,
    'supervised_ratio': 0.25,
    'forget_window': 2,              # detector retrains only on recent rounds
    'budget_pos': 6,
    'budget_neg': 15,
    # --- Adaptive Defensive Layer ---
    'adl_enabled': True,
    'threshold_policy': 'adaptive',  # 'adaptive' | 'fixed'
    't1': 0.40,
    't2': 0.75,
    'threshold_alpha': 0.05,
    'review_catch_rate': 0.5,
    'w_pf': 0.45,
    'w_centrality': 0.20,
    'w_ring': 0.15,
    'w_velocity': 0.10,
    'w_trust': 0.10,
    'describe': 'Adaptive Defensive Layer experiment',
}

_INITIAL_WEIGHTS = [0.15, 0.35, 0.20, 0.15, 0.15]

_ADL_KEYS = ['adl_enabled', 'threshold_policy', 't1', 't2', 'threshold_alpha',
             'review_catch_rate', 'w_pf', 'w_centrality', 'w_ring',
             'w_velocity', 'w_trust']


def normalize_config(raw):
    cfg = dict(DEFAULTS)
    if isinstance(raw, dict):
        for k, v in raw.items():
            if v is None or v == '':
                continue
            cfg[k] = v
    cfg['rounds'] = max(1, int(cfg['rounds']))
    cfg['seed'] = int(cfg['seed'])
    cfg['base_accounts'] = max(50, int(cfg['base_accounts']))
    cfg['initial_fraud'] = max(5, int(cfg['initial_fraud']))
    cfg['genuine_per_round'] = max(0, int(cfg['genuine_per_round']))
    cfg['fraud_per_round'] = max(0, int(cfg['fraud_per_round']))
    cfg['gan_epochs'] = max(1, int(cfg['gan_epochs']))
    cfg['gan_noise_dim'] = max(2, int(cfg['gan_noise_dim']))
    cfg['gan_hidden'] = max(4, int(cfg['gan_hidden']))
    cfg['diversity'] = float(cfg['diversity'])
    cfg['conn_coef'] = float(cfg['conn_coef'])
    cfg['ring_ratio'] = float(cfg['ring_ratio'])
    cfg['profile_window'] = max(1, int(cfg['profile_window']))
    cfg['supervised_ratio'] = min(1.0, max(0.0, float(cfg['supervised_ratio'])))
    cfg['ring_ratio'] = min(1.0, max(0.0, cfg['ring_ratio']))
    cfg['conn_coef'] = min(1.0, max(0.0, cfg['conn_coef']))
    cfg['forget_window'] = max(1, int(cfg['forget_window']))
    cfg['budget_pos'] = max(0, int(cfg['budget_pos']))
    cfg['budget_neg'] = max(0, int(cfg['budget_neg']))
    cfg['generator_mode'] = str(cfg['generator_mode']).lower()
    cfg['gen_type'] = str(cfg['gen_type']).upper()
    if cfg['generator_mode'] not in ('intelligent', 'replay'):
        cfg['generator_mode'] = 'intelligent'
    if cfg['gen_type'] not in ('GAN', 'PROB'):
        cfg['gen_type'] = 'GAN'
    # ADL fields
    cfg['adl_enabled'] = bool(cfg['adl_enabled'])
    cfg['threshold_policy'] = str(cfg['threshold_policy']).lower()
    if cfg['threshold_policy'] not in ('adaptive', 'fixed'):
        cfg['threshold_policy'] = 'adaptive'
    cfg['t1'] = min(0.95, max(0.05, float(cfg['t1'])))
    cfg['t2'] = min(0.98, max(0.10, float(cfg['t2'])))
    if cfg['t2'] <= cfg['t1']:
        cfg['t2'] = min(0.98, cfg['t1'] + 0.05)
    cfg['threshold_alpha'] = float(cfg['threshold_alpha'])
    cfg['review_catch_rate'] = min(1.0, max(0.0, float(cfg['review_catch_rate'])))
    for k in ('w_pf', 'w_centrality', 'w_ring', 'w_velocity', 'w_trust'):
        cfg[k] = max(0.0, float(cfg[k]))
    return cfg


def _adl_weights(cfg):
    return {
        'w_pf': cfg['w_pf'], 'w_centrality': cfg['w_centrality'],
        'w_ring': cfg['w_ring'], 'w_velocity': cfg['w_velocity'],
        'w_trust': cfg['w_trust'],
    }


class Simulation:
    def __init__(self, sim_id, config):
        self.id = sim_id
        self.config = normalize_config(config)
        self.state = 'pending'
        self.error = None
        self.started_at = time.time()
        self.finished_at = None

        self.lock = threading.Lock()
        self.events = []          # log lines + structured events

        self.world = None
        self.detector = None
        self.generator = None
        self.adl = None
        self.supervised = None
        self.labels = None
        self.pred_probs = None
        self.pred_threshold = 0.5
        self.rounds = []
        self.profile_summary = None
        self.strategy_evolution = []
        self._thread = None

    # ------------------------------------------------------------------ #
    # public control
    # ------------------------------------------------------------------ #
    def start(self):
        self.state = 'running'
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self.state = 'stopped'
        self.finished_at = time.time()

    def is_finished(self):
        return self.state in ('done', 'error', 'stopped')

    # ------------------------------------------------------------------ #
    # event helpers
    # ------------------------------------------------------------------ #
    def emit(self, **payload):
        with self.lock:
            payload.setdefault('t', round(time.time() - self.started_at, 3))
            self.events.append(payload)

    def log(self, text):
        self.emit(type='log', text=str(text))

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def _run(self):
        try:
            if self.state != 'running':
                self.state = 'running'
            cfg = self.config
            rng = np.random.default_rng(cfg['seed'] + 1)
            if cfg['generator_mode'] == 'replay':
                self.generator = ReplayGenerator(cfg, rng)
            else:
                self.generator = IntelligentFraudGenerator(cfg, rng)

            if cfg['adl_enabled']:
                self.adl = AdaptiveDefense(
                    weights=_adl_weights(cfg),
                    t1=cfg['t1'], t2=cfg['t2'], alpha=cfg['threshold_alpha'],
                    policy=cfg['threshold_policy'],
                    review_catch_rate=cfg['review_catch_rate'],
                    seed=cfg['seed'] + 2,
                )

            self.log(f'[start] {self.config["describe"]}')
            self.log(f'[config] generator={self.config["generator_mode"]} '
                     f'model={self.config["gen_type"]} rounds={self.config["rounds"]} '
                     f'base={self.config["base_accounts"]} fraud0={self.config["initial_fraud"]}')
            if self.adl:
                st = self.adl.state()
                self.log(f'[config] ADL={st["policy"]} T1={st["t1"]} T2={st["t2"]} '
                         f'alpha={st["alpha"]} review_catch={st["review_catch_rate"]} '
                         f'weights={ {k: round(v, 2) for k, v in st["weights"].items()} }')
            self._round_zero()
            for r in range(1, self.config['rounds']):
                if self.state != 'running':
                    break
                self._step_round(r)
            self.state = 'done'
            self.profile_summary = self.generator.profile.summary()
            self.log(f'[done] simulation finished in '
                     f'{time.time() - self.started_at:.1f}s')
        except Exception as exc:  # noqa: BLE001
            self.error = traceback.format_exc()
            self.state = 'error'
            self.log(f'[error] {exc}')
            self.log(traceback.format_exc())
        finally:
            self.finished_at = time.time()
            self.emit(type='state', state=self.state)

    # ------------------------------------------------------------------ #
    # round 0
    # ------------------------------------------------------------------ #
    def _round_zero(self):
        self.emit(type='round', round=0, phase='build')
        cfg = self.config
        world = World(seed=cfg['seed'])
        self.world = world

        self.log(f'[r0] building world: {cfg["base_accounts"]} genuine + '
                 f'{cfg["initial_fraud"]} fraud accounts')
        world.add_genuine(0, cfg['base_accounts'])
        self._add_initial_fraud(world, cfg['initial_fraud'])

        n = world.num_nodes()
        self.supervised = np.zeros(n, dtype=bool)
        sup = world.np.choice(n, size=max(2, int(round(n * cfg['supervised_ratio']))),
                              replace=False)
        self.supervised[sup] = True

        self._retrain_and_predict()
        self._run_defense(0)
        self._finish_round(0, phase='round0')

    def _step_round(self, r):
        cfg = self.config
        world = self.world
        self.emit(type='round', round=r, phase='start')
        self.log(f'[r{r}] generating new fraud + genuine users')

        # 1. new genuine users
        if cfg['genuine_per_round'] > 0:
            world.add_genuine(r, cfg['genuine_per_round'])

        # 2. new fraud from the generator (learned from *survivors* only)
        n_fraud = cfg['fraud_per_round']
        if n_fraud > 0:
            specs, edges = self.generator.generate(n_fraud, r, world)
            base = world.num_nodes()
            for spec in specs:
                refs = []
                for dst in spec.get('referrals', []):
                    dst = int(dst)
                    refs.append(base + dst)
                for dst in spec.get('victim_referrals', []):
                    refs.append(int(dst))
                spec['referrals'] = refs
            new_ids = [world.add_fraud(s, r) for s in specs]
            if specs:
                self.strategy_evolution.append({
                    'round': r,
                    'counts': self.generator.get_stats().get('gen_base_strategies', {}),
                    'strategies': self.generator.get_stats().get('gen_strategies', {}),
                })
            self.log(f'[r{r}] injected {len(new_ids)} new fraud accounts '
                     f'(mode={self.generator.name})')

        # 3. reveal budgeted ground truth (manual review of the round)
        self._reveal_budget(r)

        # 4. retrain + predict + ADL defense + evaluate
        self._retrain_and_predict()
        self._run_defense(r)
        self._finish_round(r, phase='round')

    # ------------------------------------------------------------------ #
    # detection + defense pipeline
    # ------------------------------------------------------------------ #
    def _labels(self):
        return np.array([a.label for a in self.world.accounts])

    def _active_mask(self):
        return self.world.active_mask()

    def _retrain_and_predict(self):
        self._ensure_supervised_size()
        X = compute_features(self.world, self.supervised)
        labels = self._labels()
        active = self._active_mask()

        # detector is retrained only on accounts created recently that are
        # still live (blocked accounts are gone from the system)
        cur_round = len(self.rounds)
        created = np.array([a.creation_round for a in self.world.accounts])
        recent = created >= (cur_round - self.config['forget_window'])
        sup_idx = np.where(active & self.supervised & recent)[0]

        if self.detector is None:
            self.detector = FraudDetector(seed=self.config['seed'])

        if len(sup_idx) >= 20:
            y = labels[sup_idx]
            try:
                tr, va = train_test_split(np.arange(len(sup_idx)),
                                          test_size=0.25, stratify=y,
                                          random_state=self.config['seed'] + len(self.rounds))
                self.detector.train(X[sup_idx[tr]], y[tr],
                                    X[sup_idx[va]], y[va])
            except ValueError as exc:
                self.log(f'[warn] retrain skipped: {exc}')

        try:
            probs = self.detector.predict(X)
            self.pred_probs = probs
            self.pred_threshold = self.detector.threshold
        except RuntimeError as exc:
            self.log(f'[warn] predict skipped: {exc}')
            self.pred_probs = np.full(len(labels), 0.5)

    def _run_defense(self, r):
        """Run the Adaptive Defensive Layer for round ``r``."""
        if self.adl is None:
            return None
        labels = self._labels()
        predicted_fraud = self.pred_probs > self.pred_threshold
        known_fraud = self.supervised & (labels == 1)
        record = self.adl.step(self.world, self.pred_probs, predicted_fraud,
                               known_fraud, labels, r,
                               weights=_adl_weights(self.config))
        self.log(f'[r{r}] defense: block={record["block_rate"]:.2f} '
                 f'escape={record["escape_rate"]:.2f} '
                 f'false_block={record["false_block_rate"]:.2f} '
                 f'prec={record["defense_precision"]:.2f} '
                 f'recall={record["defense_recall"]:.2f} '
                 f'avg_risk={record["avg_risk"]:.3f} '
                 f'latency={record["decision_latency_ms"]:.2f}ms '
                 f'(T1={record["t1"]:.2f} T2={record["t2"]:.2f})')
        return record

    def _finish_round(self, r, phase):
        labels = self._labels()
        probs = self.pred_probs
        thres = self.pred_threshold
        active = self._active_mask()

        metrics = evaluate(labels[active], probs[active], threshold=thres)

        # metrics restricted to accounts created in this round (still live)
        created = np.array([a.creation_round for a in self.world.accounts])
        fresh_mask = active & (created == r)
        fresh_metrics = None
        if fresh_mask.sum() >= 2 and np.unique(labels[fresh_mask]).size > 1:
            fresh_metrics = evaluate(labels[fresh_mask], probs[fresh_mask],
                                     threshold=thres)

        gen_stats = self.generator.get_stats() if self.generator else {}
        defense = None
        if self.adl and self.adl.history:
            defense = dict(self.adl.history[-1])
            defense['components'] = self.adl.component_means(
                self.world, probs, probs > thres,
                self.supervised & (labels == 1))

        record = {
            'round': r,
            'phase': phase,
            'num_nodes': int(self.world.num_nodes()),
            'active_nodes': int(active.sum()),
            'blocked_total': int((~active).sum()),
            'num_edges': int(self.world.num_edges()),
            'new_fraud': int(((created == r) & (labels == 1)).sum()),
            'new_genuine': int(((created == r) & (labels == 0)).sum()),
            'missed': int((active & self.supervised & (labels == 1)
                           & (probs < thres)).sum()),
            'metrics': metrics,
            'fresh_metrics': fresh_metrics,
            'gen': gen_stats,
            'defense': defense,
        }
        self.rounds.append(record)

        self.log(f'[r{r}] overall F1={metrics["f1"]:.3f} AUC={metrics["auc"]:.3f} '
                 f'REC={metrics["rec"]:.3f} PRE={metrics["prec"]:.3f} '
                 f'(tp={metrics["tp"]} fn={metrics["fn"]} fp={metrics["fp"]} tn={metrics["tn"]})')
        if gen_stats:
            self.log(f'[r{r}] generator: div={gen_stats.get("gen_feat_div")} '
                     f'shift={gen_stats.get("gen_feat_shift")} '
                     f'ring_ratio={gen_stats.get("gen_ring_ratio")} '
                     f'new_edges={gen_stats.get("gen_new_edges")}')

        # feed the SURVIVING fraud into the generator for the next round
        self._collect_missed(r)

        self.emit(type='round_result', round=r, metrics=metrics,
                  fresh_metrics=fresh_metrics, gen=gen_stats,
                  defense=defense,
                  num_nodes=int(self.world.num_nodes()),
                  active_nodes=int(active.sum()),
                  missed=record['missed'])

    # ------------------------------------------------------------------ #
    # missed-fraud collection (the attacker only sees survivors)
    # ------------------------------------------------------------------ #
    def _ensure_supervised_size(self):
        n = len(self.world.accounts)
        if len(self.supervised) < n:
            extra = np.zeros(n - len(self.supervised), dtype=bool)
            self.supervised = np.concatenate([self.supervised, extra])

    def _collect_missed(self, r):
        """Collect the fraud that survived the whole defense.

        Blocked (and review-caught) fraud is already soft-removed from the
        world, so ``active & supervised & fraud`` is exactly the set the
        attacker got away with -- the material the generator learns from.
        """
        labels = self._labels()
        probs = self.pred_probs
        active = self._active_mask()
        survived_mask = (active & self.supervised & (labels == 1))
        idx = np.where(survived_mask)[0]

        feats, strategies, bases, confidences = [], [], [], []
        familiar = set()
        for i in idx:
            acct = self.world.accounts[i]
            feats.append([acct.attrs[name] for name in INTRINSIC_NAMES])
            strategies.append(acct.strategy or acct.base_strategy)
            bases.append(acct.base_strategy)
            confidences.append(float(probs[i]))
            familiar |= self._neighbours_of(i)

        self.generator.analyze(
            np.array(feats) if feats else np.zeros((0, len(INTRINSIC_NAMES))),
            strategies, bases, familiar, confidences, r,
        )

    def _neighbours_of(self, acct_idx):
        out = set()
        for s, d in self.world.referral:
            if s == acct_idx and not self.world.accounts[d].blocked:
                out.add(d)
            elif d == acct_idx and not self.world.accounts[s].blocked:
                out.add(s)
        dev = self.world.accounts[acct_idx].device_id
        for b in self.world.device_users.get(dev, set()):
            if b != acct_idx and not self.world.accounts[b].blocked:
                out.add(b)
        ip = self.world.accounts[acct_idx].ip_id
        for b in self.world.ip_users.get(ip, set()):
            if b != acct_idx and not self.world.accounts[b].blocked:
                out.add(b)
        return out

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _add_initial_fraud(self, world, n):
        names = list(INITIAL_STRATEGIES.keys())
        weights = _INITIAL_WEIGHTS
        ring_members = {name: [] for name in names}
        for _ in range(n):
            base = world.rng.choices(names, weights=weights)[0]
            template = INITIAL_STRATEGIES[base]
            attrs = world.sample_strategy_attrs(base)
            referrals = []
            if ring_members[base] and world.rng.random() < template['ring_affinity']:
                dst = world.rng.choice(ring_members[base])
                referrals.append(dst)
            spec = {
                'attrs': attrs, 'base': base, 'tags': [], 'strategy': base,
                'device': None, 'ip': None,
                'spray': template['device_spray'],
                'ip_reuse': template['ip_reuse'],
                'ring': False, 'referrals': referrals,
            }
            idx = world.add_fraud(spec, 0)
            ring_members[base].append(idx)

    def _reveal_budget(self, r):
        self._ensure_supervised_size()
        labels = self._labels()
        created = np.array([a.creation_round for a in self.world.accounts])
        active = self._active_mask()
        fresh = active & (created == r)
        new_fraud = np.where(fresh & (labels == 1))[0]
        new_genuine = np.where(fresh & (labels == 0))[0]

        pos = self.config['budget_pos']
        neg = self.config['budget_neg']
        if len(new_fraud) > 0:
            pick = self.world.np.choice(len(new_fraud),
                                        size=min(pos, len(new_fraud)),
                                        replace=False)
            self.supervised[new_fraud[pick]] = True
        if len(new_genuine) > 0:
            pick = self.world.np.choice(len(new_genuine),
                                        size=min(neg, len(new_genuine)),
                                        replace=False)
            self.supervised[new_genuine[pick]] = True

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    def report(self):
        with self.lock:
            events = list(self.events)
        return {
            'id': self.id,
            'state': self.state,
            'error': self.error,
            'config': self.config,
            'rounds': self.rounds,
            'strategy_evolution': self.strategy_evolution,
            'profile_summary': self.profile_summary,
            'adl_state': self.adl.state() if self.adl else None,
            'threshold_history': self.adl.threshold_history if self.adl else [],
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'events': events[-120:],
        }

    def snapshot(self):
        labels = self._labels() if self.world is not None else np.zeros(0, dtype=int)
        active = self.world.active_mask() if self.world is not None else np.zeros(0, dtype=bool)
        return {
            'id': self.id,
            'state': self.state,
            'rounds_done': len(self.rounds),
            'last_round': self.rounds[-1] if self.rounds else None,
            'num_nodes': int(self.world.num_nodes()) if self.world else 0,
            'active_nodes': int(active.sum()) if len(active) else 0,
            'blocked_nodes': int((~active).sum()) if len(active) else 0,
            'num_fraud': int(labels[active].sum()) if len(labels) else 0,
        }


if __name__ == '__main__':
    import json

    cfg = dict(DEFAULTS)
    cfg.update({
        'rounds': 3,
        'base_accounts': 200,
        'initial_fraud': 30,
        'genuine_per_round': 25,
        'fraud_per_round': 15,
        'gan_epochs': 60,
        'describe': 'smoke test',
    })
    sim = Simulation('smoke', cfg)
    sim._run()  # run synchronously
    report = sim.report()
    print(json.dumps({
        'state': report['state'],
        'error': report['error'],
        'n_rounds': len(report['rounds']),
        'last_metrics': report['rounds'][-1]['metrics'] if report['rounds'] else None,
        'last_defense': report['rounds'][-1]['defense'] if report['rounds'] else None,
        'threshold_history': report['threshold_history'],
        'adl_state': report['adl_state'],
        'last_gen': report['rounds'][-1]['gen'] if report['rounds'] else None,
        'profile_summary': report['profile_summary'],
    }, indent=2, default=str))
