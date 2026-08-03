"""Generator engines.

Two plug-in engines implement the "next round fraud" step:

* ``IntelligentFraudGenerator`` - analyses the fraud that escaped detection,
  learns a GAN (or probabilistic) model over its behaviour, then synthesises
  brand-new strategies (mutated devices / IPs / amounts / timing, fraud
  rings and referral chains) instead of returning copies.
* ``ReplayGenerator``         - the baseline behaviour of the original
  framework: next round's fraud is an exact duplicate of the missed fraud.
  Kept around so the dashboard can compare the two approaches.
"""

import numpy as np

from ..world import INITIAL_STRATEGIES, INTRINSIC_NAMES
from .profile import FraudProfile, _pairwise_mean_distance
from .gan import train_gan, sample_gan
from .sampler import sample_probabilistic, apply_diversity
from .mutators import mutate_spec, build_structure, row_to_attrs


class _BaseGenerator:
    name = 'base'

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.profile = FraudProfile(window=config.get('profile_window', 5))
        self.last_stats = {}

    def analyze(self, missed_rows, strategy_names, base_names, familiar,
                confidences, round_idx):
        self.profile.update(missed_rows, strategy_names, base_names,
                            familiar, confidences, round_idx)

    def generate(self, n, round_idx, world, **ctx):
        raise NotImplementedError

    def get_stats(self):
        return dict(self.last_stats)


class ReplayGenerator(_BaseGenerator):
    """Baseline: clone the missed fraud. Exact copies, no learning."""

    name = 'replay'

    def generate(self, n, round_idx, world, **ctx):
        pool = self.profile.pool()
        bases = self.profile.base_strategy_pool
        specs = []
        if pool is None or len(pool) == 0:
            # nothing missed yet -> fall back to initial templates
            for _ in range(n):
                base = self.rng.choice(list(INITIAL_STRATEGIES.keys()))
                specs.append(self._fallback_spec(base))
        else:
            for i in range(n):
                src_idx = i % len(pool)
                row = pool[src_idx]
                base = bases[src_idx] if bases else 'replay'
                specs.append(mutate_spec(row, {
                    'attrs': dict(zip(INTRINSIC_NAMES, row)),
                    'device_id': None, 'ip_id': None, 'base': base,
                }, self.rng, diversity=1.0, ring=False, victims=()))

        self.last_stats = {
            'gen_mode': 'replay',
            'gen_seeds': len(pool) if pool is not None else 0,
            'gen_feat_div': 0.0,
            'gen_feat_shift': 0.0,
            'gen_new_edges': 0,
            'gen_ring_edges': 0,
            'gen_ext_edges': 0,
            'gen_ring_ratio': 0.0,
        }
        return specs, []


    def _fallback_spec(self, base):
        template = INITIAL_STRATEGIES[base]
        means = np.array(template['means'], dtype=float)
        attrs = row_to_attrs(np.clip(means + self.rng.normal(0, 0.15, len(means)), 0, 1),
                             self.rng)
        return {
            'attrs': attrs, 'base': base, 'tags': [], 'strategy': base,
            'device': None, 'ip': None, 'spray': template['device_spray'],
            'ip_reuse': template['ip_reuse'], 'ring': False,
            'victims': [], 'referrals': [],
        }


class IntelligentFraudGenerator(_BaseGenerator):
    """The add-on: learn from successful fraud, then evolve new strategies."""

    name = 'intelligent'

    def __init__(self, config, rng):
        super().__init__(config, rng)
        self.gen_type = str(config.get('gen_type', 'GAN')).upper()
        self.gan_epochs = int(config.get('gan_epochs', 200))
        self.gan_noise_dim = int(config.get('gan_noise_dim', 12))
        self.gan_hidden = int(config.get('gan_hidden', 32))
        self.diversity = float(config.get('diversity', 1.0))
        self.conn_coef = float(config.get('conn_coef', 0.6))
        self.ring_ratio = float(config.get('ring_ratio', 0.5))
        self.generator_net = None

    # ---- learning ---------------------------------------------------------
    def _learn(self, pool):
        stats = {'gen_type': self.gen_type}
        if self.gen_type == 'GAN':
            result = train_gan(pool, noise_dim=self.gan_noise_dim,
                               hidden=self.gan_hidden, epochs=self.gan_epochs,
                               seed=int(self.config.get('seed', 0)))
            if result is None:
                stats['gen_note'] = 'GAN skipped (too few samples) -> PROB'
                return None, stats
            self.generator_net, g_loss, d_loss = result
            stats['gen_gan_g_loss'] = round(g_loss, 5)
            stats['gen_gan_d_loss'] = round(d_loss, 5)
            return self.generator_net, stats
        return None, stats

    def _sample(self, pool, n):
        if self.gen_type == 'GAN' and self.generator_net is not None:
            try:
                return sample_gan(self.generator_net, n, self.gan_noise_dim)
            except Exception:
                pass
        return sample_probabilistic(pool, n, self.rng, noise=1.0)

    # ---- generation -------------------------------------------------------
    def generate(self, n, round_idx, world, **ctx):
        pool = self.profile.pool()
        bases = self.profile.base_strategy_pool
        strategies = self.profile.strategy_pool
        familiar = sorted(self.profile.familiar)
        round_of = self.profile.round_of_row

        stats = {'gen_mode': 'intelligent', 'gen_type': self.gen_type}
        specs = []
        edges = []

        if pool is None or len(pool) == 0:
            # no prior missed fraud -> seed with mutated initial strategies
            for i in range(n):
                base = self.rng.choice(list(INITIAL_STRATEGIES.keys()))
                spec = self._evolve_from_base(base, familiar)
                specs.append(spec)
            stats['gen_seeds'] = 0
        else:
            net, learn_stats = self._learn(pool)
            if learn_stats:
                stats.update(learn_stats)
            rows = self._sample(pool, n)
            rows = apply_diversity(rows, pool, self.diversity, self.rng)
            source_pool = self._choose_sources(n, pool, bases, strategies, round_of)
            stats['gen_seeds'] = int(len(pool))

            for i in range(n):
                row = rows[i]
                src = source_pool[i]
                base = src['base']
                spray = src['spray']
                ip_reuse = src['ip_reuse']
                # decide ring participation
                ring = bool(self.rng.random() < self.ring_ratio and n > 1)
                victims = (self.rng.choice(familiar, size=min(len(familiar), 6),
                                           replace=False).tolist()
                           if familiar else [])
                spec = mutate_spec(
                    row,
                    {'attrs': src['attrs'], 'device_id': src['device_id'],
                     'ip_id': src['ip_id'], 'base': base,
                     'spray': spray, 'ip_reuse': ip_reuse},
                    self.rng, diversity=self.diversity, ring=ring, victims=victims,
                )
                specs.append(spec)

        # structure (rings / chains / victim attachment)
        edges = build_structure(specs, self.rng,
                                conn_coef=self.conn_coef,
                                ring_ratio=self.ring_ratio)

        self.last_stats = self._summarize(specs, edges, stats, pool)
        return specs, edges

    def _choose_sources(self, n, pool, bases, strategies, round_of):
        """Pick a parent (missed-fraud) account for each new sample."""
        sources = []
        for i in range(n):
            j = int(self.rng.integers(0, len(pool)))
            template = INITIAL_STRATEGIES.get(bases[j] if bases else 'fake_identity',
                                              INITIAL_STRATEGIES['fake_identity'])
            row = pool[j]
            device_id = None  # device identity is not recoverable from features
            ip_id = None
            sources.append({
                'attrs': dict(zip(INTRINSIC_NAMES, row)),
                'base': bases[j] if bases else 'evolved',
                'device_id': device_id,
                'ip_id': ip_id,
                'spray': template.get('device_spray', 0.3),
                'ip_reuse': template.get('ip_reuse', 0.4),
            })
        return sources

    def _evolve_from_base(self, base, familiar):
        template = INITIAL_STRATEGIES[base]
        means = np.array(template['means'], dtype=float)
        row = np.clip(means + self.rng.normal(0, 0.2, len(means)), 0, 1)
        victims = (self.rng.choice(familiar, size=min(len(familiar), 4),
                                   replace=False).tolist()
                   if familiar else [])
        spec = mutate_spec(row, {
            'attrs': dict(zip(INTRINSIC_NAMES, means)),
            'device_id': None, 'ip_id': None, 'base': base,
            'spray': template['device_spray'], 'ip_reuse': template['ip_reuse'],
        }, self.rng, diversity=self.diversity, ring=False, victims=victims)
        return spec

    def _summarize(self, specs, edges, stats, pool):
        rows = np.array([list(s['attrs'][k] for k in INTRINSIC_NAMES)
                         for s in specs], dtype=float)
        div = _pairwise_mean_distance(rows) if len(rows) > 1 else 0.0

        shift = 0.0
        if pool is not None and len(pool) and len(rows):
            # mean distance between generated rows and the source pool
            dists = np.sqrt(((rows[:, None, :] - pool[None, :, :]) ** 2).sum(-1))
            shift = float(dists.min(axis=1).mean())

        ring_edges = sum(1 for (a, b) in edges if b < len(specs))
        ext_edges = len(edges) - ring_edges

        from collections import Counter
        base_counts = dict(Counter(s['base'] for s in specs))

        stats.update({
            'gen_feat_div': round(float(div), 5),
            'gen_feat_shift': round(float(shift), 5),
            'gen_new_edges': int(len(edges)),
            'gen_ring_edges': int(ring_edges),
            'gen_ext_edges': int(ext_edges),
            'gen_ring_ratio': round(ring_edges / max(1, len(edges)), 4),
            'gen_strategies': dict(Counter(s['strategy'] for s in specs)),
            'gen_base_strategies': base_counts,
            'gen_missed_conf': round(float(np.mean(self.profile.confidences)), 5)
                               if self.profile.confidences else -1.0,
        })
        return stats
