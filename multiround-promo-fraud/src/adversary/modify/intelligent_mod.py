import torch
import random

from torch import nn
from adversary.modify.base_mod import BaseAdversarialMod
from utils.utils_func import verPrint

###########################################################
### INTELLIGENT FRAUD GENERATOR (ADD-ON)                 ###
###########################################################
# The original paper's adversary "modify" strategies (REPLAY /
# PERTURB-ABS / PERTURB-REL / MIXING) create the next-round fraud by
# copying or lightly perturbing the previously missed fraud.
#
# This add-on replaces that with an INTELLIGENT FRAUD GENERATOR that:
#   1) ANALYZES the successful (missed) fraud: it keeps a rolling
#      "fraud profile" of the features, connection degrees and familiar
#      target nodes of every fraud that escaped detection.
#   2) LEARNS / EVOLVES: it either trains a small GAN (generator vs
#      discriminator) on the successful-fraud features, or fits a
#      probabilistic model over those features. The learned generator is
#      carried across rounds so it keeps evolving as new fraud succeeds.
#   3) GENERATES NEW STRATEGIES: instead of returning duplicates it
#      samples brand-new feature vectors (new devices, new transaction
#      amounts, new timing, ...) and builds new connection patterns
#      (fraud rings / referral chains + rewiring onto familiar targets).
#   4) LOGS diagnostics (feature diversity, shift, edge composition) so
#      the whole process is observable end to end.
###########################################################

class _FraudProfile(object):
    """Rolling statistical profile of the fraud that escaped detection."""

    def __init__(self, round_window=5):
        self.round_window = max(1, int(round_window))
        self.feature_pool = []      # list of (n, feat_dim) tensors, one per round
        self.degree_pool = []       # list of (n,) degree tensors, one per round
        self.neighbor_ids = set()   # ids of existing nodes frauds typically attach to
        self.rounds_seen = 0

    def update(self, feats, degrees, neighbors):
        self.feature_pool.append(feats.float().detach().cpu().clone())
        self.degree_pool.append(degrees.float().detach().cpu().clone())
        if neighbors is not None and neighbors.numel() > 0:
            self.neighbor_ids.update(neighbors.long().cpu().tolist())
            if len(self.neighbor_ids) > 5000:
                self.neighbor_ids = set(random.sample(sorted(self.neighbor_ids), 5000))
        if len(self.feature_pool) > self.round_window:
            self.feature_pool = self.feature_pool[-self.round_window:]
            self.degree_pool = self.degree_pool[-self.round_window:]
        self.rounds_seen += 1

    def pool_features(self):
        if not self.feature_pool:
            return None
        return torch.cat(self.feature_pool, 0)

    def feature_stats(self, eps=1e-6):
        feats = self.pool_features()
        if feats is None:
            return None
        mu = feats.mean(0)
        sd = feats.std(0).clamp(min=eps)
        mn = feats.min(0).values
        mx = feats.max(0).values
        return mu, sd, mn, mx

    def degree_mean(self):
        if not self.degree_pool:
            return None
        return torch.cat(self.degree_pool, 0).float().mean().item()

    def scale_to_range(self, feats):
        stats = self.feature_stats()
        if stats is None:
            return feats
        _, _, mn, mx = stats
        rng = (mx - mn).clamp(min=1e-6)
        return feats * rng.unsqueeze(0) + mn.unsqueeze(0)


class _GenMLP(nn.Module):
    """Generator: latent noise -> realistic fraud feature vector in [0,1]."""

    def __init__(self, noise_dim, feat_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class _DiscMLP(nn.Module):
    """Discriminator: feature vector -> real/fake logit."""

    def __init__(self, feat_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_gan(gen, disc, real_feats, noise_dim, epochs=300, batch=32):
    """Standard minimax GAN training. Returns (generator loss, discriminator loss)."""
    real = real_feats.float()
    n = real.shape[0]
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    bs = max(1, min(batch, n))
    g_loss = d_loss = 0.0
    for _ in range(epochs):
        for _ in range(2):
            idx = torch.randperm(n)[:bs]
            x = real[idx]
            z = torch.randn(bs, noise_dim)
            with torch.no_grad():
                fake = gen(z)
            d_real = disc(x)
            d_fake = disc(fake)
            d_loss = (bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))) * 0.5
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
        z = torch.randn(bs, noise_dim)
        fake = gen(z)
        d_fake = disc(fake)
        g_loss = bce(d_fake, torch.ones_like(d_fake))
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
    return float(g_loss.item()), float(d_loss.item())


class IntelligentMod(BaseAdversarialMod):
    """Intelligent replacement for the naive copy-based adversary modifier."""

    def __init__(
            self,
            adver_gen_type='GAN',
            adver_gen_epochs=300,
            adver_gen_noise_dim=16,
            adver_gen_hidden=64,
            adver_gen_feat_coef=1.0,
            adver_gen_conn_coef=0.5,
            adver_gen_ring_ratio=0.5,
            adver_gen_round_window=5,
            verbose=0,
            **kwargs):
        super().__init__()
        self.verbose = verbose
        self.gen_type = str(adver_gen_type).upper()
        self.gen_epochs = int(adver_gen_epochs)
        self.noise_dim = int(adver_gen_noise_dim)
        self.hidden = int(adver_gen_hidden)
        self.feat_coef = float(adver_gen_feat_coef)
        self.conn_coef = float(adver_gen_conn_coef)
        self.ring_ratio = float(adver_gen_ring_ratio)
        self.round_window = int(adver_gen_round_window)

        self.profile = _FraudProfile(round_window=self.round_window)
        self.generator = None
        self.discriminator = None
        self.last_stats = {}
        self.last_missed_conf = None

    ####################
    ### MAIN ENTRYPOINT ##
    ####################

    def modify_seeds(self, graph, node_data, edge_data, seed_ids, modified_ids,
                     round_num=0, missed_probs=None, **kwargs):
        seed_ids = seed_ids.long()
        modified_ids = modified_ids.long()

        n = len(seed_ids)
        if n == 0:
            return node_data, edge_data, seed_ids, modified_ids

        # ---- STEP 1 : ANALYZE the successful (missed) fraud seeds ----
        feats = graph.ndata['feature'][seed_ids].float().cpu()
        neighbors = self._seed_neighbors(graph, seed_ids)
        degrees = self._seed_degrees(graph, seed_ids)
        self.profile.update(feats, degrees, neighbors)

        self.last_missed_conf = missed_probs.mean().item() if (missed_probs is not None and missed_probs.numel()) else None

        verPrint(self.verbose, 2, f'''
>> INTELLIGENT FRAUD GENERATOR (round {round_num})
   >> ANALYZING {n} successful fraud seeds...
      - feature dims      : {feats.shape[1]} | profile contains {self.profile.rounds_seen} round(s) of successful fraud
      - mean seed degree  : {degrees.float().mean().item():.2f} | familiar targets : {len(neighbors)}
      - model confidence that these are genuine : {('%.3f' % self.last_missed_conf) if self.last_missed_conf is not None else 'n/a'}''')

        # ---- STEP 2 : LEARN / EVOLVE the generator ----
        if self.gen_type == 'GAN':
            self._fit_gan(feats)
            gen_feats = self._gan_generate(feats.shape[1], n)
        else:
            gen_feats = self._prob_generate(feats.shape[1], n)

        # ---- STEP 3 : GENERATE new fraud variants (not copies) ----
        gen_feats = self._diversify(gen_feats)
        gen_feats = self.profile.scale_to_range(gen_feats).float()
        node_data['feature'] = gen_feats

        # ---- STEP 4 : BUILD new connection patterns ----
        new_edge_data, n_ext, n_int = self._build_structure(graph, seed_ids, modified_ids)

        # ---- STEP 5 : LOG diagnostics ----
        self._log_stats(graph, seed_ids, gen_feats, new_edge_data, n_ext, n_int)

        return node_data, new_edge_data, seed_ids, modified_ids

    ############################
    ### ANALYSIS / STATISTICS ###
    ############################

    @staticmethod
    def _seed_neighbors(graph, seed_ids):
        if graph.is_homogeneous:
            in_src, _, _ = graph.in_edges(seed_ids, form='all')
            _, out_dst, _ = graph.out_edges(seed_ids, form='all')
            return torch.cat([in_src, out_dst]).unique().long()
        nbrs = []
        for etype in graph.etypes:
            in_src, _, _ = graph.in_edges(seed_ids, etype=etype, form='all')
            _, out_dst, _ = graph.out_edges(seed_ids, etype=etype, form='all')
            nbrs.append(torch.cat([in_src, out_dst]))
        return torch.cat(nbrs).unique().long()

    @staticmethod
    def _seed_degrees(graph, seed_ids):
        if graph.is_homogeneous:
            return (graph.in_degrees(seed_ids) + graph.out_degrees(seed_ids)).float()
        deg = torch.zeros(len(seed_ids), dtype=torch.long)
        for etype in graph.etypes:
            deg = deg + graph.in_degrees(seed_ids, etype=etype) + graph.out_degrees(seed_ids, etype=etype)
        return deg.float()

    @staticmethod
    def _pairwise_mean_dist(feats):
        f = feats.float()
        if len(f) < 2:
            return 0.0
        diff = f.unsqueeze(0) - f.unsqueeze(1)
        dists = diff.norm(dim=-1)
        mask = ~torch.eye(len(f), dtype=torch.bool)
        return float(dists[mask].mean().item())

    ########################
    ### FEATURE GENERATION ##
    ########################

    def _fit_gan(self, feats):
        feat_dim = feats.shape[1]
        if (self.generator is None or self.generator.net[0].in_features != self.noise_dim
                or self.generator.net[-2].out_features != feat_dim):
            self.generator = _GenMLP(self.noise_dim, feat_dim, hidden=self.hidden)
            self.discriminator = _DiscMLP(feat_dim, hidden=self.hidden)
        g_loss, d_loss = _train_gan(self.generator, self.discriminator, feats,
                                    self.noise_dim, epochs=self.gen_epochs)
        self.last_stats['gen_gan_g_loss'] = round(g_loss, 5)
        self.last_stats['gen_gan_d_loss'] = round(d_loss, 5)
        verPrint(self.verbose, 3, f'   >> Training GAN: {self.gen_epochs} epochs '
                                   f'(G loss {g_loss:.4f}, D loss {d_loss:.4f})')

    def _gan_generate(self, feat_dim, n):
        if self.generator is None:
            self.generator = _GenMLP(self.noise_dim, feat_dim, hidden=self.hidden)
        with torch.no_grad():
            self.generator.eval()
            return self.generator(torch.randn(n, self.noise_dim)).cpu()

    def _prob_generate(self, feat_dim, n):
        stats = self.profile.feature_stats()
        pool = self.profile.pool_features()
        mu, sd, _, _ = stats
        mu = mu.unsqueeze(0).repeat(n, 1)
        sd = sd.unsqueeze(0).repeat(n, 1)
        gate = torch.rand(n, feat_dim)
        base = pool[torch.randint(0, pool.shape[0], (n,))] if pool is not None else mu
        drift = torch.randn(n, feat_dim) * sd
        feats = torch.where(gate < 0.5, base + drift, mu + drift)
        return torch.clamp(feats, 0, 1)

    def _diversify(self, feats):
        if self.feat_coef <= 1.0:
            return feats
        extra = torch.randn_like(feats) * (self.feat_coef - 1.0) * 0.1
        return torch.clamp(feats + extra, 0, 1)

    ###########################
    ### STRUCTURAL GENERATION ##
    ###########################

    def _build_structure(self, graph, seed_ids, modified_ids):
        n = len(modified_ids)
        new_ids_list = modified_ids.tolist()
        new_set = set(new_ids_list)

        # Target connection count per new node follows the (scaled) seed degree
        seed_deg = self._seed_degrees(graph, seed_ids)
        mu_deg = max(1.0, seed_deg.mean().item())
        cap = max(2, int(mu_deg * 2))
        targets = torch.clamp((seed_deg * self.conn_coef).round(), 1, cap).long().tolist()

        edge_data = {}
        n_ext = n_int = 0

        for etype in graph.etypes:
            if graph.is_homogeneous:
                in_src, _, _ = graph.in_edges(seed_ids, form='all')
                _, out_dst, _ = graph.out_edges(seed_ids, form='all')
            else:
                in_src, _, _ = graph.in_edges(seed_ids, etype=etype, form='all')
                _, out_dst, _ = graph.out_edges(seed_ids, etype=etype, form='all')
            familiar = set(torch.cat([in_src, out_dst]).long().tolist())
            familiar = [x for x in familiar if x not in new_set]
            fallback = list(range(graph.num_nodes()))

            in_src_l, in_dst_l, out_src_l, out_dst_l = [], [], [], []
            for i in range(n):
                t = targets[i]
                ring = (n >= 2) and (random.random() < self.ring_ratio)
                if ring:
                    # Fraud ring / referral chain : connect to sibling fraud nodes
                    guaranteed = (i + 1) % n
                    k = max(1, min(t, n - 1))
                    chosen_idx = [guaranteed] + random.sample([j for j in range(n) if j != i], k - 1)
                    chosen = [new_ids_list[j] for j in chosen_idx]
                    n_int += k
                else:
                    # Rewire onto familiar (previously attacked) genuine nodes
                    pool = familiar if familiar else fallback
                    if not pool:
                        continue
                    k = max(1, min(t, len(pool)))
                    chosen = random.sample(pool, k)
                    n_ext += k
                for c in chosen:
                    out_src_l.append(new_ids_list[i]); out_dst_l.append(c)
                    in_src_l.append(c); in_dst_l.append(new_ids_list[i])

            edge_data[etype] = {
                'in': {'src': torch.tensor(in_src_l, dtype=torch.long),
                       'dst': torch.tensor(in_dst_l, dtype=torch.long)},
                'out': {'src': torch.tensor(out_src_l, dtype=torch.long),
                        'dst': torch.tensor(out_dst_l, dtype=torch.long)},
            }

        return edge_data, n_ext, n_int

    #####################
    ### LOGGING / STATS ##
    #####################

    def _log_stats(self, graph, seed_ids, gen_feats, edge_data, n_ext, n_int):
        src_feats = graph.ndata['feature'][seed_ids].float().cpu()
        n = len(gen_feats)
        div = self._pairwise_mean_dist(gen_feats)
        shift = (gen_feats[:len(src_feats)] - src_feats).norm(dim=-1).mean().item() if n == len(src_feats) else 0.0
        n_edges = int(edge_data[list(edge_data.keys())[0]]['in']['dst'].numel())

        self.last_stats = {
            'gen_type': self.gen_type,
            'gen_seeds': int(len(seed_ids)),
            'gen_feat_div': round(div, 5),
            'gen_feat_shift': round(shift, 5),
            'gen_new_edges': n_edges,
            'gen_ext_edges': n_ext,
            'gen_ring_edges': n_int,
            'gen_ring_ratio': round(n_int / max(1, n_ext + n_int), 4),
            'gen_missed_conf': round(self.last_missed_conf, 5) if self.last_missed_conf is not None else -1,
        }

        verPrint(self.verbose, 2, f'''   >> GENERATED {n} new fraud variants:
      - feature diversity    : {self.last_stats['gen_feat_div']:.4f} (0.0 = identical copies)
      - feature shift vs seed: {self.last_stats['gen_feat_shift']:.4f} (0.0 = exact replay)
      - new edges            : {self.last_stats['gen_new_edges']} ({n_ext} external, {n_int} internal/ring)
   >> FRAUD GENERATOR EVOLVED (profile now holds {self.profile.rounds_seen} round(s) of successful fraud)''')

    def get_generation_stats(self):
        return dict(self.last_stats)
