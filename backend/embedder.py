"""TPNE — Temporally Pre-trained Node Embedder, ported into the streaming app.

This restores the two contributions of

    Hafizh Adi Prasetya, Xin Liu, Tsuyoshi Murata, Akiyoshi Matono,
    "A multi-rounded adversarial scenario for graph-based promo fraud
    detection", Social Network Analysis and Mining (2026),
    doi:10.1007/s13278-025-01566-0

which the single-app rebuild had dropped entirely (see
``Research paper/_report_content_ch123.py`` §1.5 and §2.8 of the Phase I report,
which explicitly list graph embeddings as out of scope).

TPNE produces, for every account, a *static* half and a *temporal* half of an
embedding, trained self-supervised by three objectives (paper §5.2):

    L_recon   neighbourhood reconstruction — the parameterised embedding must
              preserve the average neighbourhood distance of a parameterless
              (GIN) aggregation of the raw features
    L_disent  temporal disentanglement — penalise the cosine correlation
              between the static and temporal halves, so each fraud type gets
              the information it needs
    L_temp    temporal maximisation — penalise a weak temporal signal

              final = (L_recon + a*L_disent + b*L_temp) / (1 + a + b)

Two modes are supported.

``offline``
    Faithful to the paper. The full history of ``round_window`` rounds is
    replayed and the whole embedding is recomputed from scratch each round.
    This is the reference implementation and the comparison baseline.

``incremental``
    The Phase-3 upgrade. Only the newest round is embedded, using the cached
    per-layer static embedding of the previous round as the base state, so a
    new round costs one propagation instead of ``round_window`` of them. This
    is the "incremental embedding rather than a full re-encode" that both the
    paper (§5.2) and our Phase I report (§5.2) name as the intended next step.

Dependency note
---------------
Torch, DGL and torch_geometric live in the separate ``.venv-tpne``
environment; they are NOT in ``requirements.txt`` and are NOT needed by the
rest of the backend. ``backend/features.py`` imports this module lazily and
degrades to "no embedding columns" when they are unavailable, so the existing
streaming app, its 137 tests and its 3-model ensemble all keep working in the
original environment.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import numpy as np

_LOG = logging.getLogger(__name__)

#: Loss term weights. The paper's control configuration used alpha=beta=1.
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 1.0
DEFAULT_ROUND_WINDOW = 7
DEFAULT_H_FEATS = 64
DEFAULT_NUM_LAYERS = 2
#: Guards log(0) in the reconstruction term, mirroring the author's E_CONSTANT.
_EPS = 1e-9


def torch_stack():
    """Import torch, or return ``None`` when the embedder is unavailable.

    Kept as a function so that importing this module never hard-fails the
    backend in an environment without the ML extras.
    """
    try:
        import torch  # noqa: WPS433 (deliberate optional import)
    except Exception:  # pragma: no cover - exercised only without torch
        return None
    return torch


def available() -> bool:
    """True when the embedder can actually run (torch + dgl present)."""
    if torch_stack() is None:
        return False
    try:
        import dgl  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Author's original embedder
# ---------------------------------------------------------------------------
#: Where the vendored reference implementation lives, relative to the repo root.
AUTHOR_SRC = "vendor/author-tpne/src"


class AuthorTemporalEmbedder:
    """Adapter around the author's original ``TemporalEmbedder``.

    The from-scratch reimplementation in :class:`TPNEEmbedder` turned out to be
    *weaker* than the raw node features at detecting generated fraud (38.45 vs
    53.06 on-round F1), which is the single blocker on round-0 performance and
    retention. The reference implementation is vendored in
    ``vendor/author-tpne/src``; this adapter drives it directly so the stronger
    encoder can be used without duplicating it.

    It is deliberately opt-in: importing this class does not import ``dgl`` or
    the vendored tree, so the streaming application still runs without them.
    """

    def __init__(self, h_feats=64, num_layers=2, round_window=7,
                 temporal_agg="weight", alpha=1.0, beta=1.0,
                 loss_type="ndist", device="cpu", **kwargs):
        torch = torch_stack()
        if torch is None:
            raise RuntimeError("torch is required for AuthorTemporalEmbedder")
        root = Path(__file__).resolve().parents[1]
        src = root / AUTHOR_SRC
        if not (src / "models" / "proposed_supervised" / "mixed.py").exists():
            raise RuntimeError(
                f"author source not found at {src}; clone "
                "https://github.com/hafizhadi/multiround-promo-fraud")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from models.proposed_supervised.mixed import TemporalEmbedder  # noqa: E402

        self.torch = torch
        self.h_feats = int(h_feats)
        self.num_layers = int(num_layers)
        self.round_window = int(round_window)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.device = device
        self.embed_dim = 2 * self.h_feats * (self.num_layers + 1)
        self._in_feats = None
        self._model = None
        self.last_loss = {}
        self.refit_count = 0
        self._kwargs = kwargs
        self._loss_type = loss_type
        self._temporal_agg = temporal_agg

    # ------------------------------------------------------------------ build
    def _to_dgl(self, graph):
        """Wrap a plain dict graph as the DGL graph the author's module wants.

        The reference code passes the graph itself as ``blocks`` (see
        ``src/models/base_model.py``), so no neighbour sampler is involved.
        """
        torch = self.torch
        import dgl
        feats = np.asarray(graph["features"], dtype=np.float32)
        ei = graph["edge_index"]
        n = feats.shape[0]
        g = dgl.graph((torch.as_tensor(np.asarray(ei[0], dtype=np.int64)),
                       torch.as_tensor(np.asarray(ei[1], dtype=np.int64))),
                      num_nodes=n).long()
        # the author's modules (GINConv / GraphConv) reject zero-in-degree
        # nodes, and the adversarial rewiring can strand a node
        in_deg = np.asarray(g.in_degrees())
        if np.any(in_deg == 0):
            g = dgl.add_self_loop(g)
        g.ndata["feature"] = torch.as_tensor(feats)
        age = np.asarray(graph.get("age", np.zeros(n)), dtype=np.float32)
        g.ndata["age"] = torch.as_tensor(age)
        g.ndata["label"] = torch.as_tensor(
            np.asarray(graph.get("label", np.zeros(n)), dtype=np.int64))
        return g

    def _ensure(self, in_feats):
        if self._model is None or self._in_feats != in_feats:
            from models.proposed_supervised.mixed import TemporalEmbedder
            self._model = TemporalEmbedder(
                in_feats, self.h_feats, self.num_layers,
                round_window=self.round_window,
                temporal_agg=self._temporal_agg,
                alpha=self.alpha, beta=self.beta,
                loss_type=self._loss_type, device=self.device,
                **self._kwargs)
            self._in_feats = in_feats

    # --------------------------------------------------------------- embedding
    def embed(self, graph, history=None):
        torch = self.torch
        feats = np.asarray(graph["features"], dtype=np.float32)
        self._ensure(feats.shape[1])
        g = self._to_dgl(graph)
        with torch.no_grad():
            h = self._model.embed_nodes(g, g.ndata["feature"])
        self.refit_count += 1
        return h.cpu().numpy().astype(np.float64), dict(self.last_loss)

    def pretrain(self, graph, epochs=30, lr=0.01, history=None, warm=False):
        """Delegate to the author's own forward pass, which computes the three
        losses and back-propagates them - the reference training loop lives in
        ``src/experiment/supervised_multi.py``."""
        raise NotImplementedError(
            "use vendor/author-tpne/scripts/main.py for the author's own "
            "pre-training loop; this adapter only provides inference")

    def columns(self, prefix="tpne"):
        half = self.h_feats * (self.num_layers + 1)
        return ([f"{prefix}_static_{i}" for i in range(half)]
                + [f"{prefix}_temp_{i}" for i in range(half)])

    def summary(self):
        return {"source": "author", "embed_dim": self.embed_dim,
                "h_feats": self.h_feats, "num_layers": self.num_layers,
                "round_window": self.round_window,
                "refit_count": self.refit_count}


class TPNEEmbedder:
    """Temporally pre-trained node embedder.

    Parameters mirror the author's ``TemporalEmbedder`` defaults so that
    ``offline`` mode reproduces the paper. ``mode`` selects the Phase-3
    incremental behaviour.
    """

    def __init__(self, h_feats=DEFAULT_H_FEATS, num_layers=DEFAULT_NUM_LAYERS,
                 round_window=DEFAULT_ROUND_WINDOW, alpha=DEFAULT_ALPHA,
                 beta=DEFAULT_BETA, temporal_agg="weight", mode="offline",
                 device="cpu", seed=42):
        torch = torch_stack()
        if torch is None:
            raise RuntimeError(
                "torch is not installed in this environment; run the embedder "
                "with .venv-tpne (see requirements-tpne.lock)"
            )
        self.torch = torch
        if mode not in ("offline", "incremental"):
            raise ValueError(f"unknown mode: {mode!r}")

        self.h_feats = int(h_feats)
        self.num_layers = int(num_layers)
        self.round_window = int(round_window)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temporal_agg = temporal_agg
        self.mode = mode
        self.device = device
        self.seed = int(seed)

        self.embed_dim = 2 * self.h_feats * (self.num_layers + 1)

        self._lock = threading.RLock()
        self._in_feats = None
        self._nonpar_cache = None
        self._cache: dict | None = None       # last embedding (offline)
        self._cache_key = None                # (n_nodes, n_edges, round_id)
        self._cache_h = None                # last static half, for incremental
        self.pretrained = False      # set True once pretrain() has fitted the weights
        self.pretrain_epochs = 0
        self._opt = None
        self._adj_sig = None
        self._adj_t = None
        self._adj_pattern = None
        self.refit_count = 0
        self.last_loss: dict = {}

        self._build()

    # ------------------------------------------------------------------ model
    def _build(self):
        torch = self.torch
        nn = torch.nn

        # The input width is only known once a graph is supplied; allocate the
        # projection lazily in embed() and seed it here so the parameter exists
        # from the start (matters for optimisers built before the first embed).
        self.init_weight = nn.Parameter(
            torch.zeros(1, self.h_feats, device=self.device))
        nn.init.xavier_uniform_(self.init_weight)


        # GraphConv layers (paper uses GraphConv with norm='none')
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = nn.Module()
            layer.linear = nn.Linear(self.h_feats, self.h_feats, bias=False)
            nn.init.xavier_uniform_(layer.linear.weight)
            self.gnn_layers.append(layer)

        if self.temporal_agg == "weight":
            self.temporal_weight = nn.Linear(
                self.h_feats * (self.num_layers + 1) * self.round_window,
                self.h_feats * (self.num_layers + 1))
            nn.init.xavier_uniform_(self.temporal_weight.weight)
        else:
            self.temporal_weight = None

        self._to(self.device)

    def _to(self, device):
        self.init_weight.data = self.init_weight.data.to(device)
        for layer in self.gnn_layers:
            layer.linear.weight.data = layer.linear.weight.data.to(device)
        if self.temporal_weight is not None:
            self.temporal_weight.weight.data = \
                self.temporal_weight.weight.data.to(device)

    # ------------------------------------------------------------ torch path
    def _params(self):
        """Trainable parameters, in a stable order."""
        ps = [self.init_weight]
        ps += [layer.linear.weight for layer in self.gnn_layers]
        if self.temporal_weight is not None:
            ps.append(self.temporal_weight.weight)
        return ps

    def _adjacency(self, graph, n):
        """Row-normalised adjacency with self-loops, as a dense tensor.

        Also returns the CSR pattern ``(row, col, val)`` so the edge-wise
        distance term can reproduce this normalisation without materialising a
        dense n-by-n matrix.

        The dense tensor is cached per (n, edge-signature): at 11,758 nodes it
        is 553 MB, and rebuilding it for every window view of every pre-training
        epoch exhausted memory. The pattern is what the loss actually uses, so
        the cache is invalidated whenever the edge set changes between rounds.
        """
        torch = self.torch
        import scipy.sparse as sp
        row, col = graph["edge_index"][0], graph["edge_index"][1]
        keep = (row < n) & (col < n)
        row, col = row[keep], col[keep]
        sig = (n, len(row), int(row[:64].sum()) if len(row) else 0,
               int(col[:64].sum()) if len(col) else 0)
        if getattr(self, "_adj_sig", None) == sig and self._adj_t is not None:
            return self._adj_t, self._adj_pattern

        A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
        A = A + sp.eye(n, format="csr")
        deg = np.asarray(A.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        A = (sp.diags(1.0 / deg) @ A).tocsr()
        coo = A.tocoo()
        pattern = (coo.row, coo.col, coo.data)
        idx = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long,
                           device=self.device)
        val = torch.tensor(coo.data.astype(np.float32), device=self.device)
        # Sparse rather than dense: the dense form is 553 MB at 11,758 nodes
        # and made multi-epoch pre-training impossible on 7 GB of RAM.
        A_t = torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()
        self._adj_sig = sig
        self._adj_t = A_t
        self._adj_pattern = pattern
        return A_t, pattern

    def _propagate_t(self, graph, feats):
        """Differentiable propagation, mirroring ``_propagate_np`` (sparse)."""
        torch = self.torch
        n = feats.shape[0]
        A_t, _ = self._adjacency(graph, n)
        h = self._norm_t(feats)
        h = self._leaky_t(h)
        layers = [h]
        for i in range(self.num_layers):
            h = torch.sparse.mm(A_t, h) @ self.gnn_layers[i].linear.weight.T
            h = self._norm_t(h)
            h = self._leaky_t(h)
            layers.append(h)
        return torch.cat(layers, dim=1), A_t

    @staticmethod
    def _norm_t(x):
        return x / (x.norm(p=2, dim=-1, keepdim=True) + _EPS)

    def _leaky_t(self, x, slope=0.01):
        torch = self.torch
        return torch.where(x > 0, x, slope * x)

    def _nonparam_t(self, graph, feats):
        """Parameterless (GIN-style) reference embedding, differentiable path."""
        torch = self.torch
        n = feats.shape[0]
        A_t, pattern = self._adjacency(graph, n)
        h = self._norm_t(feats)
        layers = [h]
        for _ in range(self.num_layers):
            h = self._norm_t(torch.sparse.mm(A_t, h))
            layers.append(h)
        return torch.cat(layers, dim=1), A_t, pattern

    def _mean_dist_t(self, A_t, h, pattern=None):
        """Mean neighbour distance, computed EDGE-WISE over the CSR pattern.

        A dense ``cdist`` is numerically identical but allocates an ``n x n``
        matrix: 553 MB at 11,758 nodes, which OOM-killed the first
        pre-training run on 7 GB of RAM. This gathers only the non-zero
        entries of the adjacency, so the cost is O(E * d) and memory is O(E).

        ``pattern`` is ``(row, col, val)`` from the same normalised adjacency
        as ``A_t``. The stored values matter: ``csr_matrix`` sums the two
        orientations of an undirected edge into a single entry of weight 2, so
        assuming unit weights would halve the result.
        """
        torch = self.torch
        n = h.shape[0]
        if pattern is None:
            return torch.ones(n, 1, dtype=h.dtype, device=h.device)
        row, col, val = pattern
        row_t = torch.as_tensor(row, dtype=torch.long, device=h.device)
        col_t = torch.as_tensor(col, dtype=torch.long, device=h.device)
        val_t = torch.as_tensor(val, dtype=h.dtype, device=h.device)
        diff = h.index_select(0, row_t) - h.index_select(0, col_t)
        edge_dist = diff.pow(2).sum(dim=1).clamp_min(_EPS).sqrt()
        acc = torch.zeros(n, dtype=h.dtype, device=h.device)
        acc = acc.index_add(0, row_t, val_t * edge_dist)
        return acc.unsqueeze(1)

    # ------------------------------------------------------------- numpy path
    def _mean_aggregate(self, edge_index, h):
        """Mean of neighbours, with self-loops so isolated nodes stay defined."""
        import scipy.sparse as sp
        n = h.shape[0]
        row, col = edge_index[0], edge_index[1]
        # an adversarial graph can carry rewired edges pointing at ids the
        # feature matrix does not cover yet; clamp rather than crash
        keep = (row < n) & (col < n)
        row, col = row[keep], col[keep]
        data = np.ones(len(row), dtype=h.dtype)
        a = sp.csr_matrix((data, (row, col)), shape=(n, n))
        a = a + sp.eye(n, format="csr", dtype=h.dtype)
        deg = np.asarray(a.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        return (sp.diags(1.0 / deg) @ a) @ h

    def _propagate_np(self, edge_index, feats):
        """Parameterless (GIN-style, no learned weights) reference embedding."""
        h = feats.copy()
        layers = [h]
        for _ in range(self.num_layers):
            h = self._mean_aggregate(edge_index, h)
            h = self._normalize_np(h)
            layers.append(h)
        return np.concatenate(layers, axis=1)

    def _recon_loss(self, graph, h_static):
        """Neighbourhood-distance preservation (paper Eq. 9), edge-wise."""
        layer = h_static.shape[1] // (self.num_layers + 1)
        edge_index = graph["edge_index"]
        total = 0.0
        for i in range(self.num_layers + 1):
            sl = slice(i * layer, (i + 1) * layer)
            d_par = self._mean_neighbour_dist(edge_index, h_static[:, sl])
            d_nonpar = self._mean_neighbour_dist(edge_index,
                                                 self._nonpar_cache[:, sl])
            total += float(np.mean((np.log(d_par + _EPS)
                                    - np.log(d_nonpar + _EPS)) ** 2))
        return total / (self.num_layers + 1)

    @staticmethod
    def _mean_neighbour_dist(edge_index, h):
        n = h.shape[0]
        src, dst = edge_index[0], edge_index[1]
        keep = (src < n) & (dst < n)
        src, dst = src[keep], dst[keep]
        if len(src) == 0:
            return np.ones(n, dtype=h.dtype)
        diff = h[src] - h[dst]
        edge_dist = np.sqrt((diff * diff).sum(axis=1))
        import scipy.sparse as sp
        a = sp.csr_matrix((np.ones(len(src)), (src, np.zeros(len(src),
                                                         dtype=int))),
                          shape=(n, len(src)))
        deg = np.asarray(a.sum(axis=1)).ravel()
        deg[deg == 0] = 1.0
        return np.asarray(a @ edge_dist).ravel() / deg

    @staticmethod
    def _disent_loss(h_static, h_temp):
        num = (h_static * h_temp).sum(axis=1)
        den = (np.linalg.norm(h_static, axis=1)
               * np.linalg.norm(h_temp, axis=1)) + _EPS
        return float(1.0 - np.mean(num / den))

    @staticmethod
    def _temp_loss(h_temp):
        norm = np.linalg.norm(h_temp, axis=1).mean()
        return float(1.0 / np.log(norm + 2.718281828459045 + _EPS))

    def _window_features(self, feats, age, window):
        n = feats.shape[0]
        return [(feats * (age >= w).astype(np.float64)[:, None])
                for w in range(window)]

    def _embed_offline(self, feats, age, edge_index, history):
        """Paper behaviour: recompute every window slice from scratch."""
        window = self.round_window
        views = history if history else self._window_features(feats, age, window)
        views = views[-window:]
        while len(views) < window:
            views.insert(0, np.zeros_like(feats))
        per_round = []
        for v in views:
            h = v @ self.init_weight.detach().cpu().numpy()
            h = self._leaky_np(self._normalize_np(h))
            per_round.append(self._propagate_np(edge_index, h))
        if self.temporal_agg == "weight" and self.temporal_weight is not None:
            h_temp = np.concatenate(per_round, axis=1) @ \
                self.temporal_weight.weight.detach().cpu().numpy().T
        else:
            diffs = [per_round[0] - per_round[i] for i in range(1, len(per_round))]
            h_temp = (np.mean(diffs, axis=0) if diffs
                      else np.zeros_like(per_round[0]))
        return per_round[0], h_temp

    def _embed_incremental(self, feats, age, edge_index):
        """Phase 3: propagate only the newest round, reusing cached state."""
        h = feats @ self.init_weight.detach().cpu().numpy()
        h = self._leaky_np(self._normalize_np(h))
        h_new = self._propagate_np(edge_index, h)
        if self._cache_h is not None and h_new.shape == self._cache_h.shape:
            h_temp = h_new - self._cache_h
        else:
            h_temp = np.zeros_like(h_new)
        self._cache_h = h_new
        return h_new, h_temp

    # -------------------------------------------------------------- embedding
    def embed(self, graph, history=None):
        """Embed a graph and return ``(embedding, loss_dict)``."""
        feats = np.asarray(graph["features"], dtype=np.float64)
        self._ensure_input_dim(feats.shape[1])
        age = np.asarray(graph.get("age", np.zeros(len(feats))), dtype=np.float64)
        edge_index = graph["edge_index"]

        with self._lock:
            if self.pretrained:
                emb, loss = self.embed_trained(graph, history)
                self._cache = emb
                self.refit_count += 1
                return emb, loss
            if self.mode == "offline":
                h_static, h_temp = self._embed_offline(feats, age, edge_index,
                                                        history)
            else:
                h_static, h_temp = self._embed_incremental(feats, age, edge_index)
            self._nonpar_cache = self._propagate_np(edge_index, feats)
            r = self._recon_loss(graph, h_static)
            d = self._disent_loss(h_static, h_temp)
            t = self._temp_loss(h_temp)
            denom = 1.0 + self.alpha + self.beta
            final = (r + self.alpha * d + self.beta * t) / denom
            emb = np.concatenate([h_static, h_temp], axis=1)
            self._cache = emb
            self.refit_count += 1
            self.last_loss = {"recon": r, "disen": d, "temp": t, "final": final}
        return emb, self.last_loss

    def _ensure_input_dim(self, d):
        if self._in_feats == d:
            return
        nn = self.torch.nn
        self._in_feats = d
        self.init_weight = nn.Parameter(
            self.torch.zeros(d, self.h_feats, device=self.device))
        nn.init.xavier_uniform_(self.init_weight)
        self._to(self.device)

    # ------------------------------------------------------------ pretraining
    def pretrain(self, graph, epochs=30, lr=0.01, history=None, warm=False):
        """Gradient-based self-supervised pre-training on the paper's losses.

        Minimises ``L_recon``, ``L_disent`` and ``L_temp`` with Adam so the
        losses *shape* the representation, as in the original TPNE, instead of
        merely scoring it (which is all the numpy path can do).

        ``warm=True`` keeps the existing optimiser state and skips re-seeding,
        which is what makes the incremental arm cost one optimisation pass per
        round rather than a fresh full re-encode.
        """
        torch = self.torch
        feats = np.asarray(graph["features"], dtype=np.float64)
        self._ensure_input_dim(feats.shape[1])
        x = torch.tensor(feats, dtype=torch.float32, device=self.device)
        age_t = torch.tensor(
            np.asarray(graph.get("age", np.zeros(len(feats))), dtype=np.float64),
            dtype=torch.float32, device=self.device)

        if not warm or self._opt is None:
            self._opt = torch.optim.Adam(self._params(), lr=lr)
        opt = self._opt
        last = {}
        for _ in range(epochs):
            opt.zero_grad()
            h_static, h_temp = self._forward_t(graph, x, age_t, history)
            nonpar, A_t, pattern = self._nonparam_t(graph, x)
            layer = h_static.shape[1] // (self.num_layers + 1)
            r = 0.0
            for i in range(self.num_layers + 1):
                sl = slice(i * layer, (i + 1) * layer)
                d_par = self._mean_dist_t(A_t, h_static[:, sl], pattern)
                d_nonpar = self._mean_dist_t(A_t, nonpar[:, sl], pattern)
                r = r + ((torch.log(d_par + _EPS)
                          - torch.log(d_nonpar + _EPS)) ** 2).mean()
            r = r / (self.num_layers + 1)
            num = (h_static * h_temp).sum(dim=1)
            den = (h_static.norm(p=2, dim=1)
                   * h_temp.norm(p=2, dim=1)) + _EPS
            d = 1.0 - (num / den).mean()
            t = 1.0 / torch.log(
                h_temp.norm(p=2, dim=1).mean() + 2.718281828459045 + _EPS)
            denom = 1.0 + self.alpha + self.beta
            loss = (r + self.alpha * d + self.beta * t) / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params(), 5.0)
            opt.step()
            last = {"recon": float(r), "disen": float(d), "temp": float(t),
                    "final": float(loss)}
        self.last_loss = last
        self.pretrain_epochs = epochs
        self.pretrained = True
        return last

    def _forward_t(self, graph, x, age_t, history=None):
        """Differentiable window embedding returning (static, temporal) halves."""
        torch = self.torch
        window = self.round_window
        if history:
            views = [torch.tensor(np.asarray(h), dtype=torch.float32,
                                  device=self.device) for h in history]
        else:
            views = [x * (age_t >= w).float().unsqueeze(1) for w in range(window)]
        while len(views) < window:
            views.insert(0, torch.zeros_like(x))
        views = views[-window:]
        per_round = []
        for v in views:
            h = self._leaky_t(self._norm_t(v @ self.init_weight))
            per_round.append(self._propagate_t(graph, h)[0])
        if self.temporal_agg == "weight" and self.temporal_weight is not None:
            h_temp = self.temporal_weight(torch.cat(per_round, dim=1))
        else:
            diffs = [per_round[0] - per_round[i] for i in range(1, len(per_round))]
            h_temp = torch.stack(diffs).mean(dim=0) if diffs \
                else torch.zeros_like(per_round[0])
        return per_round[0], h_temp

    def embed_trained(self, graph, history=None):
        """Produce the final embedding from the trained weights."""
        torch = self.torch
        feats = np.asarray(graph["features"], dtype=np.float64)
        x = torch.tensor(feats, dtype=torch.float32, device=self.device)
        age_t = torch.tensor(
            np.asarray(graph.get("age", np.zeros(len(feats))), dtype=np.float64),
            dtype=torch.float32, device=self.device)
        with torch.no_grad():
            h_static, h_temp = self._forward_t(graph, x, age_t, history)
            emb = torch.cat([h_static, h_temp], dim=1).cpu().numpy()
        return emb, dict(self.last_loss)

    @staticmethod
    def _normalize_np(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + _EPS)

    @staticmethod
    def _leaky_np(x, slope=0.01):
        return np.where(x > 0, x, slope * x)

    # ---------------------------------------------------------------- output
    def columns(self, prefix="tpne"):
        """Feature column names this embedder contributes to a source."""
        half = self.h_feats * (self.num_layers + 1)
        return ([f"{prefix}_static_{i}" for i in range(half)]
                + [f"{prefix}_temp_{i}" for i in range(half)])

    def summary(self) -> dict:
        return {
            "available": available(),
            "mode": self.mode,
            "embed_dim": self.embed_dim,
            "h_feats": self.h_feats,
            "num_layers": self.num_layers,
            "round_window": self.round_window,
            "alpha": self.alpha,
            "beta": self.beta,
            "temporal_agg": self.temporal_agg,
            "refit_count": self.refit_count,
            "last_loss": dict(self.last_loss),
        }