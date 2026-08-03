"""Minimal GAN used to synthesise new fraud feature vectors.

Generator   : latent noise -> fraud-like behaviour vector (sigmoid, in [0,1])
Discriminator: behaviour vector -> real/fake logit

Training runs a standard minimax loop on CPU. Small networks keep every round
fast even on machines without a GPU.
"""

import numpy as np
import torch
import torch.nn as nn


class _GeneratorMLP(nn.Module):
    def __init__(self, noise_dim, feat_dim, hidden=32):
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


class _DiscriminatorMLP(nn.Module):
    def __init__(self, feat_dim, hidden=32):
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


def _train(gen, disc, real, noise_dim, epochs, batch, seed):
    torch.manual_seed(seed)
    real = real.float()
    n = real.shape[0]
    bs = max(1, min(batch, n))
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
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
            d_loss = 0.5 * (bce(d_real, torch.ones_like(d_real)) +
                            bce(d_fake, torch.zeros_like(d_fake)))
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


def train_gan(real_feats, noise_dim=12, hidden=32, epochs=200, batch=32, seed=0):
    """Fit a GAN on ``real_feats`` (array of shape (n, d)).

    Returns ``None`` if there are too few samples to be meaningful.
    Otherwise returns (generator, g_loss, d_loss).
    """
    real = np.asarray(real_feats, dtype=np.float32)
    if real.ndim == 1:
        real = real[None, :]
    if real.shape[0] < 2:
        return None
    feat_dim = real.shape[1]
    gen = _GeneratorMLP(noise_dim, feat_dim, hidden)
    disc = _DiscriminatorMLP(feat_dim, hidden)
    g_loss, d_loss = _train(gen, disc, torch.from_numpy(real), noise_dim,
                            epochs, batch, seed)
    gen.eval()
    return gen, g_loss, d_loss


def sample_gan(gen, n, noise_dim):
    with torch.no_grad():
        z = torch.randn(n, noise_dim)
        return gen(z).cpu().numpy()
