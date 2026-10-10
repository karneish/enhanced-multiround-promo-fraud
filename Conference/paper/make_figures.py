#!/usr/bin/env python
"""Figure generation for the AMDL-TPNE paper (IEEE/LaTeX style, matching the reference PDF)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.linewidth": 0.6,
})


def box(ax, x, y, w, h, text="", fc="#FFFFFF", fs=6.2, lw=0.7, bold=False,
        square=False):
    style = "square,pad=0.004" if square else "round,pad=0.006,rounding_size=0.012"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                                edgecolor="#000000", facecolor=fc, zorder=2))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                zorder=3, linespacing=1.4, fontweight="bold" if bold else "normal")


def arrow(ax, p1, p2, lw=0.7, rad=0.0, ls="-", style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=6.5,
                                 linewidth=lw, color="#000000", zorder=1,
                                 connectionstyle=f"arc3,rad={rad}",
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def label(ax, x, y, text, fs=5.9, rot=0, ha="center", va="center"):
    ax.text(x, y, text, fontsize=fs, ha=ha, va=va, linespacing=1.35,
            rotation=rot, style="italic")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(os.path.join(OUT, name), dpi=400)
    plt.close(fig)


# ----------------------------------------------------------------- Figure 1
def figure1():
    fig, ax = canvas(3.45, 2.55)

    box(ax, 0.02, 0.885, 0.96, 0.10,
        "Detection Strategies for Multi-Round\nAdversarial Promo Fraud",
        fc="#D9D9D9", fs=7.0, bold=True)

    cols = [
        ("Baseline\ndetectors",
         ["Random dispatch", "Greedy by size", "Single XGBoost\nhead (parity)"]),
        ("Representation\nstrategies",
         ["Raw / static\nfeatures", "Offline TPNE\n(window replay)", "Incremental TPNE\n(round delta)"]),
        ("Adaptive multi-\nmodel layer",
         ["XGBoost\n(warm-started)", "HistGradient\nBoosting", "ExtraTrees\n(high diversity)"]),
        ("Encoder\nadaptation",
         ["Frozen encoder,\nper-round refit", "Gradient\npre-training", "Cumulative\nsupport graph"]),
    ]

    x0, gap = 0.02, 0.024
    bw = (0.96 - 3 * gap) / 4
    hdr_y, hdr_h = 0.795, 0.068
    top, bh, bgap = 0.735, 0.150, 0.030

    for i, (title, items) in enumerate(cols):
        x = x0 + i * (bw + gap)
        box(ax, x, hdr_y, bw, hdr_h, title, fc="#E8E8E8", fs=5.7, bold=True)
        arrow(ax, (x + bw / 2, 0.885), (x + bw / 2, hdr_y + hdr_h), lw=0.6)
        for j, it in enumerate(items):
            y = top - (j + 1) * bh - j * bgap
            box(ax, x, y, bw, bh, it, fs=5.7)
            if j < len(items) - 1:
                arrow(ax, (x + bw / 2, y), (x + bw / 2, y - bgap), lw=0.55)

    box(ax, x0, 0.015, 0.96, 0.105,
        "Shared challenge: a decision rule tuned at round $t$ degrades at round $t{+}k$,\n"
        "because an adaptive adversary re-targets the nodes the detector missed",
        fc="#F2F2F2", fs=5.8, square=True)

    save(fig, "fig1_taxonomy.png")


# ----------------------------------------------------------------- Figure 2
def figure2():
    fig, ax = canvas(3.45, 4.15)

    L, R = 0.02, 0.925           # content band; (R, 1) is the feedback channel
    FULL = R - L
    cw = (FULL - 3 * 0.02) / 4   # 4-column band width

    box(ax, L, 0.905, FULL, 0.075,
        "Adaptive Multi-Round Adversarial Scenario",
        fc="#D9D9D9", fs=6.5, bold=True, square=True)

    scen = [
        ("Seed\nselection", "random\ngreedy\ngreedy-\nlimited"),
        ("Node\nmodification", "replay\nrelative\nabsolute\nmixing"),
        ("Support\naugmentation", "none\nrandom\nreage"),
        ("Round budget\n(5% pos / 5% neg)", "$n_q{=}587$\nper round\n$t{=}1,2,3$"),
    ]
    hdr_y, hdr_h = 0.825, 0.052
    body_y, body_h = 0.645, 0.155
    for i, (t, items) in enumerate(scen):
        x = L + i * (cw + 0.02)
        box(ax, x, hdr_y, cw, hdr_h, t, fs=5.5, bold=True, square=True)
        box(ax, x, body_y, cw, body_h, items, fs=5.7, square=True)
        arrow(ax, (x + cw / 2, body_y + body_h), (x + cw / 2, hdr_y), lw=0.55)

    gs_y, gs_h = 0.455, 0.115
    gw = (FULL - 2 * 0.01) / 3
    graphs = [("$G_t$\ntest graph (eval)"), ("$G_t^{Q}$\nquery graph (new)"),
              ("$G_t^{S}$\nsupport graph (train)")]
    for i, t in enumerate(graphs):
        box(ax, L + i * (gw + 0.01), gs_y, gw, gs_h, t, fc="#EDEDED", fs=5.9)
    for i, fx_ in enumerate((0.12, 0.50, 0.88)):
        arrow(ax, (L + i * (cw + 0.02) + cw / 2, body_y),
              (fx_, gs_y + gs_h), lw=0.6, rad=(i - 1) * 0.12)

    # TPNE (left block)
    TW = 0.445
    box(ax, L, 0.325, TW, 0.075, "TPNE  Temporally Pre-trained\nNode Embedder",
        fc="#D9D9D9", fs=5.9, bold=True, square=True)

    ty, th = 0.185, 0.115
    sw = (TW - 2 * 0.009) / 3
    tpne = ["$h^{static}$\n192-d",
            "$h^{temp}$\n192-d\n(window 7)",
            r"$\mathcal{L}_{recon}$" "\n" r"$+\alpha\mathcal{L}_{disent}$"
            "\n" r"$+\beta\mathcal{L}_{temp}$"]
    for i, t in enumerate(tpne):
        x = L + i * (sw + 0.009)
        box(ax, x, ty, sw, th, t, fs=5.7)
        arrow(ax, (x + sw / 2, 0.325), (x + sw / 2, ty + th), lw=0.55)

    box(ax, L, 0.045, TW, 0.105,
        "concat $\\rightarrow$ 384 embedding columns\n+ 27 profile / native feature columns",
        fc="#F2F2F2", fs=5.9, square=True)
    arrow(ax, (L + TW / 2, ty), (L + TW / 2, 0.15), lw=0.6)

    # AMDL (right block)
    AL, AW = 0.48, 0.445
    box(ax, AL, 0.325, AW, 0.075, "AMDL  Adaptive Multi-Model\nDetection Layer",
        fc="#D9D9D9", fs=5.9, bold=True, square=True)

    models = ["XGBoost\nhist, d = 6", "HistGrad\nBoosting", "ExtraTrees\n200 trees"]
    mcx = []
    for i, t in enumerate(models):
        x = AL + i * (sw + 0.009)
        box(ax, x, ty, sw, th, t, fs=5.7)
        arrow(ax, (x + sw / 2, 0.325), (x + sw / 2, ty + th), lw=0.55)
        mcx.append(x + sw / 2)

    box(ax, AL, 0.045, AW, 0.105,
        r"$w_i \propto \max(s_i,\ \kappa s^{\max})$, $\ \kappa{=}0.05$"
        "\n" r"review $\geq$ 0.30  /  block $\geq$ 0.70",
        fc="#F2F2F2", fs=5.9, square=True)
    for i, x in enumerate(mcx):
        arrow(ax, (x, ty), (AL + AW / 2 + (i - 1) * 0.06, 0.15), lw=0.55,
              rad=(i - 1) * 0.25)

    arrow(ax, (L + TW, 0.0975), (AL, 0.0975), lw=0.7)

    # feedback loop, routed in the reserved right-hand channel
    fx = 0.968
    arrow(ax, (AL + AW, 0.3625), (fx, 0.3625), lw=0.7, ls=(0, (3, 2)), style="-")
    arrow(ax, (fx, 0.3625), (fx, 0.9425), lw=0.7, ls=(0, (3, 2)), style="-")
    arrow(ax, (fx, 0.9425), (L + FULL, 0.9425), lw=0.7, ls=(0, (3, 2)))
    label(ax, fx - 0.022, 0.66, "miss-set feedback", fs=5.4, rot=90)

    save(fig, "fig2_architecture.png")


# ----------------------------------------------------------------- Figure 3
def figure3():
    fig, ax = canvas(3.45, 4.30)

    steps = [
        ("Round-$t$ state", "$G_t$, $G_t^{Q}$, $G_t^{S}$ + seed store"),
        ("Adversarial expansion",
         "greedy miss-targeting;\nreplay / relative / absolute / mixing"),
        ("TPNE forward pass",
         "offline (window replay) or\nincremental (round-over-round delta)"),
        ("AMDL scoring", "three warm-started heads,\nadaptive $F_1$ weighting"),
        ("Decision", "below review $\\rightarrow$ pass;\nabove block $\\rightarrow$ flag"),
        ("Execution & scoring",
         r"$RT\!\cdot\!f_{1,0}$, $RT\!\cdot\!avg\,f_{1}$, $RT\!\cdot\!recall_{0}$"),
        ("Support accumulation", "$G_t^{S} \\leftarrow G_t^{S} \\cup G_t^{Q}$"),
    ]

    bh, vgap = 0.118, 0.019
    bw, x = 0.50, 0.235
    y = 0.945
    tops = []
    for title, sub in steps:
        box(ax, x, y - bh, bw, bh, "", square=True)
        ax.text(x + bw / 2, y - 0.028, title, ha="center", va="center",
                fontsize=6.2, fontweight="bold", zorder=4)
        ax.text(x + bw / 2, y - 0.076, sub, ha="center", va="center",
                fontsize=5.3, zorder=4, linespacing=1.30)
        tops.append(y)
        y -= (bh + vgap)

    cx = x + bw / 2
    for i in range(len(steps) - 1):
        arrow(ax, (cx, tops[i] - bh), (cx, tops[i + 1]), lw=0.7)

    notes = [
        (1, "adversary acts on\nthe current miss set"),
        (2, "representation is a\nconfig switch, not\na retrain"),
        (3, "warm start resumes\nthe previous booster\n(200 + 50 rounds)"),
        (5, "metrics recomputed\nper round, not only\nat the end of a run"),
    ]
    nx = 0.880
    for i, txt in notes:
        yy = tops[i] - bh / 2
        label(ax, nx, yy, txt, fs=5.3)
        arrow(ax, (x + bw, yy), (nx - 0.125, yy), lw=0.6)

    fy = tops[0] + 0.048
    bx = 0.055
    arrow(ax, (cx, tops[-1] - bh), (bx, tops[-1] - bh), lw=0.7)
    arrow(ax, (bx, tops[-1] - bh), (bx, fy), lw=0.7)
    arrow(ax, (bx, fy), (x, fy), lw=0.7)
    label(ax, bx + 0.032, (tops[-1] - bh + fy) / 2,
          "feedback: predicted labels\nre-target the adversary", fs=5.3, rot=90)

    save(fig, "fig3_workflow.png")


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    print("figures written to", OUT)