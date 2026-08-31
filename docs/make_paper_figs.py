"""Render three figures that MIRROR Gladstone, Ji & Du, "Explorative Modeling"
(https://explorative-modeling.github.io/), using our FrameFlow results:

  fig_xm_mode_capture.png  <- their Fig. 1 (GT + progressive XM-K point clouds:
                              "increasing exploration lets a model capture more
                              modes instead of averaging them")
  fig_xm_monotonic.png     <- their Figs. 5-6 (a metric vs exploration K, which
                              they show improving monotonically)
  fig_xm_arm_study.png     <- their Figs. 5-6 & 12 (designability vs K, and the
                              "end-to-end stepping" designability-vs-steps view)

Our results invert the paper's toy: XM-1 already captures every mode and the
curves are flat, so these panels are the paper's figures run on a target where
the mean-collapse premise does not hold.

Run: PYTHONPATH=. .venv/bin/python docs/make_paper_figs.py
Needs xm_eval_results/paper_figs/clouds.npz (from analysis/xm_paper_figs.py).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# validated light palette + 5-mode categorical order (blue, orange, green, purple, gold)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'
# 5-mode categorical order. Clears the hard normal-vision floor (ΔE 16.9);
# the marginal CVD-floor trip (olive↔orange ΔE 5.7 protan) is legalized by the
# strong secondary encoding here — every mode is a spatially separated cluster
# anchored by a labeled star, so identity is read from position, not hue alone.
MODE_C = ['#2a78d6', '#eb6834', '#1baf7a', '#c83e93', '#8f6d1f']

ASSET = 'docs/assets'
os.makedirs(ASSET, exist_ok=True)


# --------------------------------------------------------------------------- #
# Fig 1 mirror: mode-capture point clouds
# --------------------------------------------------------------------------- #
def fig_mode_capture():
    d = np.load('xm_eval_results/paper_figs/clouds.npz', allow_pickle=True)
    cols = list(d['cols'])
    labels = ['Ground truth'] + [c for c in cols[1:]]
    rows = [('toy', 'Synthetic SE(3)$^N$ mixture (4 modes)', int(d['toy_M'])),
            ('real', '5 real PDB folds (length 129)', int(d['real_M']))]

    fig, axes = plt.subplots(2, len(cols), figsize=(3.0 * len(cols), 6.2))
    fig.patch.set_facecolor(SURFACE)
    for r, (tag, row_title, M) in enumerate(rows):
        tmpl = d[f'{tag}_tmpl_xy']; blur = d[f'{tag}_blur_xy']
        # common limits from the ground-truth cloud + template spread
        gt = d[f'{tag}_GT_xy']
        pad = 0.18 * (gt.max(0) - gt.min(0) + 1e-6)
        lo = np.minimum(gt.min(0), tmpl.min(0)) - pad
        hi = np.maximum(gt.max(0), tmpl.max(0)) + pad
        for c, col in enumerate(cols):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            xy = d[f'{tag}_{col}_xy']; asg = d[f'{tag}_{col}_assign']
            # mode centres drawn FIRST as hollow rings so they frame each mode
            # without occluding the (very tight) sample cluster that sits on it
            ax.scatter(tmpl[:, 0], tmpl[:, 1], s=340, marker='o',
                       facecolors='none', edgecolors=MODE_C[:M], linewidths=1.6,
                       zorder=3)
            # sample clouds ON TOP, coloured by nearest mode
            for m in range(M):
                pts = xy[asg == m]
                ax.scatter(pts[:, 0], pts[:, 1], s=14, c=MODE_C[m],
                           alpha=0.55, edgecolors='white', linewidths=0.15,
                           zorder=5)
            # the mean/"blur" point (black x) — the spot XM-1 is supposed to pile onto
            ax.scatter(blur[:, 0], blur[:, 1], s=150, marker='X', c='#111111',
                       edgecolors='white', linewidths=1.0, zorder=6)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if r == 0:
                ax.set_title(labels[c], color=INK, fontsize=12)
        axes[r, 0].set_ylabel(row_title, color=INK, fontsize=11)

    fig.suptitle('Increasing exploration (K) — does the model capture more modes '
                 'instead of averaging them?\n'
                 'Rings = mode centres · black ✕ = the mean ("blur") point · '
                 'points coloured by nearest mode. '
                 'XM-1 (vanilla) already covers every mode; no collapse for K to fix.',
                 color=INK, fontsize=12.5, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = f'{ASSET}/fig_xm_mode_capture.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    print('wrote', out)


# --------------------------------------------------------------------------- #
# Figs 5-6 mirror: a metric vs exploration K (they show monotonic improvement)
# --------------------------------------------------------------------------- #
def _median_by_k(csv, col, ks):
    df = pd.read_csv(csv)
    out = []
    for k in ks:
        v = df[df.K == k][col].values.astype(float)
        v = v[np.isfinite(v)]
        out.append(np.median(v) if len(v) else np.nan)
    return np.array(out)


def fig_monotonic():
    ks = [1, 2, 4, 8]
    toy = 'xm_eval_results/toy/toy_results.csv'
    real = 'xm_eval_results/real_folds/real_results.csv'
    x = np.arange(len(ks))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    fig.patch.set_facecolor(SURFACE)

    # left: blur rate vs K (the exact mean-collapse failure XM targets)
    ax = axes[0]; ax.set_facecolor(SURFACE)
    for csv, lab, col in [(toy, 'synthetic toy', BLUE), (real, 'real PDB folds', ORANGE)]:
        if os.path.exists(csv):
            ax.plot(x, _median_by_k(csv, 'blur_rate', ks), '-o', color=col,
                    lw=2, ms=7, label=lab, zorder=3)
    ax.set_title('Blur rate vs exploration K  (the collapse XM targets)',
                 color=INK, fontsize=12, loc='left')
    ax.set_ylabel('fraction of samples nearer the mean than any mode', color=MUTED, fontsize=9)
    ax.set_ylim(-0.03, 1.0)
    ax.text(0.5, 0.5, 'flat at 0.000 — no collapse to remove',
            transform=ax.transAxes, ha='center', color=MUTED, fontsize=11, style='italic')

    # right: sample purity vs K (does exploration sharpen samples?)
    ax = axes[1]; ax.set_facecolor(SURFACE)
    for csv, lab, col in [(toy, 'synthetic toy', BLUE), (real, 'real PDB folds', ORANGE)]:
        if os.path.exists(csv):
            ax.plot(x, _median_by_k(csv, 'purity_comb', ks), '-o', color=col,
                    lw=2, ms=7, label=lab, zorder=3)
    ax.set_title('Sample impurity vs exploration K  (lower = sharper)',
                 color=INK, fontsize=12, loc='left')
    ax.set_ylabel('distance to nearest mode (combined)', color=MUTED, fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([('XM-1\n(vanilla)' if k == 1 else f'XM-{k}') for k in ks],
                           color=INK, fontsize=9)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.legend(frameon=False, fontsize=10, loc='best')
    fig.suptitle('The paper reports these curves improving monotonically with K '
                 '(their Figs. 5–6). On our multimodal targets they are flat: '
                 'exploration buys nothing.', color=INK, fontsize=11.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f'{ASSET}/fig_xm_monotonic.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    print('wrote', out)


# --------------------------------------------------------------------------- #
# Figs 5-6 & 12 mirror: full arm study (designability vs K; and vs steps)
# --------------------------------------------------------------------------- #
def fig_arm_study():
    ks = [1, 2, 4, 8]
    x = np.arange(len(ks))
    # from docs/xm_explorative_modeling.md (XM-1 = vanilla)
    des_10k = [15.2, 10.4, 13.7, 3.5]        # 10k, <=128
    des_50k = [69.4, 30.1, 67.4, 66.2]       # 50k, <=128
    des_256 = {1: 29.9, 4: 22.9}             # 50k, <=256 (K1, K4 only)
    # designability vs inference steps (n=201; lengths 70/100/128)
    steps = [10, 50, 100, 500]
    van_steps = [8.0, 20.9, 27.9, 23.4]
    k4_steps = [8.0, 19.4, 24.4, 22.9]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    fig.patch.set_facecolor(SURFACE)

    # left: designability vs exploration K
    ax = axes[0]; ax.set_facecolor(SURFACE)
    ax.plot(x, des_50k, '-o', color=BLUE, lw=2, ms=8, label='50k steps, ≤128 res', zorder=3)
    ax.plot(x, des_10k, '--o', color=AQUA, lw=2, ms=7, label='10k steps, ≤128 res', zorder=3)
    ax.plot([0, 2], [des_256[1], des_256[4]], ':s', color=ORANGE, lw=2, ms=8,
            label='50k steps, ≤256 res', zorder=3)
    ax.set_title('Designability vs exploration K', color=INK, fontsize=12, loc='left')
    ax.set_ylabel('% designable (scRMSD < 2 Å)', color=MUTED, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([('XM-1\n(vanilla)' if k == 1 else f'XM-{k}') for k in ks],
                       color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc='best')
    ax.text(0.02, 0.03, 'paper: expected to rise with K → here flat/declining; XM-1 is best',
            transform=ax.transAxes, color=MUTED, fontsize=9.5, style='italic')

    # right: end-to-end stepping (their Fig. 12) — does exploration buy fewer steps?
    ax = axes[1]; ax.set_facecolor(SURFACE)
    xs = np.arange(len(steps))
    ax.plot(xs, van_steps, '-o', color=BLUE, lw=2, ms=8, label='XM-1 (vanilla)', zorder=3)
    ax.plot(xs, k4_steps, '-o', color=ORANGE, lw=2, ms=8, label='XM-4', zorder=3)
    ax.axvline(2, color=MUTED, lw=1, ls=':', zorder=1)
    ax.set_title('End-to-end stepping: designability vs inference steps',
                 color=INK, fontsize=12, loc='left')
    ax.set_ylabel('% designable', color=MUTED, fontsize=10)
    ax.set_xticks(xs); ax.set_xticklabels(steps, color=INK, fontsize=9)
    ax.set_xlabel('inference steps', color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc='best')
    ax.text(0.02, 0.03, 'paper: exploration should shift the peak left (fewer steps).\n'
            'Both peak at 100 — XM-4 does not buy fewer steps.',
            transform=ax.transAxes, color=MUTED, fontsize=9.5, style='italic')

    for ax in axes:
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED)
    fig.suptitle('Full arm study: XM never beats vanilla (XM-1), and exploration '
                 'does not reduce the optimal number of inference steps',
                 color=INK, fontsize=11.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = f'{ASSET}/fig_xm_arm_study.png'
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    fig_mode_capture()
    fig_monotonic()
    fig_arm_study()
