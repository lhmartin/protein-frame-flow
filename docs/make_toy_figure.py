"""Render the XM toy-mixture figure (purity + mode balance vs K, per delta).

Two series = the two separation regimes (delta=2 overlapping, delta=6 separated).
K=1 IS the vanilla arm, so the x-axis (K=1,2,4,8) is the full XM sweep. Uses the
validated light-mode categorical palette. Divergent vanilla seeds (purity>1 A) are
excluded from the plotted points and annotated.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# validated light palette
BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'

df = pd.read_csv('xm_eval_results/toy/toy_results.csv')
df = df.sort_values('K')
Ks = [1, 2, 4, 8]
REG = [(2.0, BLUE, 'Δ=2 (overlapping)'), (6.0, ORANGE, 'Δ=6 (separated)')]


def series(delta, col, clip=None):
    med, lo, hi = [], [], []
    for K in Ks:
        v = df[(df.delta == delta) & (df.K == K)][col].values.astype(float)
        if clip is not None:
            v = v[v < clip]
        med.append(np.median(v)); lo.append(v.min()); hi.append(v.max())
    return np.array(med), np.array(lo), np.array(hi)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
fig.patch.set_facecolor(SURFACE)
x = np.arange(len(Ks))

# Panel A: purity (nearest-template combined distance; lower = purer)
ax = axes[0]
ax.set_facecolor(SURFACE)
for delta, c, lab in REG:
    m, lo, hi = series(delta, 'purity_comb', clip=1.0)
    ax.plot(x, m, '-o', color=c, lw=2, ms=7, label=lab, zorder=3)
    ax.fill_between(x, lo, hi, color=c, alpha=0.15, zorder=1)
ax.set_title('Purity: distance to nearest template', color=INK, fontsize=12, loc='left')
ax.set_ylabel('combined dist (lower = purer)', color=MUTED, fontsize=10)
ax.set_ylim(0, 0.06)
ax.annotate('vanilla Δ=6 seed 2 diverged\n(10,174 Å) — excluded',
            xy=(0, 0.021), xytext=(0.35, 0.048), color=MUTED, fontsize=8,
            arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.8))

# Panel B: mode balance (1.0 = perfectly even coverage of the 4 modes)
ax = axes[1]
ax.set_facecolor(SURFACE)
for delta, c, lab in REG:
    m, lo, hi = series(delta, 'balance', clip=None)
    # drop the diverged seed (balance 0.0) from the delta=6 vanilla point range
    m2, lo2, hi2 = [], [], []
    for K in Ks:
        v = df[(df.delta == delta) & (df.K == K)]
        v = v[v.purity_trans_A < 1.0]['balance'].values.astype(float)
        m2.append(np.median(v)); lo2.append(v.min()); hi2.append(v.max())
    ax.plot(x, m2, '-o', color=c, lw=2, ms=7, label=lab, zorder=3)
    ax.fill_between(x, lo2, hi2, color=c, alpha=0.15, zorder=1)
ax.set_title('Mode balance (1.0 = perfectly even)', color=INK, fontsize=12, loc='left')
ax.set_ylabel('min mode share × M', color=MUTED, fontsize=10)
ax.set_ylim(0.6, 1.0)

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(['K=1\n(vanilla)', 'K=2', 'K=4', 'K=8'], color=INK, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc='lower right')

fig.suptitle('XM on a controlled multimodal SE(3) mixture — blur rate = 0.000 for all 24 runs; '
             'all arms cover 4/4 modes',
             color=INK, fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs('docs/assets', exist_ok=True)
out = 'docs/assets/xm_toy_mixture.png'
fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
print('wrote', out)
