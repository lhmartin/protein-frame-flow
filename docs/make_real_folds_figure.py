"""Render the real-fold XM figure (purity + mode balance vs K).

Reads xm_eval_results/real_folds/real_results.csv (main sweep, hidden=512) and,
if present, real_results_cap.csv (capacity check, hidden=1024). K=1 IS the vanilla
arm. blur_rate is annotated (0.000 for all arms, as in the synthetic toy). Uses
the validated light-mode categorical palette.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#898781', '#e1e0d9', '#fcfcfb'

paths = ['xm_eval_results/real_folds/real_results.csv',
         'xm_eval_results/real_folds/real_results_cap.csv']
frames = [pd.read_csv(p) for p in paths if os.path.exists(p)]
df = pd.concat(frames, ignore_index=True)
Ks = sorted(df.K.unique())
x = np.arange(len(Ks))

# one series per hidden size (capacity)
hiddens = sorted(df.hidden.unique())
colors = {h: c for h, c in zip(hiddens, [BLUE, ORANGE, AQUA])}


def series(hidden, col):
    med, lo, hi = [], [], []
    for K in Ks:
        v = df[(df.hidden == hidden) & (df.K == K)][col].values.astype(float)
        v = v[np.isfinite(v)]                       # drop any diverged/NaN runs
        if len(v) == 0:
            med.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        med.append(np.median(v)); lo.append(v.min()); hi.append(v.max())
    return np.array(med), np.array(lo), np.array(hi)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
fig.patch.set_facecolor(SURFACE)

# Panel A: purity (nearest-fold combined distance; lower = purer)
ax = axes[0]
ax.set_facecolor(SURFACE)
for h in hiddens:
    m, lo, hi = series(h, 'purity_comb')
    ax.plot(x, m, '-o', color=colors[h], lw=2, ms=7, label=f'MLP hidden={h}', zorder=3)
    ax.fill_between(x, lo, hi, color=colors[h], alpha=0.15, zorder=1)
ax.set_title('Purity: distance to nearest real fold', color=INK, fontsize=12, loc='left')
ax.set_ylabel('combined dist (lower = purer)', color=MUTED, fontsize=10)

# Panel B: mode balance (1.0 = perfectly even coverage of the M real folds)
ax = axes[1]
ax.set_facecolor(SURFACE)
for h in hiddens:
    m, lo, hi = series(h, 'balance')
    ax.plot(x, m, '-o', color=colors[h], lw=2, ms=7, label=f'MLP hidden={h}', zorder=3)
    ax.fill_between(x, lo, hi, color=colors[h], alpha=0.15, zorder=1)
ax.set_title('Mode balance (1.0 = perfectly even)', color=INK, fontsize=12, loc='left')
ax.set_ylabel('min mode share x M', color=MUTED, fontsize=10)
ax.set_ylim(0, 1.05)

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([('K=1\n(vanilla)' if k == 1 else f'K={k}') for k in Ks],
                       color=INK, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=9, loc='best')

blur_max = float(df.blur_rate.max())
cov = df.modes_covered.min(), df.modes_covered.max()
fig.suptitle(f'XM on 5 REAL PDB folds (length 129) — blur rate <= {blur_max:.3f} for all arms; '
             f'coverage {cov[0]}-{cov[1]}/5 modes',
             color=INK, fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs('docs/assets', exist_ok=True)
out = 'docs/assets/xm_real_folds.png'
fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches='tight')
print('wrote', out)
