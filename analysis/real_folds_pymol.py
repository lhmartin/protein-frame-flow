"""Render the 5 real PDB fold modes (run AFTER real_folds_export.py).

  /usr/bin/python3.10 analysis/real_folds_pymol.py

Produces docs/assets/real_folds_modes.png: the 5 real, structurally-distinct
length-129 folds that make up the multimodal target, each a CA ribbon in its own
cell of a grid, with the grey between-fold "blur" average alone in the centre.
As in the toy figures the modes are laid out on a grid for display only
(translation is arbitrary in this CoM-free problem).
"""
import json
import os
import pymol
pymol.finish_launching(['pymol', '-cq'])
from pymol import cmd

SRC = 'xm_eval_results/real_folds/pymol'
DST = 'docs/assets'
os.makedirs(DST, exist_ok=True)
meta = json.load(open(os.path.join(SRC, 'meta.json')))
M = meta['M']
COLORS = meta['colors']
CHAINS = ['A', 'B', 'C', 'D', 'E'][:M]

SEP = 55.0   # grid half-spacing (Angstrom); real folds span ~30-40 A
GRID = [(-SEP, SEP), (SEP, SEP), (-SEP, -SEP), (SEP, -SEP), (0.0, 0.0)]  # 5th = centre-top


def offset(m):
    # place the 5th mode above centre so the grey blur can own the middle
    if m == 4:
        return [0.0, SEP * 1.9, 0.0]
    x, y = GRID[m]
    return [x, y, 0.0]


def name_color(i):
    n = f'modecol{i}'
    cmd.set_color(n, list(COLORS[i]))
    return n


cmd.reinitialize()
cmd.bg_color('white')
cmd.set('ray_opaque_background', 0)
cmd.set('orthoscopic', 1)
cmd.set('ray_trace_mode', 1)
cmd.set('ray_trace_gain', 0.15)
cmd.set('ambient', 0.4)
cmd.set('specular', 0.2)
cmd.set('ribbon_trace_atoms', 1)
cmd.set('ribbon_sampling', 3)
cmd.set('ribbon_width', 6)

cmd.load(os.path.join(SRC, 'modes.pdb'), 'modes')
for m in range(M):
    sel = f'modes and chain {CHAINS[m]}'
    cmd.show_as('ribbon', sel)
    cmd.color(name_color(m), sel)
    cmd.translate(offset(m), sel, camera=0)

cmd.load(os.path.join(SRC, 'blur.pdb'), 'blur')
cmd.show_as('ribbon', 'blur')
cmd.color('grey50', 'blur')
cmd.set('ribbon_width', 6, 'blur')

cmd.zoom('all', 5)
cmd.turn('y', 10)
cmd.ray(1600, 1250)
out = os.path.join(DST, 'real_folds_modes.png')
cmd.png(out, dpi=200)
print('wrote', out)
