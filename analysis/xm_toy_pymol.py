"""Render PyMOL figures of the XM toy problem (run AFTER xm_toy_export.py).

  /usr/bin/python3.10 analysis/xm_toy_pymol.py

Produces in docs/assets/:
  toy_modes.png    the 4 ground-truth modes as SE(3) CA-traces + frame triads
  toy_vanilla.png  vanilla samples colored by nearest mode, on the modes + blur ghost
  toy_xm.png       XM K=4 samples, same view

DISPLAY NOTE. The 4 modes are centered (zero centre-of-mass), so in their true
coordinates they share a centroid and pile on top of each other. Translation is
arbitrary here (the whole problem is CoM-free), so for the figures ONLY we shift
each mode into one quadrant of a 2x2 grid. Internal shape and orientation are
untouched. This makes the story legible: each generated sample snaps onto one of
the 4 clusters, while the grey "blur" trace (the mode-average, i.e. the
conditional-mean trap) sits alone in the empty centre where NO sample lands --
the visual counterpart of blur_rate = 0.
"""
import os
import json
import pymol
pymol.finish_launching(['pymol', '-cq'])
from pymol import cmd
from pymol.cgo import CYLINDER

SRC = 'xm_eval_results/toy/pymol'
DST = 'docs/assets'
os.makedirs(DST, exist_ok=True)
meta = json.load(open(os.path.join(SRC, 'meta.json')))
frames = json.load(open(os.path.join(SRC, 'templates_frames.json')))
M, N = meta['M'], meta['N']
COLORS = meta['colors']
CHAINS = ['A', 'B', 'C', 'D'][:M]

SEP = 30.0   # quadrant half-spacing for the display grid (Angstrom)
GRID = [(-SEP, SEP), (SEP, SEP), (-SEP, -SEP), (SEP, -SEP)]   # NW, NE, SW, SE


def offset(m):
    x, y = GRID[m]
    return [x, y, 0.0]


def scene_setup():
    cmd.reinitialize()
    cmd.bg_color('white')
    cmd.set('ray_opaque_background', 0)
    cmd.set('orthoscopic', 1)
    cmd.set('ray_trace_mode', 1)          # black outlines
    cmd.set('ray_trace_gain', 0.15)
    cmd.set('ambient', 0.4)
    cmd.set('specular', 0.2)
    cmd.set('ribbon_trace_atoms', 1)      # trace CA-only "backbones"
    cmd.set('ribbon_sampling', 3)
    cmd.set('sphere_scale', 0.35)
    cmd.set('all_states', 0)


def name_color(i):
    n = f'modecol{i}'
    cmd.set_color(n, list(COLORS[i]))
    return n


def frame_triads(scale=2.2, radius=0.18):
    """CGO axis triads at each template residue -- shows these are SE(3) frames,
    not just points. Axis colors: x=red, y=green, z=blue (subdued). Each mode is
    shifted into its display quadrant so the triads track their trace."""
    axcol = [(0.85, 0.2, 0.2), (0.2, 0.7, 0.2), (0.2, 0.35, 0.85)]
    obj = []
    for m in range(M):
        T = frames['templates'][m]
        R = frames['template_R'][m]
        off = offset(m)
        for i in range(N):
            o = [T[i][k] + off[k] for k in range(3)]
            for a in range(3):
                d = [R[i][r][a] for r in range(3)]   # a-th column = local axis a
                tip = [o[k] + scale * d[k] for k in range(3)]
                c = axcol[a]
                obj += [CYLINDER, o[0], o[1], o[2], tip[0], tip[1], tip[2],
                        radius, c[0], c[1], c[2], c[0], c[1], c[2]]
    cmd.load_cgo(obj, 'triads')


def place_modes(name):
    """Show templates.pdb as one colored ribbon per chain, shifted per quadrant."""
    for m in range(M):
        sel = f'{name} and chain {CHAINS[m]}'
        cmd.show_as('ribbon', sel)
        cmd.color(name_color(m), sel)
        cmd.translate(offset(m), sel, camera=0)


def render(path):
    cmd.ray(1600, 1200)
    cmd.png(path, dpi=200)
    print('wrote', path)


# -------- Scene 1: the 4 modes -------------------------------------------- #
scene_setup()
cmd.load(os.path.join(SRC, 'templates.pdb'), 'modes')
place_modes('modes')
for m in range(M):
    cmd.show('spheres', f'modes and chain {CHAINS[m]}')
cmd.set('ribbon_width', 6)
frame_triads()
# CGO triads aren't selectable; the auto-fit view from load() already frames all
# four quadrants, so we only nudge the camera.
cmd.turn('y', 12)
view = cmd.get_view()                      # reuse this camera for the sample scenes
render(os.path.join(DST, 'toy_modes.png'))


# -------- Scenes 2 & 3: samples on the modes ------------------------------ #
def sample_scene(arm, out):
    scene_setup()
    # ground-truth modes as thick reference traces, one per quadrant
    cmd.load(os.path.join(SRC, 'templates.pdb'), 'modes')
    place_modes('modes')
    cmd.set('ribbon_width', 9, 'modes')
    cmd.set('ribbon_transparency', 0.35, 'modes')
    # blur point (mode-average) -- the conditional-mean trap; stays in the centre,
    # empty, because no sample lands there.
    cmd.load(os.path.join(SRC, 'blur.pdb'), 'blur')
    cmd.show_as('ribbon', 'blur')
    cmd.color('grey50', 'blur')
    cmd.show('spheres', 'blur')
    cmd.set('sphere_scale', 0.6, 'blur')
    cmd.set('ribbon_width', 9, 'blur')
    # generated samples, colored by assigned mode, shifted onto their quadrant
    for m in range(M):
        p = os.path.join(SRC, f'{arm}_mode{m}.pdb')
        if not os.path.exists(p):
            continue
        obj = f'{arm}_m{m}'
        cmd.load(p, obj)
        cmd.show_as('ribbon', obj)
        cmd.color(name_color(m), obj)
        cmd.set('ribbon_width', 2, obj)
        cmd.set('ribbon_transparency', 0.5, obj)
        cmd.set('all_states', 1, obj)
        cmd.translate(offset(m), obj, camera=0)
    cmd.set_view(view)
    render(out)


sample_scene('vanilla', os.path.join(DST, 'toy_vanilla.png'))
sample_scene('xm', os.path.join(DST, 'toy_xm.png'))
print('done')
