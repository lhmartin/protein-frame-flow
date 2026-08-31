"""Generate paper-style point-cloud data (mirrors Fig. 1 of Gladstone, Ji & Du,
"Explorative Modeling", https://explorative-modeling.github.io/).

Their Fig. 1 shows, per domain, ground truth + progressive XM-K generations, and
their toy claim is that XM-1 "collapses to the mean" while larger K "captures more
modes instead of averaging them". We reproduce that exact panel for our two
targets -- the synthetic SE(3)^N mixture toy and the REAL foldseek-clustered PDB
folds -- to test the claim directly. (Spoiler carried in the metrics: XM-1 already
captures every mode here, so the panels look flat across K.)

We explore over the *noise coupling* (the flow-matching-native XM variant: shared
t and x_1, best-of-K over the noise->data pairing), which is what this repo
implements; the paper's default "Forward XM" explores over generations.

Projection: each structure's CA translations are flattened and projected to 2D by
a PCA fit on the M mode templates (their natural 2D analog of the paper's 2D
Gaussian-mixture toy). Rotations are folded into the nearest-mode assignment via
the same combined metric the metrics use.

Output: xm_eval_results/paper_figs/clouds.npz  (CPU-only; ~15 min).
Run: PYTHONPATH=. .venv/bin/python analysis/xm_paper_figs.py
"""
import os
import numpy as np
import torch

from analysis.xm_toy_mixture import (
    make_interpolant, ToyVF, sample, make_templates, sample_data,
    _d_trans, _d_rot, train as toy_train,
)
from analysis import xm_real_folds as rf

KS = [1, 2, 4, 8]
OUT = 'xm_eval_results/paper_figs/clouds.npz'


def pca_2d(templates_flat):
    """Fit a 2-component PCA on the M mode templates [M, D]; return (mu, W[2,D])."""
    mu = templates_flat.mean(0)
    X = templates_flat - mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return mu, Vt[:2]                              # [D], [2, D]


def project(trans, mu, W):
    """trans [S,N,3] -> [S,2] via the template PCA basis."""
    flat = trans.reshape(trans.shape[0], -1).numpy()
    return (flat - mu) @ W.T


def assign_modes(trans, rots, tmpl_T, tmpl_R):
    """Nearest mode per sample via the metrics' combined (trans+rot) distance."""
    dt = _d_trans(trans, tmpl_T)
    dr = _d_rot(rots, tmpl_R)
    off = ~torch.eye(tmpl_T.shape[0], dtype=torch.bool)
    ref_t = _d_trans(tmpl_T, tmpl_T)[off].mean().clamp(min=1e-6)
    ref_r = _d_rot(tmpl_R, tmpl_R)[off].mean().clamp(min=1e-6)
    comb = dt / ref_t + dr / ref_r
    return comb.min(dim=1).indices.numpy()


def gen_target(tmpl_T, tmpl_R, mu, W, draw_fn, n):
    """Ground-truth panel: samples straight from the target distribution."""
    data = draw_fn(n)
    trans, rots = data['trans_1'], data['rotmats_1']
    return {'xy': project(trans, mu, W),
            'assign': assign_modes(trans, rots, tmpl_T, tmpl_R)}


def build_toy(n_eval=1024, steps=3000, tsteps=100, seed=0):
    N, M, delta = 8, 4, 6.0
    torch.manual_seed(seed); np.random.seed(seed)
    tmpl_T, tmpl_R = make_templates(N, M, delta, seed=1000 + seed)
    mu, W = pca_2d(tmpl_T.reshape(M, -1).numpy())
    tmpl_xy = project(tmpl_T, mu, W)
    blur_xy = project(tmpl_T.mean(0, keepdim=True), mu, W)
    panels = {'GT': gen_target(tmpl_T, tmpl_R, mu, W,
                               lambda n: sample_data(tmpl_T, tmpl_R, n), n_eval)}
    for K in KS:
        torch.manual_seed(seed); np.random.seed(seed)
        interp = make_interpolant(xm_enabled=(K > 1), K=K)
        model = ToyVF(N)
        toy_train(model, interp, tmpl_T, tmpl_R, steps, 256, K)
        trans, rots = sample(model, interp, N, n_eval, tsteps)
        panels[f'XM-{K}'] = {'xy': project(trans, mu, W),
                             'assign': assign_modes(trans, rots, tmpl_T, tmpl_R)}
        print(f'  toy XM-{K} done', flush=True)
    return tmpl_xy, blur_xy, panels, M


def build_real(fam_path='xm_eval_results/real_folds/families_L129.pt',
               n_eval=1024, steps=1500, tsteps=100, seed=0):
    fam = torch.load(fam_path)
    families, M, N = fam['families'], fam['M'], fam['N']
    tmpl_T, tmpl_R = fam['tmpl_T'], fam['tmpl_R']
    mu, W = pca_2d(tmpl_T.reshape(M, -1).numpy())
    tmpl_xy = project(tmpl_T, mu, W)
    blur_xy = project(tmpl_T.mean(0, keepdim=True), mu, W)

    def draw(n):
        g = torch.Generator().manual_seed(999)
        return rf.real_sample_data(families, n, M, 0.5, 0.02, g)

    panels = {'GT': gen_target(tmpl_T, tmpl_R, mu, W, draw, n_eval)}
    for K in KS:
        torch.manual_seed(seed); np.random.seed(seed)
        interp = make_interpolant(xm_enabled=(K > 1), K=K)
        model = ToyVF(N, hidden=512)
        rf.train(model, interp, families, M, steps, 128, K, 0.5, 0.02, seed,
                 lr=1e-4, optimizer='adamw')
        trans, rots = sample(model, interp, N, n_eval, tsteps)
        panels[f'XM-{K}'] = {'xy': project(trans, mu, W),
                             'assign': assign_modes(trans, rots, tmpl_T, tmpl_R)}
        print(f'  real XM-{K} done', flush=True)
    return tmpl_xy, blur_xy, panels, M


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print('building toy panels...', flush=True)
    toy_tmpl, toy_blur, toy_panels, toy_M = build_toy()
    print('building real-fold panels...', flush=True)
    real_tmpl, real_blur, real_panels, real_M = build_real()

    save = {'toy_tmpl_xy': toy_tmpl, 'toy_blur_xy': toy_blur, 'toy_M': toy_M,
            'real_tmpl_xy': real_tmpl, 'real_blur_xy': real_blur, 'real_M': real_M,
            'cols': np.array(['GT'] + [f'XM-{k}' for k in KS])}
    for tag, panels in [('toy', toy_panels), ('real', real_panels)]:
        for name, d in panels.items():
            save[f'{tag}_{name}_xy'] = d['xy']
            save[f'{tag}_{name}_assign'] = d['assign']
    np.savez(OUT, **save)
    print(f'wrote {OUT}', flush=True)


if __name__ == '__main__':
    main()
