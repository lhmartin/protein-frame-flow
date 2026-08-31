"""Export the XM toy problem to PDB traces for PyMOL visualization.

Trains vanilla (K=1) and XM K=4 on the same M-mode SE(3)^N mixture, samples from
each, assigns every sample to its nearest ground-truth template, and writes:

  xm_eval_results/toy/pymol/
    templates.pdb          4 modes as CA traces (chains A-D)
    templates_frames.json  per-residue rotation frames (for SE(3) triads) + blur trace
    vanilla_mode{k}.pdb    vanilla samples assigned to mode k (multi-MODEL, one per sample)
    xm_mode{k}.pdb         XM K=4 samples assigned to mode k
    meta.json              N, M, colors, per-arm mode counts

Run: PYTHONPATH=. .venv/bin/python analysis/xm_toy_export.py   (CPU-only)
Then render with analysis/xm_toy_pymol.py (uses /usr/bin/python3.10 + PyMOL).
"""
import os
import json
import numpy as np
import torch

# xm_toy_mixture hides the GPU on import (CUDA_VISIBLE_DEVICES='') — stays CPU-only.
from analysis.xm_toy_mixture import (
    make_templates, make_interpolant, ToyVF, train, sample,
    _d_trans, _d_rot, _project_so3,
)

OUT = 'xm_eval_results/toy/pymol'
DELTA, M, N, SEED = 6.0, 4, 8, 0
STEPS, BATCH, N_SAMPLES, TSTEPS = 2500, 256, 60, 100
# 4 distinct mode colors (RGB 0-1) — matched in the PyMOL script.
MODE_RGB = [(0.165, 0.471, 0.839), (0.922, 0.408, 0.204),
            (0.106, 0.686, 0.478), (0.541, 0.310, 0.741)]


def write_ca_pdb(path, coords_list, chains=None):
    """coords_list: list of [N,3] arrays -> multi-MODEL PDB of CA traces.
    If `chains` given (list of chain letters, one per model), use them; else 'A'."""
    lines = []
    for mi, xyz in enumerate(coords_list):
        ch = chains[mi] if chains else 'A'
        lines.append(f"MODEL     {mi + 1:4d}")
        for i, (x, y, z) in enumerate(xyz):
            lines.append(
                f"ATOM  {i + 1:5d}  CA  GLY {ch}{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
        lines.append("ENDMDL")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def write_ca_pdb_chains(path, coords_list, chains):
    """Single MODEL, one chain per trace (for the templates figure)."""
    lines = []
    serial = 1
    for xyz, ch in zip(coords_list, chains):
        for i, (x, y, z) in enumerate(xyz):
            lines.append(
                f"ATOM  {serial:5d}  CA  GLY {ch}{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
            serial += 1
        lines.append("TER")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def assign_modes(trans, rots, tmpl_T, tmpl_R):
    dt = _d_trans(trans, tmpl_T)
    dr = _d_rot(rots, tmpl_R)
    off = ~torch.eye(M, dtype=torch.bool)
    ref_t = _d_trans(tmpl_T, tmpl_T)[off].mean().clamp(min=1e-6)
    ref_r = _d_rot(tmpl_R, tmpl_R)[off].mean().clamp(min=1e-6)
    comb = dt / ref_t + dr / ref_r
    return comb.argmin(1).numpy()


def run_arm(tmpl_T, tmpl_R, K):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    interp = make_interpolant(xm_enabled=(K > 1), K=K)
    model = ToyVF(N)
    train(model, interp, tmpl_T, tmpl_R, STEPS, BATCH, K)
    trans, rots = sample(model, interp, N, N_SAMPLES, TSTEPS)
    assign = assign_modes(trans, rots, tmpl_T, tmpl_R)
    return trans.numpy(), assign


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    tmpl_T, tmpl_R = make_templates(N, M, DELTA, seed=1000 + SEED)

    # templates + blur (mode-average) trace
    chains = ['A', 'B', 'C', 'D'][:M]
    write_ca_pdb_chains(
        os.path.join(OUT, 'templates.pdb'),
        [tmpl_T[m].numpy() for m in range(M)], chains)
    blur_T = tmpl_T.mean(0).numpy()
    write_ca_pdb_chains(os.path.join(OUT, 'blur.pdb'), [blur_T], ['Z'])

    frames = {
        'templates': [tmpl_T[m].numpy().tolist() for m in range(M)],
        'template_R': [tmpl_R[m].numpy().tolist() for m in range(M)],
        'blur': blur_T.tolist(),
    }
    with open(os.path.join(OUT, 'templates_frames.json'), 'w') as f:
        json.dump(frames, f)

    meta = {'N': N, 'M': M, 'delta': DELTA, 'colors': MODE_RGB, 'arms': {}}
    for arm, K in [('vanilla', 1), ('xm', 4)]:
        trans, assign = run_arm(tmpl_T, tmpl_R, K)
        counts = [int((assign == m).sum()) for m in range(M)]
        meta['arms'][arm] = {'K': K, 'counts': counts, 'n': int(len(assign))}
        for m in range(M):
            samples_m = [trans[i] for i in range(len(assign)) if assign[i] == m]
            path = os.path.join(OUT, f'{arm}_mode{m}.pdb')
            if samples_m:
                write_ca_pdb(path, samples_m)
            elif os.path.exists(path):
                os.remove(path)
        print(f"{arm:8s} (K={K}) mode counts = {counts}  n={len(assign)}")

    with open(os.path.join(OUT, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"exported to {OUT}/")


if __name__ == '__main__':
    main()
