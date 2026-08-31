"""Export the 5 real PDB fold modes to CA-trace PDBs for PyMOL.

Reads xm_eval_results/real_folds/families_L129.pt (from real_folds_prep.py) and
writes CA traces of each mode's representative fold + the between-fold "blur"
average, into xm_eval_results/real_folds/pymol/. Then render with
analysis/real_folds_pymol.py.

Run: PYTHONPATH=. .venv/bin/python analysis/real_folds_export.py
"""
import json
import os
import torch

SRC = 'xm_eval_results/real_folds/families_L129.pt'
OUT = 'xm_eval_results/real_folds/pymol'
# distinct mode colors (RGB 0-1) matched in the PyMOL script
MODE_RGB = [(0.165, 0.471, 0.839), (0.922, 0.408, 0.204), (0.106, 0.686, 0.478),
            (0.541, 0.310, 0.741), (0.902, 0.671, 0.008)]


def write_ca_pdb_chains(path, coords_list, chains):
    """Single MODEL, one chain per CA trace."""
    lines, serial = [], 1
    for xyz, ch in zip(coords_list, chains):
        for i, (x, y, z) in enumerate(xyz):
            lines.append(
                f"ATOM  {serial:5d}  CA  GLY {ch}{i + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C")
            serial += 1
        lines.append("TER")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    data = torch.load(SRC)
    tmpl_T = data['tmpl_T']                         # [M,N,3] CA coords (centered)
    M, N = data['M'], data['N']
    meta = json.load(open(SRC.replace('.pt', '_meta.json')))
    reps = [f['rep'] for f in meta['families']]

    chains = ['A', 'B', 'C', 'D', 'E'][:M]
    write_ca_pdb_chains(os.path.join(OUT, 'modes.pdb'),
                        [tmpl_T[m].numpy() for m in range(M)], chains)
    blur_T = tmpl_T.mean(0).numpy()
    write_ca_pdb_chains(os.path.join(OUT, 'blur.pdb'), [blur_T], ['Z'])

    with open(os.path.join(OUT, 'meta.json'), 'w') as f:
        json.dump({'M': M, 'N': N, 'colors': MODE_RGB[:M], 'reps': reps}, f, indent=2)
    print(f'wrote {OUT}/modes.pdb ({M} real folds, N={N}), blur.pdb; reps={reps}')


if __name__ == '__main__':
    main()
