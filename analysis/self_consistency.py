"""Self-consistency (inverse-fold + fold) for FrameFlow backbones.

Ported from se3_diffusion_sparse's run_self_consistency / run_folding, made
standalone so it can run under a folding-capable environment (esm + ProteinMPNN)
WITHOUT importing FrameFlow's training modules — keeping the training venv clean.

For each generated backbone under <inference_dir>/length_*/sample_*/:
  1. ProteinMPNN designs N sequences for the backbone.
  2. ESMFold predicts a structure for each sequence.
  3. scRMSD / scTM are computed between each prediction and the backbone.
  4. Results are written to <sample_dir>/self_consistency/sc_results.csv
     (the layout analysis/eval_designability.py expects).

Run with the se3 venv (has esm + ProteinMPNN deps):
  /home/luke/code/personal/se3_diffusion_sparse/.venv/bin/python \
     analysis/self_consistency.py --inference_dir <dir> [--gpu 3]
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from biotite.sequence.io import fasta
import biotite.structure.io.pdb as pdb
import biotite.structure as struc
from tmtools import tm_align

# 3-letter -> 1-letter for reading sequences out of PDBs.
_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
    'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
    'TYR': 'Y', 'VAL': 'V',
}


def parse_ca(pdb_path):
    """Return (ca_positions [L,3] float, sequence str) for a single-chain PDB."""
    arr = pdb.PDBFile.read(pdb_path).get_structure(model=1)
    ca = arr[(arr.atom_name == 'CA') & struc.filter_amino_acids(arr)]
    seq = ''.join(_THREE_TO_ONE.get(r, 'G') for r in ca.res_name)
    return ca.coord.astype(np.float64), seq


def kabsch_rmsd(P, Q):
    """CA-RMSD after optimal rigid alignment of P onto Q (both [L,3])."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    P_aligned = Pc @ R.T
    return float(np.sqrt(((P_aligned - Qc) ** 2).sum(-1).mean()))


class ESMFolder:
    def __init__(self, device):
        import esm
        self.device = device
        self.model = esm.pretrained.esmfold_v1().eval().to(device)

    def infer(self, sequence, save_path):
        with torch.no_grad():
            pdb_str = self.model.infer_pdb(sequence)
        with open(save_path, 'w') as f:
            f.write(pdb_str)


def find_backbone(sample_dir):
    """Final backbone PDB in a sample dir (save_traj writes sample_1.pdb)."""
    for name in ('sample_1.pdb', 'sample.pdb'):
        p = os.path.join(sample_dir, name)
        if os.path.exists(p):
            return p
    cands = sorted(glob.glob(os.path.join(sample_dir, 'sample_*.pdb')))
    return cands[0] if cands else None


def run_proteinmpnn(pmpnn_dir, decoy_dir, ref_pdb, n_seq):
    """Design n_seq sequences for the backbone(s) in decoy_dir. Runs on CPU
    (sub-second even for hundreds of residues) which keeps the GPU free for
    ESMFold and avoids GPU contention. Returns the fasta path."""
    parsed = os.path.join(decoy_dir, 'parsed_pdbs.jsonl')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')  # force ProteinMPNN onto CPU
    subprocess.run(
        [sys.executable, f'{pmpnn_dir}/helper_scripts/parse_multiple_chains.py',
         f'--input_path={decoy_dir}', f'--output_path={parsed}'],
        check=True, env=env)
    r = subprocess.run(
        [sys.executable, f'{pmpnn_dir}/protein_mpnn_run.py',
         '--out_folder', decoy_dir, '--jsonl_path', parsed,
         '--num_seq_per_target', str(n_seq),
         '--sampling_temp', '0.1', '--seed', '38', '--batch_size', '1'],
        env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f'ProteinMPNN failed (rc={r.returncode}): {r.stderr[-800:]}')
    return os.path.join(decoy_dir, 'seqs',
                        os.path.basename(ref_pdb).replace('.pdb', '.fa'))


def self_consistency_for_sample(sample_dir, folder, pmpnn_dir, n_seq, gpu):
    backbone = find_backbone(sample_dir)
    if backbone is None:
        return None
    sc_dir = os.path.join(sample_dir, 'self_consistency')
    existing_csv = os.path.join(sc_dir, 'sc_results.csv')
    if os.path.exists(existing_csv) and os.path.getsize(existing_csv) > 0:
        return pd.read_csv(existing_csv)  # reuse prior fold (skip = resumable)
    os.makedirs(sc_dir, exist_ok=True)
    # ProteinMPNN reads every .pdb in its input dir; give it a clean dir with the
    # single reference backbone named after the sample.
    ref_pdb = os.path.join(sc_dir, 'reference.pdb')
    if not os.path.exists(ref_pdb):
        import shutil
        shutil.copy(backbone, ref_pdb)
    fasta_path = run_proteinmpnn(pmpnn_dir, sc_dir, ref_pdb, n_seq)

    ref_ca, _ = parse_ca(ref_pdb)
    esmf_dir = os.path.join(sc_dir, 'esmf')
    os.makedirs(esmf_dir, exist_ok=True)
    seqs = fasta.FastaFile.read(fasta_path)
    rows = []
    for i, (header, seq) in enumerate(seqs.items()):
        if i == 0:
            continue  # ProteinMPNN's first entry is the input sequence itself.
        esmf_path = os.path.join(esmf_dir, f'sample_{i}.pdb')
        folder.infer(seq, esmf_path)
        pred_ca, _ = parse_ca(esmf_path)
        L = min(len(ref_ca), len(pred_ca))
        tm = tm_align(ref_ca[:L], pred_ca[:L], seq[:L], seq[:L])
        rows.append({
            'header': header, 'sequence': seq, 'sample_path': esmf_path,
            'rmsd': kabsch_rmsd(pred_ca[:L], ref_ca[:L]),
            'tm_score': float(tm.tm_norm_chain1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sc_dir, 'sc_results.csv'), index=False)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inference_dir', required=True)
    p.add_argument('--pmpnn_dir',
                   default='/home/luke/code/personal/se3_diffusion_sparse/ProteinMPNN')
    p.add_argument('--seq_per_sample', type=int, default=8)
    p.add_argument('--gpu', type=int, default=None,
                   help='GPU id for ProteinMPNN + ESMFold (e.g. 3). None=CPU.')
    args = p.parse_args()

    if args.gpu is None:
        device = 'cpu'
    elif os.environ.get('CUDA_VISIBLE_DEVICES'):
        # Pinned via CUDA_VISIBLE_DEVICES (e.g. by thermal_guard): the target GPU
        # is re-indexed to cuda:0, so --gpu is relative to the visible set.
        device = 'cuda:0'
    else:
        device = f'cuda:{args.gpu}'
    print(f'[sc] loading ESMFold on {device} '
          f'(CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES","")}) ...', flush=True)
    folder = ESMFolder(device)

    sample_dirs = sorted(glob.glob(
        os.path.join(args.inference_dir, 'length_*', 'sample_*')))
    print(f'[sc] {len(sample_dirs)} samples under {args.inference_dir}', flush=True)
    for sd in sample_dirs:
        try:
            df = self_consistency_for_sample(
                sd, folder, args.pmpnn_dir, args.seq_per_sample, args.gpu)
            if df is not None and not df.empty:
                print(f'[sc] {sd}: min scRMSD={df.rmsd.min():.2f} '
                      f'max scTM={df.tm_score.max():.3f} n={len(df)}', flush=True)
        except Exception as e:
            print(f'[sc] FAILED {sd}: {e}', flush=True)


if __name__ == '__main__':
    main()
