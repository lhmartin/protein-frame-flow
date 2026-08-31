"""Build a REAL multimodal SE(3)^N target from the PDB for the XM blur test.

The synthetic toy (analysis/xm_toy_mixture.py) showed vanilla flow matching never
blurs a multimodal target. The objection: its modes were random Gaussian
scaffolds, and a tiny MLP might separate them by some non-protein cue. This script
removes that objection WITHOUT fabricating any coordinates: the modes are real,
structurally-distinct PDB folds, and the within-mode variation is real deposited
structures too. The only "construction" is *which* real structures get grouped --
which we delegate to the repo's own foldseek TM clustering.

Pipeline (length-matched at L, so a between-family average is well defined):
  1. sample a pool of fully-modeled length-L backbones from the PDB metadata,
  2. write each as an atom37 PDB and run `foldseek easy-cluster` (TM-align),
  3. keep the M largest clusters as families,
  4. the foldseek representative of each cluster is the template T_m/R_m; every
     real member is Kabsch-superposed onto that template (rotating trans_1 AND
     rotmats_1) so the family shares one frame,
  5. save real templates [M,N,3]/[M,N,3,3] + real per-family member sets.

Run (CPU-only; needs the repo .venv which has openfold + tmtools):
  PYTHONPATH=. .venv/bin/python analysis/real_folds_prep.py \
      --length 129 --pool 400 --n_modes 5 --min_members 12

Output: xm_eval_results/real_folds/families_L{length}.pt (+ meta json).
Then run analysis/xm_real_folds.py.
"""
import argparse
import json
import os
import pathlib
import tempfile

import numpy as np
import pandas as pd
import torch
import tree
from openfold.data import data_transforms
from openfold.utils import rigid_utils

from data import utils as du
from analysis.utils import write_prot_to_pdb
from analysis.eval_designability import foldseek_cluster

OUT_DIR = 'xm_eval_results/real_folds'
FOLDSEEK = os.path.expanduser('~/bin/foldseek')


def load_frames(processed_path):
    """Replicate data/pdb_dataloader.py:_process_csv_row -> real SE(3) frames.

    Returns (trans_1 [N,3], rotmats_1 [N,3,3], atom37 [N,37,3], atom_mask [N,37],
    aatype [N], fully_modeled bool). All in Angstrom, CA-centered.
    """
    feats = du.parse_chain_feats(du.read_pkl(processed_path))
    modeled_idx = feats['modeled_idx']
    lo, hi = int(np.min(modeled_idx)), int(np.max(modeled_idx))
    del feats['modeled_idx']
    feats = tree.map_structure(lambda x: x[lo:hi + 1], feats)
    chain_feats = {
        'aatype': torch.tensor(feats['aatype']).long(),
        'all_atom_positions': torch.tensor(feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(feats['atom_mask']).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    rigids = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = rigids.get_rots().get_rot_mats().float()
    trans_1 = rigids.get_trans().float()
    bb_mask = torch.tensor(feats['bb_mask']).float()
    fully_modeled = bool((bb_mask > 0).all().item())
    return (trans_1, rotmats_1,
            feats['atom_positions'].astype(np.float32),
            feats['atom_mask'].astype(np.float32),
            feats['aatype'].astype(np.int64),
            fully_modeled)


def kabsch_R(P, Q):
    """Rotation R s.t. (P-P.mean) @ R.T best matches (Q-Q.mean); P,Q are [N,3].

    Same convention as analysis/self_consistency.kabsch_rmsd, but returns R.
    """
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def align_onto(trans, rotmats, R):
    """Apply global rotation R (rotating column vectors) to a structure's frames.

    trans:[N,3] rows -> rows @ R.T ; rotmats:[N,3,3] -> R @ rotmats (axes rotate).
    """
    R_t = torch.tensor(R, dtype=torch.float32)
    trans_c = trans - trans.mean(0, keepdim=True)
    trans_a = trans_c @ R_t.T
    rot_a = torch.einsum('ij,njk->nik', R_t, rotmats)
    return trans_a, rot_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--length', type=int, default=129)
    ap.add_argument('--pool', type=int, default=400,
                    help='how many length-L structures to cluster')
    ap.add_argument('--n_modes', type=int, default=5)
    ap.add_argument('--min_members', type=int, default=12)
    ap.add_argument('--tm_cutoff', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--metadata', default='metadata/pdb_metadata.csv')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    L = args.length

    df = pd.read_csv(args.metadata)
    df = df[df.modeled_seq_len == L]
    df = df.sample(min(args.pool, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f'length {L}: {len(df)} candidate structures (pool)')

    # Load frames + write atom37 PDBs for foldseek. Keep only fully-modeled, len==L.
    records = []
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='realfolds_'))
    pdb_dir = tmp / 'pdbs'
    pdb_dir.mkdir(parents=True, exist_ok=True)
    for i, row in df.iterrows():
        try:
            trans, rot, atom37, atom_mask, aatype, full = load_frames(row['processed_path'])
        except Exception as e:                       # skip unreadable/odd pkls
            continue
        if not full or trans.shape[0] != L:
            continue
        sid = f'{row["pdb_name"]}_{i}'
        write_prot_to_pdb(atom37, str(pdb_dir / f'{sid}.pdb'),
                          aatype=aatype, no_indexing=True, overwrite=True)
        records.append({'sid': sid, 'pdb_name': row['pdb_name'],
                        'trans': trans, 'rot': rot,
                        'ca': trans.numpy().astype(np.float64)})
    print(f'usable fully-modeled length-{L} structures: {len(records)}')
    by_sid = {r['sid']: r for r in records}

    # Cluster with the repo's foldseek helper (TM-align scoring).
    cluster_tsv = foldseek_cluster(FOLDSEEK, pdb_dir, args.tm_cutoff, tmp / 'fs')
    cl = pd.read_csv(cluster_tsv, sep='\t', header=None, names=['rep', 'member'])
    cl['rep'] = cl['rep'].str.replace('.pdb', '', regex=False)
    cl['member'] = cl['member'].str.replace('.pdb', '', regex=False)
    sizes = cl.groupby('rep').size().sort_values(ascending=False)
    print(f'{len(sizes)} clusters; top sizes: {sizes.head(10).tolist()}')

    chosen = [rep for rep, n in sizes.items() if n >= args.min_members][:args.n_modes]
    if len(chosen) < args.n_modes:
        print(f'WARNING: only {len(chosen)} clusters have >= {args.min_members} '
              f'members (wanted {args.n_modes}). Lower --min_members or raise --pool.')
    if not chosen:
        raise SystemExit('No clusters large enough; adjust --pool/--min_members/--tm_cutoff.')

    # Build families: rep = template; members Kabsch-aligned onto the rep.
    tmpl_T, tmpl_R, families, meta_fams = [], [], [], []
    for m, rep in enumerate(chosen):
        rep_rec = by_sid[rep]
        rep_trans = rep_rec['trans'] - rep_rec['trans'].mean(0, keepdim=True)
        tmpl_T.append(rep_trans)
        tmpl_R.append(rep_rec['rot'])
        members = cl[cl.rep == rep]['member'].tolist()
        fam_T, fam_R, kept = [], [], []
        for sid in members:
            rec = by_sid.get(sid)
            if rec is None:
                continue
            R = kabsch_R(rec['ca'], rep_rec['ca'])
            t_a, r_a = align_onto(rec['trans'], rec['rot'], R)
            fam_T.append(t_a)
            fam_R.append(r_a)
            kept.append(rec['pdb_name'])
        families.append({'trans': torch.stack(fam_T), 'rot': torch.stack(fam_R)})
        meta_fams.append({'rep': rep_rec['pdb_name'], 'n_members': len(kept),
                          'members': kept})
        print(f'  mode {m}: rep={rep_rec["pdb_name"]:12s} members={len(kept)}')

    tmpl_T = torch.stack(tmpl_T)                      # [M,N,3]
    tmpl_R = torch.stack(tmpl_R)                      # [M,N,3,3]

    out = os.path.join(OUT_DIR, f'families_L{L}.pt')
    torch.save({'tmpl_T': tmpl_T, 'tmpl_R': tmpl_R, 'families': families,
                'N': L, 'M': len(chosen)}, out)
    with open(os.path.join(OUT_DIR, f'families_L{L}_meta.json'), 'w') as f:
        json.dump({'length': L, 'n_modes': len(chosen), 'pool': len(records),
                   'tm_cutoff': args.tm_cutoff, 'families': meta_fams}, f, indent=2)
    print(f'\nwrote {out}  (M={len(chosen)}, N={L})')


if __name__ == '__main__':
    main()
