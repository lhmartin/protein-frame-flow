"""Aggregate self-consistency, structural diversity, and PDB novelty for one
inference checkpoint into a single summary row.

Run after analysis/self_consistency.py has populated an inference output dir with
length_*/sample_*/self_consistency/sc_results.csv. CPU-only (pandas + foldseek).

Example:
    python analysis/eval_designability.py \\
        --inference_dir <inference_dir> \\
        --pdb_db ~/data/foldseek_targets/pdb \\
        --out_csv designability_summary.csv
"""
import argparse
import csv
import glob
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import pandas as pd


def parse_step(inference_dir: str):
    m = re.search(r'step_(\d+)', os.path.basename(os.path.normpath(inference_dir)))
    return int(m.group(1)) if m else None


def collect_samples(inference_dir: str, scrmsd_cutoff: float) -> pd.DataFrame:
    rows = []
    pattern = os.path.join(inference_dir, 'length_*', 'sample_*', 'self_consistency', 'sc_results.csv')
    for sc_path in glob.glob(pattern):
        sample_dir = pathlib.Path(sc_path).parent.parent
        length_dir = sample_dir.parent
        length = int(re.search(r'length_(\d+)', length_dir.name).group(1))
        sample_idx = int(re.search(r'sample_(\d+)', sample_dir.name).group(1))
        df = pd.read_csv(sc_path)
        if df.empty:
            continue
        # FrameFlow inference writes the final backbone as sample.pdb (se3 used
        # sample_1.pdb); avoid the *_traj.pdb trajectory files.
        backbone = None
        for name in ('sample.pdb', 'sample_1.pdb'):
            if (sample_dir / name).exists():
                backbone = sample_dir / name
                break
        if backbone is None:
            cands = [p for p in sorted(sample_dir.glob('sample*.pdb'))
                     if 'traj' not in p.name]
            if not cands:
                continue
            backbone = cands[0]
        rows.append({
            'length': length,
            'sample_idx': sample_idx,
            'backbone': str(backbone),
            'min_rmsd': float(df['rmsd'].min()),
            'max_tm': float(df['tm_score'].max()),
            'mean_rmsd': float(df['rmsd'].mean()),
            'mean_tm': float(df['tm_score'].mean()),
            'n_seq': len(df),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(['length', 'sample_idx']).reset_index(drop=True)
    out['designable'] = out['min_rmsd'] < scrmsd_cutoff
    return out


def stage_backbones(samples: pd.DataFrame, dst: pathlib.Path) -> dict:
    """Copy backbones into `dst` with stable IDs (`len{L}_s{i}.pdb`). Returns id -> source path."""
    dst.mkdir(parents=True, exist_ok=True)
    id_to_src = {}
    for _, r in samples.iterrows():
        sid = f"len{int(r['length'])}_s{int(r['sample_idx'])}"
        shutil.copy(r['backbone'], dst / f"{sid}.pdb")
        id_to_src[sid] = r['backbone']
    return id_to_src


def foldseek_cluster(foldseek_bin: str, pdb_dir: pathlib.Path, tm_cutoff: float, tmp: pathlib.Path):
    """Cluster the PDB files in `pdb_dir` by structural similarity. Returns the cluster TSV path."""
    tmp.mkdir(parents=True, exist_ok=True)
    out_prefix = tmp / 'cluster'
    fs_tmp = tmp / 'fs_tmp'
    fs_tmp.mkdir(parents=True, exist_ok=True)
    cmd = [
        foldseek_bin, 'easy-cluster',
        str(pdb_dir), str(out_prefix), str(fs_tmp),
        '--alignment-type', '1',          # 1 = TM-align scoring
        '--tmscore-threshold', str(tm_cutoff),
        '-c', '0.0',                      # coverage; do not restrict by length overlap
        '--cov-mode', '0',
        '--min-seq-id', '0',
        '-v', '1',
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return pathlib.Path(f"{out_prefix}_cluster.tsv")


def count_clusters(cluster_tsv: pathlib.Path) -> int:
    df = pd.read_csv(cluster_tsv, sep='\t', header=None, names=['rep', 'member'])
    return df['rep'].nunique()


def foldseek_search(foldseek_bin: str, query_dir: pathlib.Path, target_db: str, tmp: pathlib.Path):
    """Search query PDBs against target DB. Returns hits dataframe (one row per hit)."""
    tmp.mkdir(parents=True, exist_ok=True)
    hits_tsv = tmp / 'hits.tsv'
    fs_tmp = tmp / 'fs_tmp_search'
    fs_tmp.mkdir(parents=True, exist_ok=True)
    cmd = [
        foldseek_bin, 'easy-search',
        str(query_dir), target_db, str(hits_tsv), str(fs_tmp),
        '--alignment-type', '1',
        '--format-output', 'query,target,alntmscore',
        '--tmscore-threshold', '0.0',
        '-e', '10',
        '--max-seqs', '1000',
        '-v', '1',
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if not hits_tsv.exists() or hits_tsv.stat().st_size == 0:
        return pd.DataFrame(columns=['query', 'target', 'alntmscore'])
    return pd.read_csv(hits_tsv, sep='\t', header=None,
                       names=['query', 'target', 'alntmscore'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inference_dir', required=True,
                   help='Path to one ckpt inference output dir (contains length_*/sample_*/...)')
    p.add_argument('--pdb_db', default=os.path.expanduser('~/data/foldseek_targets/pdb'),
                   help='Foldseek target DB prefix for novelty search')
    p.add_argument('--scrmsd_cutoff', type=float, default=2.0)
    p.add_argument('--tm_cluster_cutoff', type=float, default=0.5)
    p.add_argument('--out_csv', required=True)
    p.add_argument('--foldseek_bin', default=os.path.expanduser('~/bin/foldseek'))
    p.add_argument('--skip_novelty', action='store_true',
                   help='Skip the PDB-novelty Foldseek search (useful before DB is ready)')
    p.add_argument('--keep_tmp', action='store_true',
                   help='Keep the working dir for debugging')
    args = p.parse_args()

    inference_dir = os.path.abspath(args.inference_dir)
    ckpt_name = os.path.basename(os.path.normpath(inference_dir))
    step = parse_step(inference_dir)
    print(f"[eval] inference_dir={inference_dir}")
    print(f"[eval] ckpt={ckpt_name}  step={step}")

    samples = collect_samples(inference_dir, args.scrmsd_cutoff)
    if samples.empty:
        sys.exit(f"No sc_results.csv files found under {inference_dir}")
    n_total = len(samples)
    n_designable = int(samples['designable'].sum())
    designability = n_designable / n_total
    print(f"[eval] n_total={n_total}  n_designable={n_designable}  designability={designability:.3f}")

    work = pathlib.Path(tempfile.mkdtemp(prefix='eval_designability_'))
    print(f"[eval] tmp dir: {work}")

    summary = {
        'ckpt': ckpt_name,
        'step': step,
        'n_total': n_total,
        'n_designable': n_designable,
        'designability': round(designability, 4),
        'median_best_scRMSD': round(float(samples['min_rmsd'].median()), 3),
        'median_best_scTM':   round(float(samples['max_tm'].median()),   3),
        'mean_best_scRMSD':   round(float(samples['min_rmsd'].mean()),   3),
        'mean_best_scTM':     round(float(samples['max_tm'].mean()),     3),
    }

    # --- Diversity over all samples ---
    all_dir = work / 'all_pdbs'
    stage_backbones(samples, all_dir)
    try:
        all_cluster_tsv = foldseek_cluster(args.foldseek_bin, all_dir,
                                           args.tm_cluster_cutoff, work / 'all_cluster')
        n_clusters_all = count_clusters(all_cluster_tsv)
    except subprocess.CalledProcessError as e:
        print(f"[eval] foldseek easy-cluster (all) failed: {e.stderr.decode(errors='ignore')[:500]}")
        n_clusters_all = None
    summary['n_clusters_all']  = n_clusters_all
    summary['diversity_all']   = round(n_clusters_all / n_total, 4) if n_clusters_all else None
    print(f"[eval] diversity_all: {n_clusters_all}/{n_total} = {summary['diversity_all']}")

    # --- Diversity over designable subset ---
    designable = samples[samples['designable']]
    summary['n_clusters_designable'] = None
    summary['diversity_designable']  = None
    if len(designable) >= 2:
        des_dir = work / 'des_pdbs'
        stage_backbones(designable, des_dir)
        try:
            des_cluster_tsv = foldseek_cluster(args.foldseek_bin, des_dir,
                                               args.tm_cluster_cutoff, work / 'des_cluster')
            n_clusters_des = count_clusters(des_cluster_tsv)
            summary['n_clusters_designable'] = n_clusters_des
            summary['diversity_designable']  = round(n_clusters_des / len(designable), 4)
            print(f"[eval] diversity_designable: {n_clusters_des}/{len(designable)} = {summary['diversity_designable']}")
        except subprocess.CalledProcessError as e:
            print(f"[eval] foldseek easy-cluster (designable) failed: {e.stderr.decode(errors='ignore')[:500]}")
    elif len(designable) == 1:
        summary['n_clusters_designable'] = 1
        summary['diversity_designable']  = 1.0
        print("[eval] diversity_designable: only 1 designable sample -> 1.0 by convention")
    else:
        print("[eval] diversity_designable: skipped (no designable samples)")

    # --- Novelty vs PDB (on designable samples only) ---
    summary['mean_novelty']      = None
    summary['median_novelty']    = None
    summary['pct_novel_tm_lt_0p5'] = None
    summary['n_novelty_queried'] = 0
    if args.skip_novelty:
        print("[eval] novelty: skipped (--skip_novelty)")
    elif len(designable) == 0:
        print("[eval] novelty: skipped (no designable samples)")
    elif not os.path.exists(args.pdb_db) and not glob.glob(args.pdb_db + '*'):
        print(f"[eval] novelty: skipped (target DB not found at {args.pdb_db})")
    else:
        nov_dir = work / 'nov_pdbs'
        stage_backbones(designable, nov_dir)
        try:
            hits = foldseek_search(args.foldseek_bin, nov_dir, args.pdb_db, work / 'novelty')
            if hits.empty:
                print("[eval] novelty: no Foldseek hits returned")
            else:
                per_q = hits.groupby('query')['alntmscore'].max().reset_index()
                # Queries with no hits will be missing from `per_q` -> assume max_tm = 0 (novel).
                all_qs = {pathlib.Path(p).stem for p in nov_dir.glob('*.pdb')}
                missing = all_qs - set(per_q['query'])
                if missing:
                    per_q = pd.concat([per_q, pd.DataFrame({'query': list(missing),
                                                            'alntmscore': [0.0]*len(missing)})])
                per_q['novelty'] = 1.0 - per_q['alntmscore']
                summary['n_novelty_queried']    = len(per_q)
                summary['mean_novelty']         = round(float(per_q['novelty'].mean()), 4)
                summary['median_novelty']       = round(float(per_q['novelty'].median()), 4)
                summary['pct_novel_tm_lt_0p5']  = round(float((per_q['alntmscore'] < 0.5).mean()), 4)
                print(f"[eval] novelty: mean={summary['mean_novelty']} "
                      f"median={summary['median_novelty']} "
                      f"pct(max_pdb_tm<0.5)={summary['pct_novel_tm_lt_0p5']}")
        except subprocess.CalledProcessError as e:
            print(f"[eval] foldseek easy-search failed: {e.stderr.decode(errors='ignore')[:500]}")

    # --- Write summary row ---
    out_path = pathlib.Path(args.out_csv)
    header = list(summary.keys())
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with open(out_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow(summary)
    print(f"[eval] appended row to {out_path}")

    if not args.keep_tmp:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
