"""XM blur test on a REAL multimodal PDB target (the "real-fold screen").

Same experiment as analysis/xm_toy_mixture.py -- reusing the REAL corruption,
the EXACT model_step loss, the REAL XM argmin selection, the interpolant's own
ODE sampling, and the ground-truth blur/purity/balance metrics -- but the modes
are real, structurally-distinct PDB folds (built by analysis/real_folds_prep.py)
and each training example is a REAL deposited structure drawn from a family. This
removes the "the toy's modes were random Gaussians / the model separates them by
a non-protein cue" objection: here the model must tell apart real protein folds
and reproduce real geometry.

Still CPU-only (xm_toy_mixture hides the GPU on import) so it never touches the
running GPU jobs. The velocity field is the same tiny MLP -- if vanilla still
produces zero blur here, and does so across a capacity sweep, the null result is
not a small-model artifact.

Run: PYTHONPATH=. .venv/bin/python analysis/xm_real_folds.py \
        --families xm_eval_results/real_folds/families_L129.pt
"""
import argparse
import csv
import os

import numpy as np
import torch

# Importing xm_toy_mixture sets CUDA_VISIBLE_DEVICES='' -> stays CPU-only, and
# gives us the faithful corruption/loss/selection/sampling/metrics unchanged.
from analysis.xm_toy_mixture import (
    make_interpolant, ToyVF, se3_losses, xm_select, sample, evaluate,
)
from data import so3_utils


def real_sample_data(families, B, M, jitter_t, jitter_r, gen):
    """Draw a batch of REAL structures: pick a mode uniformly, then a real member.

    Small optional jitter (Angstrom / rad) only smooths the finite member set; it
    is negligible vs the tens-of-Angstrom spacing between distinct folds.
    """
    modes = torch.randint(0, M, (B,), generator=gen)
    Ts, Rs = [], []
    for m in modes.tolist():
        fam = families[m]
        j = int(torch.randint(0, fam['trans'].shape[0], (1,), generator=gen).item())
        Ts.append(fam['trans'][j])
        Rs.append(fam['rot'][j])
    trans_1 = torch.stack(Ts)
    rotmats_1 = torch.stack(Rs)
    if jitter_t > 0:
        trans_1 = trans_1 + torch.randn(trans_1.shape, generator=gen) * jitter_t
    trans_1 = trans_1 - trans_1.mean(1, keepdim=True)
    if jitter_r > 0:
        rvec = torch.randn(rotmats_1.shape[:-1], generator=gen) * jitter_r
        rotmats_1 = so3_utils.apply_rotvec_to_rotmat(rotmats_1, rvec)
    N = trans_1.shape[1]
    return {
        'trans_1': trans_1,
        'rotmats_1': rotmats_1,
        'res_mask': torch.ones(B, N),
        'diffuse_mask': torch.ones(B, N),
    }


def train(model, interp, families, M, steps, batch, K, jitter_t, jitter_r, seed,
          lr=1e-4, optimizer='adamw'):
    """Optimizer defaults match the REAL FrameFlow trainer: AdamW, lr=1e-4, no
    gradient clipping (configs/base.yaml:85-86, models/flow_module.py:547-550).
    The earlier Adam lr=1e-3 setting was 10x the real lr and diverged vanilla
    seeds to NaN -- an artifact of the too-high lr, not of the coupling."""
    gen = torch.Generator().manual_seed(1234 + seed)
    Opt = torch.optim.AdamW if optimizer == 'adamw' else torch.optim.Adam
    opt = Opt(model.parameters(), lr=lr)
    model.train()
    loss = torch.tensor(float('nan'))
    skipped = 0
    for _ in range(steps):
        data = real_sample_data(families, batch, M, jitter_t, jitter_r, gen)
        nb = xm_select(model, interp, data, K) if K > 1 else interp.corrupt_batch(data)
        tl, rl = se3_losses(model(nb), nb)
        loss = (tl + rl).mean()
        opt.zero_grad()
        loss.backward()
        # data/so3_utils.rotmat_to_rotvec has a log-map singularity at theta~=pi:
        # sqrt(0) in the pi-branch leaks a NaN gradient (0*inf) even though the
        # forward loss is finite. It fires for the RARE step whose rotation error
        # lands near 180deg. The real trainer shares this util but rarely hits it
        # (the IPA fits rotations tightly); this weak MLP proxy lingers in the
        # high-error regime, so we skip the poisoned step instead of letting it
        # NaN the weights. Clipping can't fix it -- a NaN grad-norm stays NaN.
        finite_grad = all(p.grad is None or torch.isfinite(p.grad).all()
                          for p in model.parameters())
        if not finite_grad:
            skipped += 1
            continue
        opt.step()
    if skipped:
        print(f'    [skipped {skipped}/{steps} non-finite-grad steps '
              f'(so3 log-map pi singularity)]', flush=True)
    return float(loss.item())


def run_one(fam_data, K, seed, hidden, steps, batch, n_eval, tsteps, jitter_t,
            jitter_r, lr, optimizer):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N, M = fam_data['N'], fam_data['M']
    families = fam_data['families']
    tmpl_T, tmpl_R = fam_data['tmpl_T'], fam_data['tmpl_R']
    interp = make_interpolant(xm_enabled=(K > 1), K=K)
    model = ToyVF(N, hidden=hidden)
    final_loss = train(model, interp, families, M, steps, batch, K,
                       jitter_t, jitter_r, seed, lr=lr, optimizer=optimizer)
    trans, rots = sample(model, interp, N, n_eval, tsteps)
    m = evaluate(trans, rots, tmpl_T, tmpl_R)
    m['final_loss'] = final_loss
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--families', default='xm_eval_results/real_folds/families_L129.pt')
    ap.add_argument('--out', default='xm_eval_results/real_folds/real_results.csv')
    ap.add_argument('--hidden', type=int, nargs='+', default=[512])
    ap.add_argument('--Ks', type=int, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--n_eval', type=int, default=512)
    ap.add_argument('--tsteps', type=int, default=100)
    ap.add_argument('--jitter_t', type=float, default=0.5)
    ap.add_argument('--jitter_r', type=float, default=0.02)
    ap.add_argument('--lr', type=float, default=1e-4,
                    help='matches real trainer (base.yaml lr=1e-4)')
    ap.add_argument('--optimizer', default='adamw', choices=['adamw', 'adam'])
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    if args.quick:
        args.hidden, args.Ks, args.seeds = [512], [1, 4], [0]
        args.steps, args.batch, args.n_eval, args.tsteps = 300, 64, 128, 50

    fam_data = torch.load(args.families)
    print(f"real families: M={fam_data['M']} modes, N={fam_data['N']} residues, "
          f"member counts={[f['trans'].shape[0] for f in fam_data['families']]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for hidden in args.hidden:
        for K in args.Ks:
            for seed in args.seeds:
                m = run_one(fam_data, K, seed, hidden, args.steps, args.batch,
                            args.n_eval, args.tsteps, args.jitter_t, args.jitter_r,
                            args.lr, args.optimizer)
                arm = 'vanilla' if K == 1 else f'XM_K{K}'
                rows.append({'hidden': hidden, 'arm': arm, 'K': K, 'seed': seed, **m})
                print(f"hidden={hidden} {arm:8s} seed={seed} | "
                      f"purity_comb={m['purity_comb']:.3f} "
                      f"trans={m['purity_trans_A']:.2f}A rot={m['purity_rot_rad']:.3f} "
                      f"blur={m['blur_rate']:.3f} covered={m['modes_covered']}/{fam_data['M']} "
                      f"balance={m['balance']:.2f} loss={m['final_loss']:.3f}",
                      flush=True)

    keys = ['hidden', 'arm', 'K', 'seed', 'purity_comb', 'purity_trans_A',
            'purity_rot_rad', 'margin', 'blur_rate', 'modes_covered', 'balance',
            'final_loss', 'shares']
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in keys})
    print(f"\nwrote {args.out}  ({len(rows)} rows)")


if __name__ == '__main__':
    main()
