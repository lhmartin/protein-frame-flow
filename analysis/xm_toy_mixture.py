"""XM toy: a controlled multimodal SE(3)^N mixture — give XM its best shot.

This is the one experiment that can settle the "does winner-take-all best-of-K
coupling selection help on a *genuinely* multimodal target?" question in a
setting where the ground truth is known EXACTLY and all protein-specific
confounds (data scarcity, IPA capacity, eval-pipeline noise) are removed.

Design
------
* Target = explicit M-mode mixture of rigid templates T_1..T_M in SE(3)^N
  (N frames). Each training example draws a mode uniformly + tiny jitter, so the
  target is provably M-modal by construction. Mode separation is a single dial
  `delta` (Angstrom for translations; a coupled rotation spread).
* Corruption reuses the REAL data/interpolant.py: `corrupt_batch` /
  `corrupt_batch_xm` (shared-t / shared-x_1, IGSO(3) sigma=1.5 rotation prior +
  centered-Gaussian * NM_TO_ANG_SCALE translation prior).
* Loss reuses the EXACT model_step trans_loss + rots_vf_loss (same weights,
  clamps, t-normalization) — the same quantity XM's argmin selects on.
* Selection reuses the REAL XM rule: K no_grad candidate forwards sharing t and
  x_1, score = trans_loss + rots_vf_loss per example, backprop only the argmin.
* Velocity field = a tiny MLP (no IPA), CPU-trainable in seconds.
* Sampling reuses the interpolant's own Euler steps.

Because the templates are known, we measure against ground truth exactly:
* purity      : distance from each sample to its NEAREST template (lower=purer).
* blur rate   : fraction of samples closer to the mode-average ("blur point")
                than to any single template — the conditional-mean failure XM
                claims to fix.
* coverage    : how the samples spread across the M modes (catches XM collapse).

CPU-only by construction (CUDA is hidden below) so it never touches the GPU jobs.
"""
import os
# Hide the GPU: this experiment is CPU-only and must not disturb running jobs.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
import csv
import math
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from data import so3_utils
from data import utils as du
from data.interpolant import Interpolant

DEVICE = 'cpu'
SCALE = du.NM_TO_ANG_SCALE  # 10.0; the translation prior lives at this scale.

# Be a polite CPU citizen: cap threads so we don't starve the GPU jobs' CPU-side
# work (ProteinMPNN/ESMFold data handling) running concurrently.
torch.set_num_threads(int(os.environ.get('TOY_THREADS', '12')))


# --------------------------------------------------------------------------- #
# Interpolant config (faithful to configs/base.yaml, xm overridable)
# --------------------------------------------------------------------------- #
def make_interpolant(xm_enabled=False, K=1):
    base = OmegaConf.load(os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'base.yaml'))
    icfg = base.interpolant
    icfg.self_condition = False          # keep the mechanism test clean
    icfg.xm.enabled = xm_enabled
    icfg.xm.K = K
    interp = Interpolant(icfg)
    interp.set_device(DEVICE)
    _ = interp.igso3                      # warm the (cached) IGSO(3) table
    return interp


# --------------------------------------------------------------------------- #
# Ground-truth mixture of rigid templates in SE(3)^N
# --------------------------------------------------------------------------- #
def make_templates(N, M, delta, seed):
    """M centered translation configs + M rotation-frame configs.

    delta = mode separation. delta=0 -> all templates identical (unimodal);
    larger delta -> better-separated modes. Rotation spread scales with delta.
    """
    g = torch.Generator().manual_seed(seed)
    rot_spread = 0.20 * delta            # rad; couples rotation separation to delta

    base_t = torch.randn(N, 3, generator=g) * 8.0
    base_t -= base_t.mean(0, keepdim=True)
    base_R = so3_utils.rotvec_to_rotmat(torch.randn(N, 3, generator=g) * 0.5)

    T, R = [], []
    for _ in range(M):
        dt = torch.randn(N, 3, generator=g)
        dt -= dt.mean(0, keepdim=True)
        t_m = base_t + delta * dt
        t_m -= t_m.mean(0, keepdim=True)
        dr = torch.randn(N, 3, generator=g) * rot_spread
        R_m = so3_utils.apply_rotvec_to_rotmat(base_R, dr)
        T.append(t_m)
        R.append(R_m)
    return torch.stack(T), torch.stack(R)  # [M,N,3], [M,N,3,3]


def sample_data(tmpl_T, tmpl_R, B, jitter_t=0.2, jitter_r=0.03):
    """Draw a batch: pick a mode uniformly, add tiny jitter -> the M-modal target."""
    M, N = tmpl_T.shape[0], tmpl_T.shape[1]
    modes = torch.randint(0, M, (B,))
    trans_1 = tmpl_T[modes].clone()
    trans_1 += torch.randn(B, N, 3) * jitter_t
    trans_1 -= trans_1.mean(1, keepdim=True)
    rvec = torch.randn(B, N, 3) * jitter_r
    rotmats_1 = so3_utils.apply_rotvec_to_rotmat(tmpl_R[modes].clone(), rvec)
    return {
        'trans_1': trans_1,
        'rotmats_1': rotmats_1,
        'res_mask': torch.ones(B, N),
        'diffuse_mask': torch.ones(B, N),
    }


# --------------------------------------------------------------------------- #
# Tiny MLP velocity field (predicts clean x_1 = pred_trans, pred_rotmats)
# --------------------------------------------------------------------------- #
class ToyVF(nn.Module):
    def __init__(self, N, hidden=256):
        super().__init__()
        self.N = N
        in_dim = N * 3 + N * 9 + 1
        out_dim = N * 3 + N * 3
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, batch):
        trans_t = batch['trans_t']
        rotmats_t = batch['rotmats_t']
        t = batch['r3_t']
        B = trans_t.shape[0]
        x = torch.cat([
            trans_t.reshape(B, -1),
            rotmats_t.reshape(B, -1),
            t.reshape(B, 1),
        ], dim=-1)
        out = self.net(x)
        pred_trans = out[:, :self.N * 3].reshape(B, self.N, 3)
        pred_rotvec = out[:, self.N * 3:].reshape(B, self.N, 3)
        pred_rotmats = so3_utils.rotvec_to_rotmat(pred_rotvec)
        return {'pred_trans': pred_trans, 'pred_rotmats': pred_rotmats}


# Exact replica of model_step's trans_loss + rots_vf_loss (mask all ones).
def se3_losses(pred, batch):
    gt_trans = batch['trans_1']
    gt_rot = batch['rotmats_1']
    rotmats_t = batch['rotmats_t']
    r3_t = batch['r3_t']
    so3_t = batch['so3_t']
    N = gt_trans.shape[1]
    r3_norm = 1 - torch.min(r3_t[..., None], torch.tensor(0.9))
    so3_norm = 1 - torch.min(so3_t[..., None], torch.tensor(0.9))
    loss_denom = N * 3

    trans_err = (gt_trans - pred['pred_trans']) / r3_norm * 0.1  # trans_scale
    trans_loss = 2.0 * torch.sum(trans_err ** 2, dim=(-1, -2)) / loss_denom
    trans_loss = torch.clamp(trans_loss, max=5)

    gt_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rot)
    pred_vf = so3_utils.calc_rot_vf(rotmats_t, pred['pred_rotmats'])
    rot_err = (gt_vf - pred_vf) / so3_norm
    rot_loss = 1.0 * torch.sum(rot_err ** 2, dim=(-1, -2)) / loss_denom
    return trans_loss, rot_loss  # each [B]


def xm_select(model, interp, batch, K):
    """The real argmin best-of-K: K no_grad scoring forwards sharing t + x_1,
    backprop only the winning candidate. Returns the winner noisy_batch."""
    cands = interp.corrupt_batch_xm(batch, K)
    with torch.no_grad():
        scores = []
        for nb in cands:
            tl, rl = se3_losses(model(nb), nb)
            scores.append(tl + rl)
        scores = torch.stack(scores, dim=0)      # [K, B]
    winner = torch.argmin(scores, dim=0)          # [B]
    nb = dict(cands[0])
    tr = torch.stack([c['trans_t'] for c in cands], 0)
    ro = torch.stack([c['rotmats_t'] for c in cands], 0)
    idx_t = winner.view(1, -1, 1, 1).expand(1, *tr.shape[1:])
    idx_r = winner.view(1, -1, 1, 1, 1).expand(1, *ro.shape[1:])
    nb['trans_t'] = torch.gather(tr, 0, idx_t).squeeze(0)
    nb['rotmats_t'] = torch.gather(ro, 0, idx_r).squeeze(0)
    return nb


def train(model, interp, tmpl_T, tmpl_R, steps, batch, K, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        data = sample_data(tmpl_T, tmpl_R, batch)
        if K > 1:
            nb = xm_select(model, interp, data, K)
        else:
            nb = interp.corrupt_batch(data)
        tl, rl = se3_losses(model(nb), nb)
        loss = (tl + rl).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.item())


# --------------------------------------------------------------------------- #
# ODE sampling (reuses the interpolant's own Euler steps)
# --------------------------------------------------------------------------- #
def _uniform_so3(B, N):
    return torch.tensor(
        Rotation.random(B * N).as_matrix(), dtype=torch.float32).reshape(B, N, 3, 3)


@torch.no_grad()
def sample(model, interp, N, n_samples, num_timesteps=100):
    model.eval()
    trans_t = torch.randn(n_samples, N, 3)
    trans_t = (trans_t - trans_t.mean(1, keepdim=True)) * SCALE
    rotmats_t = _uniform_so3(n_samples, N)
    ts = torch.linspace(interp._cfg.min_t, 1.0, num_timesteps)
    t_1 = ts[0]
    for t_2 in ts[1:]:
        batch = {
            'trans_t': trans_t, 'rotmats_t': rotmats_t,
            'r3_t': torch.ones(n_samples, 1) * t_1,
            'so3_t': torch.ones(n_samples, 1) * t_1,
        }
        out = model(batch)
        d_t = (t_2 - t_1).item()
        trans_t = interp._trans_euler_step(d_t, t_1, out['pred_trans'], trans_t)
        rotmats_t = interp._rots_euler_step(d_t, t_1, out['pred_rotmats'], rotmats_t)
        t_1 = t_2
    # final step: use the model's clean prediction directly (mirrors Interpolant.sample)
    batch = {
        'trans_t': trans_t, 'rotmats_t': rotmats_t,
        'r3_t': torch.ones(n_samples, 1) * ts[-1],
        'so3_t': torch.ones(n_samples, 1) * ts[-1],
    }
    out = model(batch)
    return out['pred_trans'], out['pred_rotmats']


# --------------------------------------------------------------------------- #
# Ground-truth metrics against the known templates
# --------------------------------------------------------------------------- #
def _project_so3(R):
    """Nearest rotation matrix to R via SVD (per leading dim)."""
    U, _, Vh = torch.linalg.svd(R)
    Rp = U @ Vh
    det = torch.linalg.det(Rp)
    Vh = Vh.clone()
    Vh[..., -1, :] *= det[..., None]
    return U @ Vh


def _d_trans(x, tmpl):
    # x:[S,N,3], tmpl:[M,N,3] -> [S,M] rmsd over N,3
    diff = x[:, None] - tmpl[None]                    # [S,M,N,3]
    return torch.sqrt((diff ** 2).mean(dim=(-1, -2)))


def _d_rot(R, tmplR):
    # R:[S,N,3,3], tmplR:[M,N,3,3] -> [S,M] mean geodesic (rad)
    S, N = R.shape[0], R.shape[1]
    M = tmplR.shape[0]
    Re = R[:, None].expand(S, M, N, 3, 3)
    Te = tmplR[None].expand(S, M, N, 3, 3)
    return so3_utils.geodesic_dist(Re, Te).mean(dim=-1)  # [S,M]


def evaluate(trans, rots, tmpl_T, tmpl_R):
    M = tmpl_T.shape[0]
    dt = _d_trans(trans, tmpl_T)              # [S,M] Angstrom
    dr = _d_rot(rots, tmpl_R)                 # [S,M] rad
    # reference inter-template spacing to normalize the two channels
    ref_t = _d_trans(tmpl_T, tmpl_T)
    ref_r = _d_rot(tmpl_R, tmpl_R)
    off = ~torch.eye(M, dtype=torch.bool)
    ref_t = ref_t[off].mean().clamp(min=1e-6)
    ref_r = ref_r[off].mean().clamp(min=1e-6)
    comb = dt / ref_t + dr / ref_r           # [S,M] normalized combined distance

    near = comb.min(dim=1)
    assign = near.indices                     # [S] nearest mode
    near_val = near.values
    # second-nearest margin (confidence of the assignment)
    comb_sorted = comb.sort(dim=1).values
    margin = (comb_sorted[:, 1] - comb_sorted[:, 0])

    # blur point = mode average (the conditional-mean the model is tempted toward)
    blur_T = tmpl_T.mean(0, keepdim=True)          # [1,N,3]
    blur_R = _project_so3(tmpl_R.mean(0)).unsqueeze(0)  # [1,N,3,3]
    bdt = _d_trans(trans, blur_T)[:, 0] / ref_t
    bdr = _d_rot(rots, blur_R)[:, 0] / ref_r
    blur_comb = bdt + bdr
    blur_rate = (blur_comb < near_val).float().mean()

    counts = torch.bincount(assign, minlength=M).float()
    shares = counts / counts.sum()
    modes_covered = int((shares > 0.05).sum())
    # balance: min share relative to uniform (1.0 = perfectly balanced)
    balance = float((shares.min() * M).item())

    return {
        'purity_trans_A': float(dt.gather(1, assign[:, None]).mean().item()),
        'purity_rot_rad': float(dr.gather(1, assign[:, None]).mean().item()),
        'purity_comb': float(near_val.mean().item()),
        'margin': float(margin.mean().item()),
        'blur_rate': float(blur_rate.item()),
        'modes_covered': modes_covered,
        'balance': balance,
        'shares': shares.tolist(),
    }


# --------------------------------------------------------------------------- #
def run_one(delta, M, N, K, seed, steps, batch, n_eval, tsteps):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tmpl_T, tmpl_R = make_templates(N, M, delta, seed=1000 + seed)
    interp = make_interpolant(xm_enabled=(K > 1), K=K)
    model = ToyVF(N)
    final_loss = train(model, interp, tmpl_T, tmpl_R, steps, batch, K)
    trans, rots = sample(model, interp, N, n_eval, tsteps)
    m = evaluate(trans, rots, tmpl_T, tmpl_R)
    m['final_loss'] = final_loss
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default='xm_eval_results/toy/toy_results.csv')
    args = ap.parse_args()

    N, M = 8, 4
    if args.quick:
        deltas = [6.0]
        Ks = [1, 4]
        seeds = [0]
        steps, batch, n_eval, tsteps = 400, 256, 256, 50
    else:
        deltas = [2.0, 6.0]
        Ks = [1, 2, 4, 8]
        seeds = [0, 1, 2]
        steps, batch, n_eval, tsteps = 3000, 256, 1024, 100

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for delta in deltas:
        for K in Ks:
            for seed in seeds:
                m = run_one(delta, M, N, K, seed, steps, batch, n_eval, tsteps)
                arm = 'vanilla' if K == 1 else f'XM_K{K}'
                row = {'delta': delta, 'arm': arm, 'K': K, 'seed': seed, **m}
                rows.append(row)
                print(f"delta={delta} {arm:8s} seed={seed} | "
                      f"purity_comb={m['purity_comb']:.3f} "
                      f"trans={m['purity_trans_A']:.2f}A rot={m['purity_rot_rad']:.3f} "
                      f"blur={m['blur_rate']:.3f} covered={m['modes_covered']}/{M} "
                      f"balance={m['balance']:.2f} loss={m['final_loss']:.3f}",
                      flush=True)

    keys = ['delta', 'arm', 'K', 'seed', 'purity_comb', 'purity_trans_A',
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
