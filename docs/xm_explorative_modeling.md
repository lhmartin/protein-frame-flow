# Explorative Modeling (XM) for FrameFlow — experiment & findings

Interactive results: https://claude.ai/code/artifact/f25696cc-0581-4972-9689-20d986bd0ab4

## TL;DR

At **10k-step** training on PDB monomers **≤128 residues**, none of the coupling
interventions we tested improve backbone **designability** over the plain vanilla
independent-coupling baseline:

- **Best-of-K "explorative" selection (XM) does not help**, and larger K hurts
  (K=8 collapses the model).
- **Minibatch-OT ties vanilla** — it lowers the training loss but the gain does not
  transfer to samples, and it needs *more* inference steps, not fewer.
- **SE(3)-consistent OT hurts** — forcing the frames to follow the OT rotation
  inflates the rotation prior.
- **No arm generalizes past the 128-residue training limit.**

Vanilla independent coupling is the best and simplest. Two follow-ups confirm this:
a **50k-step retrain** (5× longer; ranking unchanged, larger K still trades diversity
for nothing) and a **longer-protein retrain** (PDB ≤256, vanilla vs K=4). At ≤256
vanilla beats K=4 by a *wider* margin than at ≤128, so the null result is **not** a
small-protein artifact. See the two follow-up sections below.

## What XM is

XM is a winner-take-all / best-of-K objective (scaled-up Multiple Choice Learning).
For flow matching it explores over the **noise**: for one example, draw K noise
samples that **share the same timestep t and the same ground-truth x₁**, score each
candidate coupling against the target, and backprop only the argmin. It is a
per-example, best-of-K surrogate for minibatch-OT coupling. The premise is that
conditional-mean objectives blur multimodal targets; letting the model commit to a
mode should sharpen it.

## Arms

| Arm | Coupling |
|-----|----------|
| **Vanilla** | independent (baseline) |
| **XM K=2 / 4 / 8** | best-of-K coupling selection (shared t, shared x₁) |
| **Minibatch-OT** | Hungarian assignment on Kabsch-aligned translations |
| **SE(3)-OT** | OT that *also* rotates the frames by the same Kabsch rotation |

All arms: 10k steps, seed 123, PDB filtered to ≤128 residues, otherwise identical.

## Implementation notes

Config flags (all no-ops when off; see `configs/base.yaml`, `configs/xm_pdb128.yaml`):

- `interpolant.xm.{enabled,K,selection,soft_temperature,selection_loss,warmup_steps,numeric_check}`
- `interpolant.trans.batch_ot` — minibatch-OT coupling on translations.
- `interpolant.trans.se3_ot` — SE(3)-consistent OT (also rotate frames).

Correctness points that mattered:

- **Shared-t is enforced** for XM (`corrupt_batch_xm` asserts every candidate shares
  t and x₁). Per-candidate t would let argmin trivially pick the easiest timestep.
- **Bit-identical guarantee**: XM with K=1 reproduces the vanilla loss exactly
  (0.0 diff over 50 steps); the K selection forwards run under `no_grad`, only the
  winner is back-propped (K forwards + 1 backward, memory flat in K).
- **Minibatch-OT had a latent indexing bug** in the historical code: it returned
  `aligned_nm_0[gt_perm, noise_perm]`, which mis-pairs `trans_0[k]` with `trans_1[k]`
  unless the assignment is an involution — empirically making the coupling *worse*
  than identity. Fixed to order by GT index (`_batch_ot`).
- **SE(3)-OT convention** is `frames → Rᵀ @ F` (translations use `pos @ R`), verified
  to reconstruct a rigid body to 3.8e-6 via `to_atom37`; plain OT's position/frame
  inconsistency measures 2.79 Å.

## Eval pipeline

Standalone, runs under a folding-capable environment (esm + ProteinMPNN) rather than
the training venv:

- `analysis/self_consistency.py` — per backbone: ProteinMPNN (CPU) → ESMFold (GPU) →
  best-of-8 Cα scRMSD / scTM. Skips already-folded samples (resumable).
- `analysis/eval_designability.py` — aggregates designability (scRMSD < 2 Å),
  structural diversity (foldseek TM-cluster count), and novelty (max pdbTM to PDB).

## Results (best-of-8, scRMSD < 2 Å)

### Baseline — 402 backbones/arm, lengths 70–160

| Arm | Designability | Distinct designable | Median scRMSD | Diversity |
|-----|---------------|---------------------|---------------|-----------|
| Vanilla | **15.2 %** | **38** | 6.68 | 0.84 |
| Minibatch-OT | 14.9 % | 31 | 8.26 | 0.87 |
| XM K=4 | 13.7 % | 32 | 6.94 | 0.81 |
| XM K=2 | 10.4 % | 25 | 7.50 | 0.85 |
| SE(3)-OT | 6.0 % | 17 | 8.74 | 0.92 |
| XM K=8 | 3.5 % | 11 | 9.08 | 0.94 |

### Designability by length (%)

| Length | Vanilla | K2 | K4 | K8 | OT | SE(3)-OT |
|--------|---------|----|----|----|----|----------|
| 70  | 40.3 | 34.3 | 34.3 | 17.9 | 47.8 | 20.9 |
| 90  | 25.4 | 22.4 | 23.9 | 1.5  | 20.9 | 9.0  |
| 110 | 9.0  | 6.0  | 13.4 | 1.5  | 9.0  | 4.5  |
| 128 | 13.4 | 0.0  | 9.0  | 0.0  | 9.0  | 0.0  |
| 140\* | 3.0 | 0.0  | 1.5  | 0.0  | 3.0  | 1.5  |
| 160\* | 0.0 | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  |

\* extrapolation beyond the 128-residue training limit — everything is ~0.

### Designability vs. inference steps (n=201/config, lengths 70/100/128)

| Model | 10 | 50 | 100 | 500 |
|-------|----|----|-----|-----|
| Vanilla | 8.0 | 20.9 | **27.9** | 23.4 |
| XM K=4 | 8.0 | 19.4 | 24.4 | 22.9 |
| Minibatch-OT | 8.0 | 16.4 | 12.9 | **26.4** |
| XM K=2 | 10.0 | 14.4 | 16.9 | 19.4 |
| SE(3)-OT | 4.0 | 10.0 | 7.0 | 13.4 |

The "exploration/OT buys fewer steps" claim does **not** hold: vanilla peaks highest;
OT needs *more* steps; K=2 is flatter only because it plateaus lower.

## 50k-step follow-up (2026-08)

We retrained the four viable arms (vanilla, XM K=2/4/8, minibatch-OT) to **50k
steps** — 5× longer — to test the "needs longer training" limitation below, still
on PDB ≤128 residues, seed 123, otherwise identical. Full metrics now include
foldseek diversity (TM-cluster count, cutoff 0.5) and novelty (max pdbTM to the
full PDB). Eval: 402 backbones/arm, lengths 70–160, best-of-8, scRMSD < 2 Å.
Consolidated CSV: `xm_eval_results/eval50k/eval50k_full_summary.csv`.

| Arm | Designability | Median scRMSD | Diversity (all) | Clusters (all) | Diversity (des.) | Median novelty |
|-----|---------------|---------------|-----------------|----------------|------------------|----------------|
| **Vanilla** | **69.4 %** | 1.44 | 0.415 | 167 | 0.323 | 0.197 |
| XM K=4 | 67.4 % | 1.36 | 0.401 | 161 | 0.306 | 0.203 |
| XM K=8 | 66.2 % | 1.44 | 0.368 | 148 | 0.308 | 0.205 |
| Minibatch-OT | 63.2 % | 1.52 | 0.547 | 220 | 0.402 | 0.227 |
| XM K=2 | 30.1 % | 4.26 | 0.612 | 246 | 0.455 | 0.215 |

Longer training raises everyone (designability 15 %→69 % for vanilla) but does
**not** change the ranking or the conclusion:

- **XM still does not help, and larger K monotonically drops diversity.** As K
  rises, distinct clusters fall (vanilla 167 → K4 161 → K8 148) *and* designability
  edges down. The paper's mode-committing signature (diversity↓) is visible, but the
  designability it was supposed to buy never materializes — XM is strictly dominated
  by vanilla here.
- **Minibatch-OT is the diversity outlier** — most clusters (220) at a modest
  designability cost (63.2 %). If diversity were the goal, OT beats every XM arm.
- **K=2 is broken, not diverse** — its high cluster count is scattered *undesignable*
  junk (scRMSD 4.26).
- **Novelty is uninformative at ≤128 residues** — all arms ~0.20 median, and
  `pct(max pdbTM < 0.5) = 0` for every arm (every designable small backbone matches
  something in the PDB). It does not separate the arms.

## Longer-protein follow-up — PDB ≤256 (2026-08)

To test whether the null result is a **small-protein** artifact (at ≤128 residues
the target is barely multimodal, so exploration has little to exploit), we retrained
the two headline arms — **vanilla and XM K=4** — on PDB monomers **≤256 residues**
(median PDB monomer is 246, so this is the "typical protein" regime; 29k chains vs
7.6k at ≤128). Each arm warm-started from its own seed-matched 10k ≤128 base and
trained to 50k steps at cap 256, then the same eval (402 backbones/arm, lengths
100–256, best-of-8, scRMSD < 2 Å). CSV: `xm_eval_results/eval256/eval256_summary.csv`.

| Arm | Designability | Median scRMSD | Diversity (all) | Diversity (des.) | Median novelty |
|-----|---------------|---------------|-----------------|------------------|----------------|
| **Vanilla** | **29.9 %** | 3.53 | 0.774 | 0.733 | 0.247 |
| XM K=4 | 22.9 % | 4.97 | 0.766 | 0.674 | 0.236 |

**The null result holds — and XM is worse here, not better.** Vanilla beats K=4 by
~7 points (29.9 % vs 22.9 %), a *wider* margin than the ~2-point gap at ≤128, so the
small-protein hypothesis is refuted: XM does not help on longer, more multimodal
proteins. K=4 again shows the same mode-dropping direction (lower diversity among
designable, 0.674 vs 0.733) while buying no designability. Longer proteins are just
harder for both (vanilla 69 %→30 %; overall diversity ~0.42→0.77), which does not
change the ranking.

## Verdict

XM is **not useful**, and neither longer training (50k steps) nor longer proteins
(≤256) rescue it — at ≤256 it is clearly *worse* than vanilla. Its lower *training*
loss is a mechanical artifact of best-of-K selection and does not transfer to
designability; its extra diversity is largely undesignable. OT ties vanilla on
designability while giving the most diversity; SE(3)-OT hurts. Across every regime
tested — 10k/50k steps, ≤128/≤256 residues — plain independent coupling wins.
**Experiment complete.**

## Limitations & how to firm it up

These are **10k-step** models — lightly trained; absolute designability is low, so read
the ranking/shape, not the magnitude. Single seed, single length regime (≤128), and a
thermally-limited single-GPU eval. The open question is whether **longer training** (or
a harder, more multimodal data regime) would let XM's mode-committing help. To answer
that you would: (1) train the strongest arms (vanilla, K=2/4, OT) to convergence
(≥100k–200k steps) and re-run this eval; (2) track the designability↔diversity frontier
over training, not just the endpoint; (3) test on a regime where the coupling actually
matters (larger/more multimodal proteins), since at ≤128 residues the target is not very
multimodal and there is little for exploration to exploit.

## Reproduce

```bash
# Train an arm (10k steps, PDB<=128), GPU pinned + thermal-guarded via scratchpad scripts
python experiments/train_se3_flows.py --config-name xm_pdb128 \
  experiment.seed=123 experiment.wandb.name=<arm> \
  experiment.trainer.strategy=ddp +experiment.trainer.max_steps=10000 \
  interpolant.xm.enabled=<bool> interpolant.xm.K=<K> \
  interpolant.trans.batch_ot=<bool> interpolant.trans.se3_ot=<bool>

# Sample backbones, then evaluate
python experiments/inference_se3_flows.py --config-name xm_pdb128 \
  inference.ckpt_path=<ckpt> inference.samples.length_subset=[70,100,128] \
  inference.samples.samples_per_length=67 inference.interpolant.sampling.num_timesteps=100
python analysis/self_consistency.py --inference_dir <dir> --gpu 0
python analysis/eval_designability.py --inference_dir <dir> --out_csv summary.csv
```
