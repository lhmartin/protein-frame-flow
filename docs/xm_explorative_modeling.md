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

Vanilla independent coupling is the best and simplest.

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

## Verdict

XM is **not useful** at this training scale. Its lower *training* loss is a mechanical
artifact of best-of-K selection and does not transfer to designability; its extra
diversity is largely undesignable. OT ties vanilla; SE(3)-OT hurts.

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
