# Task 3 — Domain Generalization (PACS, unseen target = Sketch)

Run everything from the repository root. Shared protocol, splits and leakage guards: see `task2/README.md`.

## Leakage guarantees
* `task3/train.py` and `task3/source_diagnostics.py` set `PA1_FORBID_TARGET=1` before any data access.
  Under this flag `shared/pacs.py` drops every Sketch row while reading the parquet (no Sketch image or
  label is held in memory) and any Sketch dataset request raises `TargetAccessError`.
* Checkpoint selection: mean macro-F1 over the three source validation splits (`selection/source_validation.py`).
* `task3/evaluate_sketch.py` is the only Task 3 code that loads Sketch. It first writes
  `results/final/lock.json` (commit, checkpoint SHA-256s, hashes of the source-side diagnostics and of the
  study pre-registration) and refuses to run with uncommitted training/selection code. Labelled access is
  logged in `results/target_label_access.jsonl`.
* No Task 2 Sketch result was used to choose any Task 3 setting: the manual fixes λ_DG = 1, ρ = 0.05;
  the controlled study (λ_DG) was chosen because it matches the Task 2 DAN study knob and values.

## Commands
```
python -m task3.train --config task3/configs/erm.yaml      # ERM: verify + re-evaluate reused checkpoint (no training)
python -m task3.train --config task3/configs/dan_dg.yaml   # DAN-DG, λ_DG = 1
python -m task3.train --config task3/configs/sam.yaml      # SAM, ρ = 0.05
python -m task3.train --config task3/configs/dan_dg.yaml --set method.lambda_dg=0.1 --run-name dan_dg_l0p1
python -m task3.train --config task3/configs/dan_dg.yaml --set method.lambda_dg=10  --run-name dan_dg_l10
python -m task3.source_diagnostics                         # source val, separability, sharpness, curves (no Sketch)
python -m task3.evaluate_sketch --confirm-final            # lock -> Sketch -> tables/figures
python -m task3.evaluate_sketch --stage analyze            # recompute from cache
```
Add `--smoke` to a training command for a 2 × 5-step test in temporary folders.

## Methods
* **ERM** = Task 2 `source_only/best.pt`, reused unchanged: SHA-256 in `configs/erm.yaml` must equal the file
  and `task2/results/source_only/run_meta.json`; its source-val metrics are re-evaluated and compared with the
  Task 2 record. L_ERM = 1/3 Σ_e R_e realised by 8+8+8 domain-balanced batches.
* **DAN-DG**: CE + λ_DG/3 Σ_{e<e'} MMD²(F(X_e), F(X_e')) on the 512-d feature, pairs (P,A), (P,C), (A,C) of every
  8+8+8 batch; the Task 2 `mmd2` unchanged (biased estimate, RBF kernels {0.5,1,2} × median pairwise squared
  distance of the pair's 16 features, detached). Logged: `mmd` (mean) and `mmd_PA/PC/AC`.
  Note: the biased estimate contains the kernel diagonal, so its level depends on batch size (8 vs 8 here,
  24 vs 24 in Task 2 DAN); logged MMD values are not comparable across tasks, gradients are unaffected.
* **SAM** (Foret et al., 2021), non-adaptive, ρ = 0.05: ε = ρ g/‖g‖₂ over all trainable parameters (backbone incl.
  BN γ/β, head); second forward/backward at θ+ε on the same augmented batch; θ restored from a saved copy; AdamW
  step (lr 1e-4, wd 1e-4) with the second gradient. BN running statistics frozen in both passes. fp32.
  Same epoch definition as ERM (203 batches), i.e. twice the forward/backward passes. Logged: loss at θ
  (`cls_loss`), at θ+ε (`sam_loss`), `sam_gap`, `grad_norm`. Checked against an independent reference
  implementation (max |Δθ| ≈ 6e-8 after 3 steps).
* All trained methods use the single Task 2 loop (`task2/train.py`) with a Task 3 method registry. The loop calls
  `method.update(...)`; the default in `task2/methods/base.py` is the original single step and was verified
  bit-identical to the pre-hook code (model, optimiser state and step logs after 10 smoke steps, source_only and dan).

## Diagnostics (`source_diagnostics.py`, no Sketch)
* Per-source val accuracy / macro-F1, mean, worst (fp32).
* Source-domain separability: 512-d source-val features balanced to 334 per domain (smallest split; seeded
  subsample identical for all models), stratified 70/30 (seed 6304), StandardScaler + multinomial logistic
  regression C = 1, balanced; held-out accuracy (chance 1/3). Task 2 probe code reused.
* Sharpness proxy: fixed batch of 32 val images per source (seed 6304; IDs in `results/source_side/sharpness_batch_ids.json`),
  model in eval mode, mean CE; Δ = L(θ + 0.05 g/‖g‖) − L(θ), g over all trainable parameters; parameters restored.

## Controlled study
λ_DG ∈ {0.1, 1, 10}; λ_DG = 1 is the main DAN-DG run. Expectations were written before the study runs in
`results/study_preregistration.md` (hash recorded in the lock).

## Final analysis extras (after the lock, analysis only)
* Source-val vs Sketch separability of the Task 3 models with the Task 2 probe (same Sketch subset as Task 2).
* Three-way source separability of the Task 2 models (from the Task 2 feature cache) for a matched comparison.
* Check that ERM's Sketch predictions equal Task 2 Source-only's (same checkpoint).

## Outputs
`results/<run>/{train_steps.csv, train_epochs.csv, source_val.json, run_meta.json}`, `results/erm/`,
`results/source_side/`, `results/final/{lock.json, metrics.json, flip_examples.csv}`,
`report/tables/task3_*.csv`, `report/figures/task3/*.pdf|png`.
Checkpoints: `$PA1_STORAGE/checkpoints/task3/<run>/best.pt`; caches: `$PA1_STORAGE/cache/task3/`.
