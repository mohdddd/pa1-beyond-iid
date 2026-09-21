# Task 2 — Unsupervised Domain Adaptation (PACS, target = Sketch)

Run everything from the repository root.

## Data and protocol (shared with Task 3)
```
python -m shared.pacs_protocol prepare   # download + verify PACS, make/verify splits, sample figure
python -m shared.pacs_protocol check     # batch composition, throughput, sampling/augmentation invariance
```
* PACS parquet from HuggingFace `flwrlabs/pacs`, pinned revision, SHA-256 verified
  (9,991 images; photo 1670, art_painting 2048, cartoon 2344, sketch 3929). Kept on local disk only.
* Splits: `shared/splits/pacs_sketch_seed6304.json` — per source domain, stratified 80/20
  (seed 6304). Sketch IDs carry no class name; target labels are never stored in the split file.
* Leakage guards (`shared/pacs.py`): labelled sketch data only with `purpose="final_evaluation"`
  (logged to `task2/results/target_label_access.jsonl`); `PA1_FORBID_TARGET=1` blocks all sketch access.
* Batches: 8 photo + 8 art_painting + 8 cartoon (+ 24 unlabeled sketch for adaptation methods).
  Per-domain generators and per-item augmentation seeds make the source images and their
  crops/flips identical for every method. One epoch = 203 steps (= ceil(4849 / 24)).

## Training
```
python -m task2.train --config task2/configs/source_only.yaml [--smoke] [--fresh]
python -m task2.train --config task2/configs/dan.yaml
python -m task2.train --config task2/configs/dann.yaml
python -m task2.train --config task2/configs/cdan.yaml
```
* DAN: loss = CE + lambda_mmd * MMD^2 on the 512-d feature, biased estimate, sum of RBF kernels with
  bandwidths {0.5, 1, 2} x median pairwise squared distance of the combined 48-example batch (detached).
* DANN: discriminator 512-256-ReLU-Dropout(0.5)-2 behind a GRL, alpha(p) = grl_max (2/(1+e^{-10p}) - 1),
  p = step / (30 x 203) (maximum budget, unaffected by early stopping); loss = CE(source) + 1 x CE_domain(source+target).
* CDAN: same as DANN but the discriminator sees vec(f ⊗ softmax(C(f))) (3584-d); no entropy
  conditioning, nothing detached. Discriminators are trained by the same AdamW optimiser.
* ResNet-18 `IMAGENET1K_V1`, fc -> Identity (512-d feature), 7-class linear head, full fine-tuning.
* BatchNorm running mean/var frozen at ImageNet values (BN modules in eval mode after
  `model.train()`); gamma/beta trainable. Verified after training (`bn_running_stats_equal_imagenet`).
* AdamW lr 1e-4, wd 1e-4 (all parameters, incl. discriminators), constant LR, <= 30 epochs, early
  stopping after 5 epochs without improved mean source-val macro-F1 (strict improvement), seed 6304.
* Precision: fp32 for ALL methods (`train.amp: false`). fp16 autocast overflowed in DANN/CDAN;
  `evaluation/stability_probe.py` + `results/diagnostics/dann_stability_fp16_vs_fp32.txt` show that the
  overflow is caused by exploding layer-4 activations under gradient reversal, which also occurs in fp32.
  All reported runs were retrained in fp32 through the same pipeline.
* Checkpoints: `$PA1_STORAGE/checkpoints/task2/<run>/best.pt` (network only). A finished run is
  never retrained without `--fresh`; `run_meta.json` records the SHA-256 of `best.pt`, so the
  Task 3 ERM baseline can be verified to be this exact file.
* Results per run: `task2/results/<run>/{train_steps.csv, train_epochs.csv, source_val.json, run_meta.json}`.

## Controlled study (DAN, λ_MMD ∈ {0.1, 1, 10})
```
python -m task2.train --config task2/configs/dan.yaml --set method.lambda_mmd=0.1 --run-name dan_lmmd0p1
python -m task2.train --config task2/configs/dan.yaml --set method.lambda_mmd=10  --run-name dan_lmmd10
```
λ = 1 is the main DAN run (`results/dan`), reused because every other setting is identical.

## Final evaluation (only place Sketch labels are used)
```
python -m task2.evaluate_final --confirm-final     # lock -> extract (GPU) -> analyze -> figures
python -m task2.evaluate_final --stage analyze     # recompute tables/figures from the feature cache
```
* Lock: all runs finished, every `best.pt` SHA-256 equal to its committed `run_meta.json`, no uncommitted
  training code; `results/final/lock.json` written before labels are loaded; access logged in
  `results/target_label_access.jsonl`.
* Domain separability: frozen 512-d features of all 1213 source-val images vs an equal-size seeded
  (6304) subset of Sketch (same images for every run); stratified 70/30 split (seed 6304); standardised
  features; logistic regression C = 1, balanced class weights; held-out accuracy (chance 0.5).
* Outputs: `results/final/{lock.json, metrics.json, flip_examples.csv}`;
  `report/tables/task2_{main_comparison, per_class_target, top_confusions, controlled_study}.csv`;
  `report/figures/task2/task2_{training_curves, controlled_study, per_class_delta, confusion_target, flip_examples}.pdf`.
  Features/predictions cached in `$PA1_STORAGE/cache/task2/final_<run>.npz`.

## Layout
```
configs/     base.yaml (shared settings), source_only.yaml, dan.yaml, dann.yaml, cdan.yaml
models/      backbone.py (ResNet-18 + frozen-BN policy), classifier_head.py (head + Net),
             domain_discriminator.py (discriminator, GRL, alpha schedule)
methods/     base.py (common interface), source_only.py, dan.py (mmd2 reused by Task 3), dann.py, cdan.py
evaluation/  metrics.py (per-domain / mean / worst metrics), domain_separability.py, class_analysis.py,
             curves.py (training-curve / study figures), stability_probe.py (diagnostic)
train.py         single training loop for all methods
evaluate_final.py  lock + final Sketch evaluation, separability, per-class analysis, tables, figures
```
