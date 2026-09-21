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
```
* ResNet-18 `IMAGENET1K_V1`, fc -> Identity (512-d feature), 7-class linear head, full fine-tuning.
* BatchNorm running mean/var frozen at ImageNet values (BN modules in eval mode after
  `model.train()`); gamma/beta trainable. Verified after training (`bn_running_stats_equal_imagenet`).
* AdamW lr 1e-4, wd 1e-4 (all parameters), constant LR, <= 30 epochs, early stopping after 5
  epochs without improved mean source-val macro-F1, seed 6304, fp16 autocast on GPU.
* Checkpoints: `$PA1_STORAGE/checkpoints/task2/<run>/best.pt` (network only). A finished run is
  never retrained without `--fresh`; `run_meta.json` records the SHA-256 of `best.pt`, so the
  Task 3 ERM baseline can be verified to be this exact file.
* Results per run: `task2/results/<run>/{train_steps.csv, train_epochs.csv, source_val.json, run_meta.json}`.

## Layout
```
configs/     base.yaml (shared settings), source_only.yaml
models/      backbone.py (ResNet-18 + frozen-BN policy), classifier_head.py (head + Net)
methods/     base.py (common interface), source_only.py
evaluation/  metrics.py (per-domain / mean / worst metrics)
train.py     single training loop for all methods
```
