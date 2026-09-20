# Task 1 — Inductive Biases and Feature Representations

Dataset: STL-10 (common 224x224 base image, bicubic upsampling). Backbones (frozen):
ResNet-50 (IMAGENET1K_V2), ViT-B/16 (IMAGENET1K_V1), OpenCLIP ViT-B-32 (openai).
All settings: `configs/task1.yaml`. Seed 6304 everywhere.

## Layout
```
task1/
  configs/            task1.yaml + loader (shared names/paths)
  data/
    make_subset.py    splits, 500-image test subset, base images, image store
    transforms.py     grayscale, hue/palette, translation, patch shuffle
    make_cue_conflicts.py   (Part 3) AdaIN cue-conflict generation
    stl10_split_seed6304.json   saved image identifiers (generated)
  models/
    backbones.py      ResNet / ViT / CLIP wrappers + feature cache
    heads.py          linear heads + CLIP zero-shot classifier
  analysis/
    evaluate_bias.py  predictions, metrics, consistency, translation curve (+ shape bias later)
    feature_similarity.py   (Part 4) cosine representation stability
    representation.py       (Part 4) t-SNE / UMAP
  scripts/run_task1.py      single entry point (stages)
  results/            machine-readable results (committed)
```

## Reproduce
```bash
export PA1_STORAGE=/path/to/storage        # datasets, feature cache, checkpoints
python -m task1.scripts.run_task1 --stage all
```
Stages: `subset`, `features`, `heads`, `clean`, `color`, `patch`, `translation`, `summary`.
Translation image files are deleted after feature extraction (deterministic, rebuildable);
pass `--keep-images` to keep them.

## Outputs
| File | Content |
|---|---|
| `data/stl10_split_seed6304.json` | train/val/test-subset identifiers and labels |
| `results/heads/*_history.csv`, `summary.json` | head training curves, best epoch |
| `results/predictions/<condition>.csv` | per-image prediction + confidence, all 4 predictors |
| `results/metrics/<condition>.json` | accuracy, macro-F1, mean max confidence, per-class, confusion |
| `../report/tables/task1_clean_baseline.csv` | clean baseline table |
| `../report/tables/task1_interventions.csv` | clean / grayscale / colour / patch comparison |
| `results/metrics/translation_curve.csv` | accuracy + consistency vs displacement |
| `results/metrics/color_summary.csv` | colour intervention summary |
| `../report/figures/task1/` | figures |
