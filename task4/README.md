# Task 4 — Open-Set Recognition (CIFAR-10 known, CIFAR-100 unknowns)

Run everything from the repository root. Compute: Kaggle, 2× T4 (environment: `results/environment_kaggle.txt`).
Storage: `$PA1_STORAGE/checkpoints/task4/<run>/{best.pt,last.pt,status.json}` (not in git);
`$PA1_FAST_DATA/cifar` (decoded data cache); `task4/cache/<run>.npz` (saved logits/features, git-ignored).

## Leakage guarantees
* `train.py` and `evaluation/bn_diagnostic.py` set `PA1_FORBID_UNKNOWN=1`; under this flag
  `data/cifar100_unknowns.py` raises `UnknownAccessError`. No CIFAR-100 training image is ever loaded.
* Checkpoint selection: CIFAR-10 validation accuracy on the 10 known-class logits only.
* `extract_outputs.py --stage lock` refuses uncommitted code, checks every `best.pt` SHA-256 against the
  hash recorded at training time (`results/<run>/best.json`) and writes `results/final/lock.json` (committed
  before extraction). CIFAR-100 loads only for `purpose="final_evaluation"` after the lock, re-verifying the
  hashes; every access is logged in `results/unknown_access.jsonl` (one entry).
* Thresholds use CIFAR-10 validation unknownness only (95th percentile; accept iff u ≤ τ).

## Commands
```bash
python -m task4.data.cifar10                                  # download (pinned, SHA-256), cache, verify split
python -m task4.data.make_splits                              # (re)creates data/splits/cifar10_seed6304.json
python -m task4.train --config task4/configs/vanilla.yaml     # add --smoke for a tiny end-to-end test
python -m task4.train --config task4/configs/gcsc.yaml
python -m task4.train --config task4/configs/proser.yaml          # needs vanilla/best.pt
python -m task4.train --config task4/configs/proser_cleanbn.yaml  # needs vanilla/best.pt
python -m task4.evaluation.bn_diagnostic                      # CIFAR-10-only BN diagnostic for PROSER
python -m task4.extract_outputs --stage lock --proser-main proser_cleanbn
python -m task4.extract_outputs --stage extract               # the only CIFAR-100 access
python -m task4.evaluate_osr                                  # tables, figures, failures (CPU)
```
Training resumes exactly from `last.pt` after a disconnect (re-run the same command); a finished run needs `--fresh`.
Test hooks (sandbox only): `PA1_TEST_STOP_AFTER=<epoch>`, `PA1_T4_LIMIT=<n>`, `--set data.train_subset=<n>`.

## Data
* HF parquet mirrors `uoft-cs/cifar10` @ `0b27149…`, `uoft-cs/cifar100` @ `aadb3af…`, SHA-256 verified
  (`data/cifar10.py`). Sampled images were verified byte-identical to the official CIFAR-10 archive.
  Split indices refer to HF row order.
* Split: per class, seeded permutation (numpy `default_rng(6304)`), first 10 % → validation: 45,000 / 5,000
  (500 per class). Final known-class evaluation: full CIFAR-10 test set (10,000).
* Unknowns: the 16 fixed CIFAR-100 test classes, 100 images each (near 800, far 800).

## Models and training
* `models/resnet_cifar.py`: torchvision ResNet-18, conv1 → 3×3 stride 1 (kaiming init), max-pool removed,
  random init, 32×32 inputs. f(x) = 512-d pooled feature, z(x) = 10 logits.
* Vanilla: crop 32 (pad 4, zeros) + flip, standard CIFAR-10 normalisation; SGD 0.1, momentum 0.9, wd 5e-4
  (all parameters), cosine (per epoch, η_min 0), batch 128, 100 epochs, seed 6304; fp16 AMP + channels_last +
  cudnn autotuning (seeds fixed, not bitwise deterministic); per-epoch data order seeded (SEED·1000 + epoch).
* GCSC: identical recipe + `RandAugment(num_ops=2, magnitude=9)` after crop/flip, before ToTensor/Normalize.
* PROSER (`methods/proser.py`, `methods/manifold_mixup.py`): from Vanilla `best.pt` (hash in
  `results/<run>/run_meta.json`) + `Linear(512, 5)` dummy head; SGD 1e-3, cosine, 50 epochs, batch 128,
  Vanilla augmentation. Paper formulation, class K+1 = max dummy logit. First half-batch:
  CE(ẑ, y) + β·CE(ẑ without y, K+1), β = 1. Second half-batch: manifold mixup after layer2 between
  different-class pairs, λ ~ Beta(2, 2) per pair, γ·CE(ẑ(h̃), K+1), γ = 0.1.
* `proser_cleanbn` (`configs/proser_cleanbn.yaml`): identical, except that the mixed pass normalises with its
  batch statistics without updating the layer3/layer4 BN running statistics (`mixed_bn_stats: frozen`).
  Motivation: PROSER validation accuracy fell while train accuracy stayed ≈ 1 and selection chose epoch 1;
  the CIFAR-10-only diagnostic (`results/diagnostics/proser_bn_diagnostic.json`) re-estimates BN statistics
  on clean training images. Main PROSER row chosen by a rule fixed before any CIFAR-100 access
  (`lock.json: proser_main`); both PROSER runs are evaluated.
* All four models were retrained once with identical code/configs after the first checkpoints were lost
  (not saved from an interactive session); reported numbers are from the retrained runs (commit `fd512ec`),
  earlier logs remain in git history.

## Scores (`scores/`), all computed from the same saved cache
MSP `1 − max softmax`; MLS `− max z`; Energy `− logsumexp z`; Mahalanobis `min_c (f−μ_c)ᵀΣ⁻¹(f−μ_c)` with class
means and one shared diagonal covariance (pooled within-class ML variance + 1e-6) from unaugmented CIFAR-10
training-split features (Vanilla); PROSER placeholder score as in the authors' evaluation code:
softmax over [10 logits, max dummy logit] at temperature 1024, `u = p_dummy − max p_known`.
Metrics: AUROC (unknown = positive) for known-vs-near / far / all; τ = 95th percentile of validation
unknownness; test acceptance, near/far/all rejection, FPR@95 = unknown acceptance; CSA from the 10 known logits.

## Outputs
| File | Content |
|---|---|
| `report/tables/task4_posthoc_scores.csv` | MSP / MLS / Energy / Mahalanobis on Vanilla |
| `report/tables/task4_model_comparison.csv` | Vanilla, GCSC, both PROSER runs (MLS + placeholder score) |
| `report/tables/task4_per_class_acceptance.csv` | acceptance rate per unknown class, every model/score |
| `report/tables/task4_absorption_mls.csv` | unknown class × predicted CIFAR-10 label among accepted (MLS) |
| `report/tables/task4_score_agreement.csv` | Spearman correlation and decision disagreement between scores |
| `report/tables/task4_failures_vanilla_mls.csv` | most confidently accepted unknown per class (Vanilla + MLS) |
| `report/figures/task4/task4_score_distributions_roc.pdf` | MSP / MLS / Mahalanobis distributions + ROC |
| `report/figures/task4/task4_failures_vanilla_mls.pdf` | failure image grid |
| `report/figures/task4/task4_training_curves.pdf` | Vanilla/GCSC accuracy; PROSER accuracy and losses |
| `results/<run>/{train_epochs.csv, best.json, run_meta.json}` | training logs, selected epoch, checkpoint hash |
| `results/final/{lock.json, extract_meta.json, metrics.json, vanilla_mls_accepted_unknowns.csv}` | final evaluation |
| `results/diagnostics/proser_bn_diagnostic.json` | BN diagnostic (CIFAR-10 only) |

Files beyond the manual's suggested structure: `configs/proser_cleanbn.yaml`, `data/splits/` (split indices),
`evaluation/bn_diagnostic.py`, `scores/proser_placeholder.py`, `results/environment_kaggle.txt`.
RPL (optional) was not implemented.

## External code
PROSER was implemented from Zhou et al. (2021). The authors' repository
(github.com/zhoudw-zdw/CVPR21-Proser, no licence file) was consulted for the evaluation score and training
structure; no code was copied. Its differences from the implemented paper objective are listed in
`methods/proser.py`. RandAugment and ResNet-18 come from torchvision.
