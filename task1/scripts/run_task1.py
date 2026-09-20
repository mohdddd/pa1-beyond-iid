"""Single entry point for Task 1. Each stage reads the previous stage's saved
outputs, so stages can be run one at a time (or all at once) and re-run safely.

  python -m task1.scripts.run_task1 --stage subset     # splits + 500-image subset + clean images
  python -m task1.scripts.run_task1 --stage features   # frozen features (+ CLIP zero-shot text weights)
  python -m task1.scripts.run_task1 --stage heads      # linear heads on train/val features
  python -m task1.scripts.run_task1 --stage clean      # clean baseline table
  python -m task1.scripts.run_task1 --stage all        # everything above, in order
(Later parts add stages: color, cue_conflict, translation, patch, representation.)
"""
import argparse
import time

import pandas as pd
import torch

from common.io import save_json, save_run_metadata
from common.paths import TABLE_DIR
from common.seed import set_seed
from task1.analysis.evaluate_bias import evaluate_condition
from task1.configs import BACKBONES, HEAD_OF, MODELS, RESULTS, cfg, split
from task1.data.make_subset import base_images, load_images, load_stl10, make_subset
from task1.models.backbones import extract, has_features, load_backbone, load_features, save_features
from task1.models.heads import has_zeroshot, save_head, save_zeroshot, train_linear_head

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def stage_features(conditions, overwrite=False):
    """'train'/'val' come from the STL-10 train partition; other names from the image store."""
    c, s = cfg(), split()
    set_seed(c["seed"])
    cache, train_ds = {}, None
    for bb_name in BACKBONES:
        todo = [cd for cd in conditions if overwrite or not has_features(bb_name, cd)]
        need_text = bb_name == "clip_vitb32" and not has_zeroshot()
        if not todo and not need_text:
            print(f"[{bb_name}] requested features already cached")
            continue
        bb = load_backbone(bb_name, DEVICE)
        for cd in todo:
            if cd not in cache:
                if cd in ("train", "val"):
                    train_ds = train_ds or load_stl10("train")
                    cache[cd] = base_images(train_ds, s[f"{cd}_idx"])
                else:
                    cache[cd] = load_images(cd)
            t0 = time.time()
            feats = extract(bb, cache[cd], c["feature_batch_size"], DEVICE)
            save_features(bb_name, cd, feats)
            print(f"[{bb_name}] {cd:<20} {feats.shape}  {time.time() - t0:.1f}s")
        if bb_name == "clip_vitb32":
            W, scale = bb.clip_text_classifier(s["classes"], c["zeroshot"]["prompt"])
            save_zeroshot(W, scale, c["zeroshot"]["prompt"], s["classes"])
            print(f"[clip] zero-shot classifier saved (logit scale {scale:.1f})")
        del bb
        torch.cuda.empty_cache()


def stage_heads():
    c, s = cfg(), split()
    summary = {}
    for bb in BACKBONES:
        name = HEAD_OF[bb]
        head, info = train_linear_head(
            load_features(bb, "train"), s["train_labels"], load_features(bb, "val"), s["val_labels"],
            num_classes=c["num_classes"], hcfg=c["head"], seed=c["seed"], device=DEVICE,
            log_path=RESULTS / "heads" / f"{name}_history.csv")
        save_head(head, name)
        summary[name] = info
        print(f"{name:<12} best val acc {info['best_val_acc']:.4f} at epoch {info['best_epoch']} "
              f"(ran {info['epochs_run']} epochs)")
    save_json(summary, RESULTS / "heads" / "summary.json")
    save_run_metadata(RESULTS / "heads" / "run_meta.json", config=c)


def stage_clean():
    m = evaluate_condition("test_clean", device=DEVICE)
    df = pd.DataFrame([{"model": k, "accuracy": m[k]["accuracy"], "macro_f1": m[k]["macro_f1"],
                        "mean_max_conf": m[k]["mean_max_confidence"]} for k in MODELS])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "task1_clean_baseline.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["subset", "features", "heads", "clean", "all"])
    ap.add_argument("--conditions", nargs="+", default=["train", "val", "test_clean"])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.stage in ("subset", "all"):
        make_subset()
    if a.stage in ("features", "all"):
        stage_features(a.conditions, a.overwrite)
    if a.stage in ("heads", "all"):
        stage_heads()
    if a.stage in ("clean", "all"):
        stage_clean()


if __name__ == "__main__":
    main()
