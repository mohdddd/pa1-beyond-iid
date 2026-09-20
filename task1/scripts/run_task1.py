"""Single entry point for Task 1. Each stage reads the previous stage's saved
outputs, so stages can be run one at a time (or all at once) and re-run safely.

  --stage subset       splits + 500-image test subset + clean base images
  --stage features     frozen features for train/val/test_clean (+ CLIP zero-shot weights)
  --stage heads        linear head per backbone
  --stage clean        clean baseline table
  --stage color        grayscale + the additional colour transform
  --stage patch        4x4 patch shuffle
  --stage translation  0/8/16/32 px in 4 directions + curve and plot
  --stage summary      clean / grayscale / colour / patch comparison table
  --stage all          everything above, in order
(Later parts add: cue_conflict, representation.)
"""
import argparse
import time

import pandas as pd
import torch

from common.io import save_json, save_run_metadata
from common.paths import TABLE_DIR
from common.seed import set_seed
from task1.analysis.evaluate_bias import (cue_conflict_by_pair, evaluate_cue_conflict,
                                          evaluate_condition, plot_cue_examples,
                                          plot_translation, summarize_conditions,
                                          translation_curve)
from task1.configs import (BACKBONES, CUE_ACCEPTED, CUE_CANDIDATES, HEAD_OF, MODELS,
                           RESULTS, cfg, color_condition, patch_condition,
                           shift_condition, split)
from task1.data import transforms as T
from task1.data import make_cue_conflicts as CC
from task1.data.make_subset import (base_images, delete_images, images_exist, load_images,
                                    load_stl10, make_subset, save_grid, save_images)
from task1.models.backbones import (extract, has_features, load_backbone, load_features,
                                    save_features)
from task1.models.heads import has_zeroshot, save_head, save_zeroshot, train_linear_head

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Part 1 ----------------
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
            print(f"[{bb_name}] {cd:<22} {feats.shape}  {time.time() - t0:.1f}s")
        if need_text:
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


def _print(df):
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def stage_clean():
    m = evaluate_condition("test_clean", device=DEVICE)
    df = pd.DataFrame([{"model": k, "accuracy": m[k]["accuracy"], "macro_f1": m[k]["macro_f1"],
                        "mean_max_conf": m[k]["mean_max_confidence"]} for k in MODELS])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "task1_clean_baseline.csv", index=False)
    _print(df)


# ---------------- Part 2: controlled interventions ----------------
def _run_condition(condition, builder, overwrite=False, keep_images=True):
    """Build the images once -> features for all backbones -> evaluate."""
    if images_exist(condition) and not overwrite:
        print(f"[build] {condition}: already stored")
    else:
        imgs, meta = builder()
        save_images(condition, imgs, meta=meta)
        save_grid(imgs[::50][:10], f"samples_{condition}", nrow=10)
        print(f"[build] {condition}: {tuple(imgs.shape)}")
    stage_features([condition], overwrite)
    m = evaluate_condition(condition, device=DEVICE)
    if not keep_images:
        delete_images(condition)   # deterministic, rebuildable; saves Drive space
    return m


def stage_color(overwrite=False):
    c, s = cfg(), split()
    clean, labels = load_images("test_clean"), s["test_subset_labels"]
    _run_condition("test_gray",
                   lambda: (T.grayscale(clean),
                            {"desc": "grayscale: colour removed, luminance and geometry kept"}),
                   overwrite)
    name = color_condition(c)
    if c["color"]["additional"] == "hue_rotation":
        deg = c["color"]["hue_degrees"]
        def builder():
            return T.hue_rotation(clean, deg), {
                "desc": f"hue rotated by {deg} deg; saturation, value and geometry preserved",
                "degrees": deg}
    else:
        def builder():
            imgs, donors = T.palette_transfer(clean, labels, c["seed"])
            return imgs, {"desc": "per-channel mean/std matched to a donor image of another class",
                          "donor_rows": donors}
    _run_condition(name, builder, overwrite)
    df = summarize_conditions(["test_clean", "test_gray", name])
    df.to_csv(RESULTS / "metrics" / "color_summary.csv", index=False)
    _print(df)


def stage_patch(overwrite=False):
    c = cfg()
    clean = load_images("test_clean")
    g, name = c["patch_shuffle"]["grid"], patch_condition(c)

    def builder():
        imgs, perms = T.patch_shuffle(clean, g, c["seed"])
        return imgs, {"desc": f"{g}x{g} pixel-space patch permutation, non-identity, seed {c['seed']}",
                      "permutations": perms}
    _run_condition(name, builder, overwrite)
    _print(summarize_conditions(["test_clean", name]))


def stage_translation(overwrite=False, keep_images=False):
    c = cfg()
    clean = load_images("test_clean")
    tc = c["translation"]
    for d in [x for x in tc["shifts"] if x != 0]:
        for direction in tc["directions"]:
            keep = keep_images or (d == 32 and direction == "right")   # keep one for the figure
            _run_condition(shift_condition(d, direction),
                           lambda d=d, direction=direction: (
                               T.translate(clean, d, direction),
                               {"desc": f"reflection pad + shifted crop: {d}px {direction}"}),
                           overwrite, keep_images=keep)
    curve = translation_curve(tc["shifts"], tc["directions"])
    curve.to_csv(RESULTS / "metrics" / "translation_curve.csv", index=False)
    plot_translation(curve)
    print("\naccuracy vs displacement (mean over directions)")
    print(curve.pivot(index="shift", columns="model", values="accuracy")
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nprediction consistency vs displacement")
    print(curve.pivot(index="shift", columns="model", values="consistency")
          .to_string(float_format=lambda v: f"{v:.4f}"))


def stage_summary():
    """Required evidence: clean / grayscale / additional-colour / patch-shuffle comparison."""
    c = cfg()
    df = summarize_conditions(["test_clean", "test_gray", color_condition(c), patch_condition(c)])
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / "task1_interventions.csv", index=False)
    _print(df)


def stage_cue_generate():
    """Generate the AdaIN cue-conflict candidates (no model predictions involved)."""
    CC.generate_candidates(DEVICE)
    CC.contact_sheets()
    print("\nNext: review the sheets in the notebook, then record the rejections "
          "before running --stage cue_eval")


def stage_cue_eval():
    """Build the accepted condition, extract features, report shape bias + coverage."""
    CC.build_accepted()
    stage_features([CUE_ACCEPTED])
    df = evaluate_cue_conflict(DEVICE)
    df.to_csv(TABLE_DIR / "task1_shape_bias.csv", index=False)
    _print(df)
    for m in MODELS:
        by_pair = cue_conflict_by_pair(m)
        by_pair.insert(0, "model", m)
        by_pair.to_csv(RESULTS / "metrics" / f"cue_conflict_by_pair_{m}.csv", index=False)
    print("\nshape bias by pair (resnet50)")
    _print(cue_conflict_by_pair("resnet50"))
    plot_cue_examples()


STAGES = ["subset", "features", "heads", "clean", "color", "patch", "translation", "summary",
          "cue_generate", "cue_eval", "all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--conditions", nargs="+", default=["train", "val", "test_clean"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-images", action="store_true",
                    help="keep all 12 translation image files on Drive (~900 MB)")
    a = ap.parse_args()
    if a.stage in ("subset", "all"):
        make_subset()
    if a.stage in ("features", "all"):
        stage_features(a.conditions, a.overwrite)
    if a.stage in ("heads", "all"):
        stage_heads()
    if a.stage in ("clean", "all"):
        stage_clean()
    if a.stage in ("color", "all"):
        stage_color(a.overwrite)
    if a.stage in ("patch", "all"):
        stage_patch(a.overwrite)
    if a.stage in ("translation", "all"):
        stage_translation(a.overwrite, a.keep_images)
    if a.stage in ("summary", "all"):
        stage_summary()
    if a.stage == "cue_generate":
        stage_cue_generate()
    if a.stage == "cue_eval":
        stage_cue_eval()


if __name__ == "__main__":
    main()
