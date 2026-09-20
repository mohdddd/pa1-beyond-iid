"""Cue-conflict generation and manual review.

Generation (no model involved): for each unordered class pair and both
directions, AdaIN stylises a content image of one class with a style image of
the other, so shape and texture point to different labels.

Review: the rejection rule is written down BEFORE any model sees the images;
contact sheets are displayed in the notebook, the rejected rows are recorded,
and accepted/rejected counts are saved. Model predictions are never used.
"""
import numpy as np
import torch

from common.io import save_json
from common.paths import storage_dir
from common.seed import set_seed
from task1.configs import CUE_CANDIDATES, CUE_META, CUE_REVIEW, cfg, split
from task1.data.make_subset import load_images, save_grid, save_images


def generate_candidates(device="cuda"):
    """Builds all candidates and stores them + their metadata. Deterministic (seed 6304)."""
    from task1.models.adain import AdaINStyleTransfer
    c, s = cfg(), split()
    cc = c["cue_conflict"]
    classes, labels = s["classes"], np.asarray(s["test_subset_labels"])
    clean = load_images("test_clean")
    set_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    model = AdaINStyleTransfer(device)

    rows, out = [], []
    for a, b in cc["pairs"]:
        ia, ib = classes.index(a), classes.index(b)
        for content_cls, style_cls in ((ia, ib), (ib, ia)):
            pool_c = np.where(labels == content_cls)[0]
            pool_s = np.where(labels == style_cls)[0]
            n = min(cc["per_group"], len(pool_c), len(pool_s))
            c_rows = rng.choice(pool_c, size=n, replace=False)
            s_rows = rng.choice(pool_s, size=n, replace=False)
            cont = clean[c_rows].float().to(device) / 255.0
            styl = clean[s_rows].float().to(device) / 255.0
            img = model(cont, styl, alpha=cc["alpha"]).cpu()
            out.append((img * 255).round().to(torch.uint8))
            for k in range(n):
                rows.append({"candidate": len(rows),
                             "content_row": int(c_rows[k]), "style_row": int(s_rows[k]),
                             "content_label": int(content_cls), "style_label": int(style_cls),
                             "content_class": classes[content_cls], "style_class": classes[style_cls],
                             "pair": f"{a}-{b}", "direction": f"{classes[content_cls]}->{classes[style_cls]}"})
    images = torch.cat(out)
    save_images(CUE_CANDIDATES, images, meta={"desc": f"AdaIN cue conflicts, alpha={cc['alpha']}",
                                              "alpha": cc["alpha"], "n": len(images)})
    save_json({"alpha": cc["alpha"], "pairs": cc["pairs"], "seed": c["seed"], "rows": rows}, CUE_META)
    save_grid(images[:10], "samples_cue_candidates", nrow=10)
    print(f"generated {len(images)} candidates over {len(cc['pairs'])} pairs x 2 directions")
    return images


def contact_sheets(out_dir=None):
    """Writes numbered contact sheets for manual review; returns their paths."""
    import matplotlib.pyplot as plt
    c = cfg()
    k = c["cue_conflict"]["sheet_size"]
    images = load_images(CUE_CANDIDATES)
    out = out_dir or (storage_dir("cache", "task1") / "review")   # not committed
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, len(images), k):
        chunk = images[start:start + k]
        rows = int(np.ceil(len(chunk) / 5))
        fig, axes = plt.subplots(rows, 5, figsize=(11, 2.3 * rows))
        for ax in np.ravel(axes):
            ax.axis("off")
        for i, ax in enumerate(np.ravel(axes)[:len(chunk)]):
            ax.imshow(chunk[i].permute(1, 2, 0).numpy())
            ax.set_title(str(start + i), fontsize=9)
        fig.tight_layout()
        p = out / f"sheet_{start:03d}.png"
        fig.savefig(p, dpi=90, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    print(f"{len(paths)} sheets of {k} in {out}")
    return paths


def record_review(rule: str, rejected, auto_rejected=None) -> dict:
    """Saves the (pre-registered) rejection rule and the rejected candidate rows."""
    c = cfg()
    meta = _load_meta()
    n = len(meta["rows"])
    rejected = sorted({int(r) for r in rejected})
    assert all(0 <= r < n for r in rejected), "rejected index out of range"
    accepted = [i for i in range(n) if i not in set(rejected)]
    review = {"rule": rule, "generated": n, "rejected": rejected,
              "auto_rejected": sorted(auto_rejected or []),
              "n_rejected": len(rejected), "n_accepted": len(accepted),
              "accepted": accepted}
    save_json(review, CUE_REVIEW)
    print(f"generated {n} | rejected {len(rejected)} | accepted {len(accepted)} "
          f"(minimum required: {c['cue_conflict']['min_accepted']})")
    if len(accepted) < c["cue_conflict"]["min_accepted"]:
        print("WARNING: fewer accepted conflicts than the manual requires -> generate more")
    return review


def _load_meta() -> dict:
    from common.io import load_json
    return load_json(CUE_META)


def build_accepted():
    """Creates the evaluation condition from the accepted candidates only."""
    from common.io import load_json
    from task1.configs import CUE_ACCEPTED
    import pandas as pd
    review = load_json(CUE_REVIEW)
    meta = _load_meta()
    images = load_images(CUE_CANDIDATES)
    keep = review["accepted"]
    save_images(CUE_ACCEPTED, images[torch.as_tensor(keep)],
                meta={"desc": "accepted cue conflicts", "candidate_rows": keep,
                      "rule": review["rule"], "alpha": meta["alpha"]})
    save_grid(images[torch.as_tensor(keep[:10])], "samples_cue_accepted", nrow=10)
    df = pd.DataFrame([meta["rows"][i] for i in keep])
    print(f"accepted condition built: {len(keep)} images")
    print(df.groupby(["pair", "direction"]).size().to_string())
    return df
