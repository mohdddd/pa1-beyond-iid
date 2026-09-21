"""Shared PACS protocol for Tasks 2 and 3 (manual §2 "Dataset, Models, and Experimental Setup").

* Target: sketch.  Sources: photo, art_painting, cartoon.
* Per source domain: stratified 80/20 train/val split, seed 6304, committed to
  ``shared/splits/pacs_sketch_seed6304.json``. The sketch list is stored without
  labels (IDs carry no class name).
* Preprocessing: resize to 256x256; train = random 224 crop + horizontal flip;
  val/test = 224 center crop; normalisation of ResNet18_Weights.IMAGENET1K_V1.
* Batches: one DataLoader driven by ``DomainBalancedBatchSampler``, which draws
  a fixed number of examples from each domain per step and cycles (reshuffles)
  every domain independently when it is exhausted. Each domain has its own
  generator and each item its own augmentation seed, so the source images and
  their crops/flips are identical for every method (Tasks 2 and 3). Source-only / Task 3:
  8+8+8. Adaptation (Task 2): 8+8+8 source + 24 target.
* Epoch definition (our choice; the manual fixes the per-step composition but
  not the epoch length): one "source epoch" = ceil(N_source_train / 24) steps,
  i.e. as many source images as the pooled source training set.

CLI (run from the repo root):
    python -m shared.pacs_protocol prepare   # download+verify, make/verify splits, figure
    python -m shared.pacs_protocol check     # loader throughput + batch composition
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights

from common.io import load_json, save_json
from common.seed import SEED, make_generator, seed_worker
from shared.pacs import (CLASSES, DOMAINS, HF_REPO, HF_REVISION, PARQUET_SHA256,
                         TARGET_DOMAIN, PACSDataset, PACSStore)

SOURCES = ["photo", "art_painting", "cartoon"]
TARGET = TARGET_DOMAIN
NUM_CLASSES = len(CLASSES)
VAL_FRACTION = 0.2
SPLIT_PATH = Path(__file__).resolve().parent / "splits" / "pacs_sketch_seed6304.json"

PER_SOURCE_BATCH = 8
TARGET_BATCH = 24

_W = ResNet18_Weights.IMAGENET1K_V1
MEAN, STD = list(_W.transforms().mean), list(_W.transforms().std)


# ---------------------------------------------------------------- transforms
def train_transform():
    return T.Compose([T.Resize((256, 256)), T.RandomCrop(224), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_transform():
    return T.Compose([T.Resize((256, 256)), T.CenterCrop(224),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


# ---------------------------------------------------------------- splits
def _build_splits() -> dict:
    store = PACSStore.get()
    out = {"meta": {"dataset": HF_REPO, "revision": HF_REVISION, "parquet_sha256": PARQUET_SHA256,
                    "seed": SEED, "val_fraction": VAL_FRACTION, "stratified_by": "class",
                    "sources": SOURCES, "target": TARGET, "classes": CLASSES,
                    "method": "sklearn.train_test_split on sorted IDs, per source domain"},
           "sources": {}, "target": {}}
    for d in SOURCES:
        ids = store.ids(d)
        y = [store.label[u] for u in ids]
        tr, va = train_test_split(ids, test_size=VAL_FRACTION, stratify=y,
                                  random_state=SEED, shuffle=True)
        out["sources"][d] = {"train": sorted(tr), "val": sorted(va)}
    out["target"] = {"domain": TARGET, "ids": store.ids(TARGET)}   # no labels stored
    return out


def make_splits(overwrite: bool = False) -> dict:
    new = _build_splits()
    if SPLIT_PATH.exists() and not overwrite:
        old = load_json(SPLIT_PATH)
        same = old["sources"] == new["sources"] and old["target"]["ids"] == new["target"]["ids"]
        if not same:
            raise RuntimeError(f"{SPLIT_PATH} differs from a fresh rebuild — investigate before continuing")
        print(f"[splits] existing split file verified identical: {SPLIT_PATH.name}")
        return old
    save_json(new, SPLIT_PATH)
    print(f"[splits] wrote {SPLIT_PATH}")
    return new


def load_splits() -> dict:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"{SPLIT_PATH} missing — run `python -m shared.pacs_protocol prepare`")
    return load_json(SPLIT_PATH)


def split_summary(splits: dict) -> list[dict]:
    """Per-domain, per-class counts (target: total only — no labels)."""
    store = PACSStore.get()
    rows = []
    for d in SOURCES:
        for part in ("train", "val"):
            ids = splits["sources"][d][part]
            counts = np.bincount([store.label[u] for u in ids], minlength=NUM_CLASSES)
            rows.append({"domain": d, "split": part, "n": len(ids),
                         **{c: int(k) for c, k in zip(CLASSES, counts)}})
    rows.append({"domain": TARGET, "split": "unlabeled_all", "n": len(splits["target"]["ids"]),
                 **{c: "" for c in CLASSES}})
    return rows


def steps_per_epoch(splits: dict) -> int:
    n = sum(len(splits["sources"][d]["train"]) for d in SOURCES)
    return math.ceil(n / (PER_SOURCE_BATCH * len(SOURCES)))


# ---------------------------------------------------------------- datasets
def source_dataset(splits, domain, part, train_aug=False):
    return PACSDataset(splits["sources"][domain][part],
                       train_transform() if train_aug else eval_transform(), labeled=True)


def target_unlabeled(splits, train_aug=True):
    """All sketch images WITHOUT labels (Task 2 adaptation / separability)."""
    return PACSDataset(splits["target"]["ids"],
                       train_transform() if train_aug else eval_transform(), labeled=False)


def target_labeled_for_final_evaluation(splits, task: str):
    """Sketch WITH labels. Only for final evaluation after all decisions are fixed (logged)."""
    return PACSDataset(splits["target"]["ids"], eval_transform(), labeled=True,
                       purpose="final_evaluation", task=task)


# ---------------------------------------------------------------- sampler
class DomainBalancedBatchSampler(Sampler):
    """Yields ``steps`` batches per iteration; each batch concatenates
    ``counts[k]`` items from group k, as ``(dataset_index, augmentation_seed)``.

    Controlled-comparison guarantees (manual: keep source sampling and
    augmentation fixed across methods):
    * every group has its OWN generator (seed, k), so a group's shuffle order
      never depends on whether other groups (e.g. the sketch target) are present;
    * every drawn item gets an augmentation seed derived only from
      (seed, group, pass, position), so its random crop/flip is identical across
      methods, worker counts and resumes.
    Groups are reshuffled independently when exhausted (loaders are cycled).
    State persists across epochs and is checkpointable."""

    def __init__(self, group_sizes: list[int], counts: list[int], steps: int, seed: int = SEED):
        self.sizes, self.counts, self.steps, self.seed = list(group_sizes), list(counts), steps, seed
        self.offsets = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.gens = [torch.Generator().manual_seed(seed * 100 + k) for k in range(len(self.sizes))]
        self.perms = [torch.randperm(n, generator=g) for n, g in zip(self.sizes, self.gens)]
        self.pos = [0] * len(self.sizes)
        self.passes = [0] * len(self.sizes)

    def _aug_seed(self, k: int) -> int:
        return int(np.random.SeedSequence([self.seed, k, self.passes[k], self.pos[k]]).generate_state(1)[0])

    def _take(self, k: int) -> list[tuple[int, int]]:
        out = []
        for _ in range(self.counts[k]):
            if self.pos[k] >= self.sizes[k]:
                self.passes[k] += 1
                self.perms[k] = torch.randperm(self.sizes[k], generator=self.gens[k])
                self.pos[k] = 0
            out.append((int(self.perms[k][self.pos[k]]) + self.offsets[k], self._aug_seed(k)))
            self.pos[k] += 1
        return out

    def __iter__(self):
        for _ in range(self.steps):
            yield [item for k in range(len(self.sizes)) for item in self._take(k)]

    def __len__(self):
        return self.steps

    def state_dict(self):
        return {"gens": [g.get_state() for g in self.gens], "perms": [p.clone() for p in self.perms],
                "pos": list(self.pos), "passes": list(self.passes)}

    def load_state_dict(self, s):
        for g, st in zip(self.gens, s["gens"]):
            g.set_state(st)
        self.perms = [p.clone() for p in s["perms"]]
        self.pos, self.passes = list(s["pos"]), list(s["passes"])


class SeededConcat(Dataset):
    """ConcatDataset whose items are requested as (index, seed): the transform
    runs under a forked CPU RNG seeded with ``seed`` (global RNG untouched)."""

    def __init__(self, parts):
        self.cat = ConcatDataset(parts)

    def __len__(self):
        return len(self.cat)

    def __getitem__(self, key):
        i, seed = key
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return self.cat[i]


def train_loader(splits, with_target: bool, num_workers: int = 2, seed: int = SEED,
                 steps: int | None = None):
    """Domain-balanced training loader.

    Batch layout (in order): 8 photo, 8 art_painting, 8 cartoon[, 24 unlabeled sketch].
    Items: (image, label, domain_idx, index); sketch labels are -1.
    Returns (loader, sampler) — keep the sampler to checkpoint its state.
    """
    parts = [source_dataset(splits, d, "train", train_aug=True) for d in SOURCES]
    counts = [PER_SOURCE_BATCH] * len(SOURCES)
    if with_target:
        parts.append(target_unlabeled(splits, train_aug=True))
        counts.append(TARGET_BATCH)
    sampler = DomainBalancedBatchSampler([len(p) for p in parts], counts,
                                         steps or steps_per_epoch(splits), seed=seed)
    loader = DataLoader(SeededConcat(parts), batch_sampler=sampler, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                        generator=make_generator(seed), persistent_workers=num_workers > 0)
    return loader, sampler


def eval_loader(dataset, batch_size: int = 128, num_workers: int = 2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def source_val_loaders(splits, **kw) -> dict:
    return {d: eval_loader(source_dataset(splits, d, "val"), **kw) for d in SOURCES}


# ---------------------------------------------------------------- CLI
def _sample_figure(splits, n_per_class: int = 1):
    """Grid: rows = source domains x columns = classes (from TRAIN splits),
    plus one row of random unlabeled sketch images (no class arrangement)."""
    import matplotlib.pyplot as plt
    from common.plotting import savefig, setup_style
    from shared.pacs import decode
    setup_style()
    store = PACSStore.get()
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(4, NUM_CLASSES, figsize=(1.5 * NUM_CLASSES, 1.6 * 4))
    for r, d in enumerate(SOURCES):
        tr = splits["sources"][d]["train"]
        for c, name in enumerate(CLASSES):
            ids = [u for u in tr if store.label[u] == c]
            ax = axes[r, c]
            ax.imshow(decode(store.bytes[ids[rng.integers(len(ids))]]))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=9)
            if c == 0:
                ax.set_ylabel(d.replace("_", " "), fontsize=9)
    tgt = splits["target"]["ids"]
    for c, i in enumerate(rng.choice(len(tgt), NUM_CLASSES, replace=False)):
        ax = axes[3, c]
        ax.imshow(decode(store.bytes[tgt[i]]))
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel("sketch\n(random)", fontsize=9)
    fig.tight_layout()
    return savefig(fig, "pacs_samples", "task2")


def cmd_prepare(args):
    PACSStore.get(keep_drive_copy=args.keep_drive_copy)
    splits = make_splits(overwrite=args.overwrite)
    rows = split_summary(splits)
    import csv
    from common.paths import REPO_ROOT
    out = Path(REPO_ROOT) / "task2" / "results" / "pacs_split_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[splits] counts -> {out}")
    for r in rows:
        print(f"  {r['domain']:>13s} {r['split']:>13s} n={r['n']:5d}  " +
              " ".join(f"{c[:4]}={r[c]!s:>3s}" for c in CLASSES))
    print(f"[protocol] steps per source epoch = {steps_per_epoch(splits)}")
    print(f"[figure] {_sample_figure(splits)}")


def cmd_check(args):
    splits = load_splits()
    for with_target in (False, True):
        loader, sampler = train_loader(splits, with_target, num_workers=args.workers, steps=args.steps)
        t0 = time.time()
        for step, (x, y, dom, _) in enumerate(loader):
            if step == 0:
                comp = {DOMAINS[k]: int((dom == k).sum()) for k in dom.unique().tolist()}
                print(f"[check] with_target={with_target} batch {tuple(x.shape)} composition {comp} "
                      f"target labels all -1: {bool((y[dom == DOMAINS.index(TARGET)] == -1).all())}")
        dt = time.time() - t0
        print(f"[check]   {args.steps} steps in {dt:.1f}s -> {dt / args.steps * 1000:.0f} ms/step "
              f"(epoch of {steps_per_epoch(splits)} steps ~ {dt / args.steps * steps_per_epoch(splits):.0f}s data time)")
    # 1) source stream identical with/without target (indices+aug seeds), past sketch reshuffle (~164 steps)
    n = [len(splits["sources"][d]["train"]) for d in SOURCES]
    a = DomainBalancedBatchSampler(n, [8, 8, 8], steps=400)
    b = DomainBalancedBatchSampler(n + [len(splits["target"]["ids"])], [8, 8, 8, 24], steps=400)
    same = all(x == y[:24] for x, y in zip(a, b))
    print(f"[check] source indices+augmentation seeds identical with/without target over 400 steps: {same}")
    # 2) identical source pixels with/without target, and for 0 vs 2 workers
    la, _ = train_loader(splits, False, num_workers=0, steps=2)
    lb, _ = train_loader(splits, True, num_workers=args.workers, steps=2)
    px = all(torch.equal(xa, xb[:24]) and torch.equal(ya, yb[:24])
             for (xa, ya, _, _), (xb, yb, _, _) in zip(la, lb))
    print(f"[check] source pixels identical (source-only, 0 workers) vs (with target, {args.workers} workers): {px}")
    # 3) resume reproducibility
    s1 = DomainBalancedBatchSampler([10, 7], [3, 2], steps=12)
    it = iter(s1); [next(it) for _ in range(4)]
    st = s1.state_dict(); rest = list(it)
    s2 = DomainBalancedBatchSampler([10, 7], [3, 2], steps=8); s2.load_state_dict(st)
    print(f"[check] sampler resume reproducible: {rest == list(s2)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--keep-drive-copy", action="store_true",
                   help="also keep the 191 MB parquet on Drive (off: re-download each session)")
    p.add_argument("--overwrite", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
