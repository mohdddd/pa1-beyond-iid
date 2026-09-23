"""Stratified 90/10 train/val split of the official CIFAR-10 training partition, seed 6304.
Run: python -m task4.data.make_splits   (writes task4/data/splits/cifar10_seed6304.json)"""
import hashlib

import numpy as np

from common.io import save_json
from common.paths import REPO_ROOT
from common.seed import SEED
from task4.data.cifar10 import load_cifar10

OUT = REPO_ROOT / "task4/data/splits/cifar10_seed6304.json"


def main():
    _, y = load_cifar10(train=True)
    rng = np.random.default_rng(SEED)
    val = []
    for c in range(10):
        idx = np.flatnonzero(y == c)
        val.append(rng.permutation(idx)[: round(0.1 * len(idx))])
    val = np.sort(np.concatenate(val))
    train = np.setdiff1d(np.arange(len(y)), val)
    save_json({
        "seed": SEED, "rule": "per-class permutation (numpy default_rng, classes 0..9 in order), first 10% -> val",
        "n_train": len(train), "n_val": len(val),
        "val_per_class": np.bincount(y[val], minlength=10).tolist(),
        "train_sha256": hashlib.sha256(train.astype(np.int64).tobytes()).hexdigest(),
        "val_indices": val.tolist(),
    }, OUT)
    print(f"train {len(train)}  val {len(val)}  -> {OUT}")


if __name__ == "__main__":
    main()
