"""STL-10 loading, the common 224x224 base image, the fixed splits/subset, and
the image store that holds every evaluation condition.

* stratified 80/20 train/val split of the official train partition (seed 6304)
* class-balanced 500-image subset of the official test partition (seed 6304)
* identifiers saved to task1/data/stl10_split_seed6304.json (committed)
* each condition (test_clean, later grayscale, shuffled, ...) is saved ONCE as a
  uint8 [N,3,224,224] tensor so every model receives exactly the same images.
"""
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split

from common.io import save_json
from common.paths import DATA_DIR, FIG_DIR
from common.seed import set_seed
from task1.configs import SPLIT_PATH, cfg, image_dir

IMAGE_SIZE = 224


# ---------- dataset + common base image ----------
class STL10Split:
    """Minimal STL-10 reader for the official binary files (train or test).

    Reads stl10_binary/{split}_X.bin and {split}_y.bin directly. Unlike
    torchvision.datasets.STL10 it does not require the 2.7 GB unlabeled file
    nor re-hash every file on Drive at each load. Same layout as torchvision:
    .data uint8 [N,3,96,96], .labels int in 0..9, .classes list[str].
    """

    def __init__(self, split: str):
        root = DATA_DIR / "stl10_binary"
        xf, yf = root / f"{split}_X.bin", root / f"{split}_y.bin"
        missing = [str(f) for f in (xf, yf, root / "class_names.txt") if not f.exists()]
        if missing:
            raise FileNotFoundError(f"STL-10 files missing: {missing}. Is PA1_STORAGE set and "
                                    f"was STL-10 downloaded to {DATA_DIR}?")
        x = np.fromfile(xf, dtype=np.uint8)
        n = x.size // (3 * 96 * 96)
        if x.size != n * 3 * 96 * 96:
            raise RuntimeError(f"{xf} has an unexpected size (incomplete download?)")
        # binary is column-major per channel -> transpose H/W exactly like torchvision
        self.data = np.transpose(x.reshape(n, 3, 96, 96), (0, 1, 3, 2)).copy()
        self.labels = np.fromfile(yf, dtype=np.uint8).astype(np.int64) - 1   # stored as 1..10
        if len(self.labels) != n:
            raise RuntimeError(f"{yf} has {len(self.labels)} labels but {xf} has {n} images")
        self.classes = (root / "class_names.txt").read_text().split()


def load_stl10(split: str):
    """split in {'train', 'test'}; files downloaded to Drive by the setup notebook."""
    return STL10Split(split)


def raw_to_pil(raw_chw: np.ndarray) -> Image.Image:
    return Image.fromarray(np.transpose(raw_chw, (1, 2, 0)))   # STL-10 stores [3,96,96]


def to_base(img: Image.Image, size: int = IMAGE_SIZE) -> torch.Tensor:
    """PIL -> uint8 [3,224,224]: bicubic upsampling to the common base image."""
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).contiguous()


def base_images(dataset, indices) -> torch.Tensor:
    return torch.stack([to_base(raw_to_pil(dataset.data[i])) for i in indices])


# ---------- image store (one file per condition, on Drive) ----------
def image_path(condition: str):
    return image_dir() / f"{condition}.pt"


def save_images(condition: str, images: torch.Tensor, meta: dict | None = None) -> None:
    assert images.dtype == torch.uint8 and images.ndim == 4 and images.shape[1] == 3
    torch.save({"images": images.contiguous(), "meta": meta or {}}, image_path(condition))


def load_images(condition: str) -> torch.Tensor:
    return torch.load(image_path(condition), weights_only=False)["images"]


def images_exist(condition: str) -> bool:
    return image_path(condition).exists()


def delete_images(condition: str) -> None:
    """Remove a stored condition (used for the 12 translation conditions, which
    are deterministic and can be rebuilt from the clean images at any time)."""
    image_path(condition).unlink(missing_ok=True)


def save_grid(images: torch.Tensor, name: str, nrow: int = 8) -> None:
    """Preview grid -> report/figures/task1/<name>.png (for the report/appendix)."""
    out = FIG_DIR / "task1"
    out.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(images.float() / 255.0, out / f"{name}.png", nrow=nrow, padding=2)


# ---------- the stage ----------
def make_subset():
    c = cfg()
    seed = c["seed"]
    set_seed(seed)
    train, test = load_stl10("train"), load_stl10("test")
    classes = list(train.classes)
    y_train, y_test = np.asarray(train.labels), np.asarray(test.labels)

    tr_idx, va_idx = train_test_split(np.arange(len(y_train)), test_size=c["split"]["val_fraction"],
                                      stratify=y_train, random_state=seed)
    tr_idx, va_idx = np.sort(tr_idx), np.sort(va_idx)

    rng = np.random.default_rng(seed)
    per_class = c["test_subset"]["per_class"]
    subset, counts, notes = [], {}, []
    for k, name in enumerate(classes):
        pool = np.where(y_test == k)[0]
        take = min(per_class, len(pool))
        if take < per_class:
            notes.append(f"class {name} has only {len(pool)} test images")
        subset.extend(sorted(rng.choice(pool, size=take, replace=False).tolist()))
        counts[name] = int(take)
    subset = np.asarray(subset)

    save_json({
        "seed": seed, "classes": classes,
        "train_idx": tr_idx, "train_labels": y_train[tr_idx],
        "val_idx": va_idx, "val_labels": y_train[va_idx],
        "test_subset_idx": subset, "test_subset_labels": y_test[subset],
        "test_subset_class_counts": counts, "imbalance_notes": notes,
    }, SPLIT_PATH)
    print(f"train={len(tr_idx)} val={len(va_idx)} test_subset={len(subset)}")
    print("per-class test counts:", counts, "| notes:", notes or "none (balanced)")

    imgs = base_images(test, subset)
    save_images("test_clean", imgs, meta={"desc": "clean base images", "indices": subset.tolist()})
    save_grid(imgs[::50][:10], "samples_test_clean", nrow=10)
    print("saved condition 'test_clean':", tuple(imgs.shape))


if __name__ == "__main__":
    make_subset()
