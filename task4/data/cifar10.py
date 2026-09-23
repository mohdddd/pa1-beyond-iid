"""CIFAR-10 known-class data (manual §4).

Source: Hugging Face parquet mirrors uoft-cs/cifar10 and uoft-cs/cifar100 pinned to fixed commits and
SHA-256 verified (the Toronto server was ~100 KB/s). Images are lossless PNG copies of the official
archive (sampled images verified byte-identical). Row order = HF row order; all split indices refer to it.
Decoded uint8 arrays are cached on local disk (FAST_DATA_DIR/cifar).
"""
import io
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from common.io import load_json, sha256_file
from common.paths import FAST_DATA_DIR, REPO_ROOT

ROOT = FAST_DATA_DIR / "cifar"
MEAN = (0.4914, 0.4822, 0.4465)     # standard CIFAR-10 statistics
STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


HF_FILES = {   # name -> (repo, pinned commit, path, sha256)
    "cifar10_train": ("uoft-cs/cifar10", "0b2714987fa478483af9968de7c934580d0bb9a2",
                      "plain_text/train-00000-of-00001.parquet",
                      "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883"),
    "cifar10_test": ("uoft-cs/cifar10", "0b2714987fa478483af9968de7c934580d0bb9a2",
                     "plain_text/test-00000-of-00001.parquet",
                     "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de"),
    "cifar100_test": ("uoft-cs/cifar100", "aadb3af77e9048adbea6b47c21a81e47dd092ae5",
                      "cifar100/test-00000-of-00001.parquet",
                      "98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68"),
}


def fetch_parquet(name: str):
    repo, rev, path, sha = HF_FILES[name]
    out = ROOT / f"{name}.parquet"
    if not out.exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        urllib.request.urlretrieve(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}", tmp)
        if sha256_file(tmp) != sha:
            tmp.unlink()
            raise RuntimeError(f"SHA-256 mismatch for {name}")
        tmp.rename(out)
    return pq.read_table(out)


def decode(table, rows=None, img_col="img"):
    col = table.column(img_col).to_pylist() if rows is None else [table.column(img_col)[int(i)].as_py() for i in rows]
    return np.stack([np.array(Image.open(io.BytesIO(c["bytes"])).convert("RGB")) for c in col])


def load_cifar10(train: bool):
    """(N,32,32,3) uint8 images and int64 labels, HF row order."""
    name = "cifar10_train" if train else "cifar10_test"
    cache = ROOT / f"{name}.npz"
    if not cache.exists():
        t = fetch_parquet(name)
        np.savez(cache, x=decode(t), y=np.asarray(t.column("label").to_pylist(), dtype=np.int64))
    d = np.load(cache)
    return d["x"], d["y"]


def load_split(splits_path, part: str):
    """part in {train, val, test}; returns (images uint8, labels, indices into the official partition)."""
    if part == "test":
        x, y = load_cifar10(train=False)
        return x, y, np.arange(len(y))
    x, y = load_cifar10(train=True)
    val = np.asarray(load_json(REPO_ROOT / splits_path)["val_indices"])
    idx = val if part == "val" else np.setdiff1d(np.arange(len(y)), val)
    return x[idx], y[idx], idx


def train_transform(randaugment=None):
    ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if randaugment:
        ops.append(T.RandAugment(num_ops=randaugment["num_ops"], magnitude=randaugment["magnitude"]))
    return T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])


class ArrayDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), int(self.labels[i])


def to_eval_tensor(images) -> torch.Tensor:
    """Unaugmented images, identical to ToTensor + Normalize (val/test/feature extraction)."""
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div_(255)
    return (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)


if __name__ == "__main__":     # python -m task4.data.cifar10 : download/verify/cache CIFAR-10, check split fingerprint
    import hashlib
    x, y = load_cifar10(train=True)
    xt, yt = load_cifar10(train=False)
    s = load_json(REPO_ROOT / "task4/data/splits/cifar10_seed6304.json")
    tr = np.setdiff1d(np.arange(len(y)), np.asarray(s["val_indices"]))
    assert hashlib.sha256(tr.astype(np.int64).tobytes()).hexdigest() == s["train_sha256"], "split mismatch"
    print(f"CIFAR-10 ready: train {x.shape}, test {xt.shape}; split verified ({len(tr)}/{len(s['val_indices'])})")
