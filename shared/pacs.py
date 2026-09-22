"""PACS raw-data layer shared by Tasks 2 and 3.

Source: HuggingFace dataset ``flwrlabs/pacs`` pinned to one revision (a single
191 MB parquet file with all 9,991 images, 227x227 RGB). Counts match the
official PACS release (Li et al., 2017): photo 1670, art_painting 2048,
cartoon 2344, sketch 3929.

Design choices
--------------
* The parquet is downloaded to local disk (``FAST_DATA_DIR``), never to Drive,
  and verified by SHA-256. Images are kept as encoded bytes in RAM and decoded
  per item in DataLoader workers, so no extraction of thousands of small files.
* Every image gets a stable ID. Source IDs are ``<domain>/<class>/<filename>``
  (filenames repeat across classes in art_painting / cartoon). Sketch IDs are
  ``sketch/<filename>`` (unique within sketch) so that the committed split file
  carries no target-label information. Uniqueness is checked at load time.
* Target-leakage guards:
    - Target (sketch) datasets are built **without labels** unless the caller
      passes ``purpose="final_evaluation"``; every labelled access is appended
      to ``<task>/results/target_label_access.jsonl`` (audit trail).
    - If the environment variable ``PA1_FORBID_TARGET=1`` is set (Task 3
      training / selection / diagnostic scripts set it), *any* sketch access
      raises, and ``PACSStore`` drops every sketch row while reading the parquet,
      so no sketch image or label is ever held in memory by those processes.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from common.paths import FAST_DATA_DIR, STORAGE_ROOT

# ---------------------------------------------------------------- constants
HF_REPO = "flwrlabs/pacs"
HF_REVISION = "394113073258ead631f617d2e13bb377c0715c4b"
PARQUET_NAME = "train-00000-of-00001.parquet"
PARQUET_URL = (f"https://huggingface.co/datasets/{HF_REPO}/resolve/"
               f"{HF_REVISION}/data/{PARQUET_NAME}")
PARQUET_SHA256 = "4fc041ee92eec6043fe6e2859e8bdd138e5f958bc621afd153879812cbe65ff5"
PARQUET_BYTES = 191_395_900

CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
TARGET_DOMAIN = "sketch"
EXPECTED_COUNTS = {"photo": 1670, "art_painting": 2048, "cartoon": 2344, "sketch": 3929}

LOCAL_DIR = Path(FAST_DATA_DIR) / "pacs"
DRIVE_DIR = Path(STORAGE_ROOT) / "data" / "pacs"   # optional copy (off by default)


# ---------------------------------------------------------------- guards
class TargetAccessError(RuntimeError):
    pass


def assert_target_allowed(domain: str) -> None:
    if domain == TARGET_DOMAIN and os.environ.get("PA1_FORBID_TARGET") == "1":
        raise TargetAccessError(
            "Sketch access attempted while PA1_FORBID_TARGET=1 "
            "(Task 3 training / selection must never load the target domain).")


def log_target_label_access(task: str, purpose: str, note: str = "") -> None:
    from common.paths import REPO_ROOT
    root = Path(os.environ.get("PA1_TARGET_LOG_ROOT", REPO_ROOT))   # override only for sandbox tests
    path = root / task / "results" / "target_label_access.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "purpose": purpose, "note": note}
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- download
def _sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def ensure_parquet(keep_drive_copy: bool = False, verbose: bool = True) -> Path:
    """Return a verified local path to the PACS parquet, downloading if needed."""
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    local = LOCAL_DIR / PARQUET_NAME
    drive = DRIVE_DIR / PARQUET_NAME

    def ok(p: Path) -> bool:
        return p.exists() and p.stat().st_size == PARQUET_BYTES and _sha256(p) == PARQUET_SHA256

    if ok(local):
        if verbose:
            print(f"[pacs] local parquet verified: {local}")
    else:
        if drive.exists() and ok(drive):
            if verbose:
                print(f"[pacs] copying Drive copy -> {local}")
            import shutil
            shutil.copyfile(drive, local)
        else:
            if verbose:
                print(f"[pacs] downloading {PARQUET_URL}")
            tmp = local.with_suffix(".part")
            urllib.request.urlretrieve(PARQUET_URL, tmp)
            tmp.replace(local)
        if not ok(local):
            raise RuntimeError("PACS parquet failed SHA-256 verification")
        if verbose:
            print(f"[pacs] verified sha256 {PARQUET_SHA256[:12]}...")
    if keep_drive_copy and not drive.exists():
        import shutil
        DRIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local, drive)
        if verbose:
            print(f"[pacs] kept Drive copy: {drive}")
    return local


# ---------------------------------------------------------------- store
class PACSStore:
    """All PACS images as encoded bytes + metadata, keyed by stable ID."""

    _instance: "PACSStore | None" = None

    def __init__(self, parquet_path: Path):
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        self.target_excluded = os.environ.get("PA1_FORBID_TARGET") == "1"
        if self.target_excluded:          # Task 3: never materialise sketch images/labels
            table = table.filter(pc.not_equal(table.column("domain"), TARGET_DOMAIN))
        images = table.column("image").to_pylist()
        domains = table.column("domain").to_pylist()
        labels = table.column("label").to_pylist()
        self.bytes: dict[str, bytes] = {}
        self.label: dict[str, int] = {}
        self.domain: dict[str, str] = {}
        for im, d, y in zip(images, domains, labels):
            uid = (f"{d}/{im['path']}" if d == TARGET_DOMAIN
                   else f"{d}/{CLASSES[y]}/{im['path']}")
            if uid in self.bytes:
                raise RuntimeError(f"duplicate PACS id {uid}")
            self.bytes[uid] = im["bytes"]
            self.label[uid] = int(y)
            self.domain[uid] = d
        counts = {d: sum(1 for v in self.domain.values() if v == d) for d in DOMAINS}
        expected = dict(EXPECTED_COUNTS)
        if self.target_excluded:
            expected[TARGET_DOMAIN] = 0
        if counts != expected:
            raise RuntimeError(f"unexpected PACS domain counts {counts}")

    @classmethod
    def get(cls, keep_drive_copy: bool = False) -> "PACSStore":
        if cls._instance is None:
            cls._instance = cls(ensure_parquet(keep_drive_copy=keep_drive_copy))
        return cls._instance

    def ids(self, domain: str) -> list[str]:
        """Sorted IDs of one domain (sorted -> order independent of parquet)."""
        assert_target_allowed(domain)
        return sorted(u for u, d in self.domain.items() if d == domain)


def decode(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


# ---------------------------------------------------------------- dataset
class PACSDataset(Dataset):
    """Images of one or more domains.

    Returns ``(image, label, domain_idx, index)``. For unlabelled datasets the
    label is -1 and the true labels are never stored in the object.
    """

    def __init__(self, ids: list[str], transform=None, labeled: bool = True,
                 purpose: str = "", task: str = ""):
        store = PACSStore.get()
        for d in {u.split("/", 1)[0] for u in ids}:   # before lookup: sketch rows may be absent
            assert_target_allowed(d)
        domains = {store.domain[u] for u in ids}
        for d in domains:
            assert_target_allowed(d)
        if labeled and TARGET_DOMAIN in domains:
            if purpose != "final_evaluation":
                raise TargetAccessError(
                    "Labelled sketch data requested outside final evaluation "
                    f"(purpose={purpose!r}). Use labeled=False for adaptation.")
            log_target_label_access(task or "unknown", purpose, f"{len(ids)} sketch ids")
        self.ids = list(ids)
        self.transform = transform
        self.labeled = labeled
        self._bytes = [store.bytes[u] for u in self.ids]
        self.domain_idx = np.array([DOMAINS.index(store.domain[u]) for u in self.ids], dtype=np.int64)
        self.labels = (np.array([store.label[u] for u in self.ids], dtype=np.int64) if labeled
                       else np.full(len(self.ids), -1, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        img = decode(self._bytes[i])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), int(self.domain_idx[i]), i
