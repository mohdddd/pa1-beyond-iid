"""Central path configuration.

Large artefacts (datasets, checkpoints, feature caches) live OUTSIDE the repo,
under PA1_STORAGE (a Google Drive folder in Colab). Small results (JSON/CSV)
and figures live inside the repo so they are committed.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

STORAGE_ROOT = Path(os.environ.get("PA1_STORAGE", REPO_ROOT / "_storage"))
DATA_DIR = STORAGE_ROOT / "data"            # downloaded datasets (persistent, Drive)
CKPT_DIR = STORAGE_ROOT / "checkpoints"     # model checkpoints (persistent, Drive)
CACHE_DIR = STORAGE_ROOT / "cache"          # saved features / logits (persistent, Drive)

# Fast local disk for many-small-file datasets (e.g. PACS); wiped each session.
FAST_DATA_DIR = Path(os.environ.get("PA1_FAST_DATA", "/content/fast_data"))

FIG_DIR = REPO_ROOT / "report" / "figures"
TABLE_DIR = REPO_ROOT / "report" / "tables"


def task_dir(task: str, *sub: str) -> Path:
    """task_dir('task2', 'results') -> <repo>/task2/results (created)."""
    d = REPO_ROOT / task / Path(*sub) if sub else REPO_ROOT / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def storage_dir(kind: str, task: str) -> Path:
    """storage_dir('checkpoints', 'task2') -> <storage>/checkpoints/task2 (created)."""
    base = {"data": DATA_DIR, "checkpoints": CKPT_DIR, "cache": CACHE_DIR}[kind]
    d = base / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_dirs() -> None:
    for d in (DATA_DIR, CKPT_DIR, CACHE_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
