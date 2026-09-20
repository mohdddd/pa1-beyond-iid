"""JSON/YAML helpers and run metadata, so every reported number traces back
to a saved file, a config, and a git commit."""
import datetime
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from common.paths import REPO_ROOT


def _to_serializable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serializable: {type(o)}")


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_to_serializable)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=REPO_ROOT,
                                stderr=subprocess.DEVNULL) != 0
        return out + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_metadata(config: dict | None = None, **extra) -> dict:
    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit_hash(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config": config or {},
    }
    meta.update(extra)
    return meta


def save_run_metadata(path, config: dict | None = None, **extra) -> None:
    save_json(run_metadata(config, **extra), path)
