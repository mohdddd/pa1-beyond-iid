"""Task 3 Step 1 — ERM baseline.

L_ERM = 1/3 * sum_{e in {P,A,C}} R_e(theta), realised by domain-balanced 8+8+8
batches (equal counts per domain make the batch-mean CE the domain-averaged risk).
This is exactly the Task 2 Source-only objective, and the Task 2 Source-only
checkpoint is reused unchanged: it is located and verified here, never retrained.
"""
from common.io import load_json, load_yaml
from common.paths import REPO_ROOT, storage_dir
from task2.methods.source_only import SourceOnly
from task2.train import sha256

ERM_CONFIG = REPO_ROOT / "task3" / "configs" / "erm.yaml"


class ERM(SourceOnly):
    name = "erm"


def erm_checkpoint():
    """Path of the reused checkpoint and its expected SHA-256 (from erm.yaml)."""
    reuse = load_yaml(ERM_CONFIG)["reuse"]
    path = storage_dir("checkpoints", reuse["task"]) / reuse["run"] / "best.pt"
    return path, reuse["sha256"]


def verify_erm_checkpoint() -> dict:
    """Refuse to continue unless the file is byte-identical to the Task 2 Source-only best.pt."""
    path, expected = erm_checkpoint()
    reuse = load_yaml(ERM_CONFIG)["reuse"]
    meta = load_json(REPO_ROOT / reuse["task"] / "results" / reuse["run"] / "run_meta.json")
    if meta["best_sha256"] != expected:
        raise RuntimeError("erm.yaml sha256 differs from task2 source_only run_meta.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — is PA1_STORAGE pointing to the Drive with Task 2 checkpoints?")
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f"ERM checkpoint sha256 {got[:12]} != expected {expected[:12]}")
    return {"path": str(path), "sha256": got, "task2_meta": meta}
