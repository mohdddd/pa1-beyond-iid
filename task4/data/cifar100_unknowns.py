"""Fixed CIFAR-100 test unknowns (manual §4). Evaluation-only: loads only for purpose="final_evaluation", only
after task4/results/final/lock.json exists and every locked checkpoint hash still matches, never when
PA1_FORBID_UNKNOWN=1 (set by training / diagnostics). Every access is appended to task4/results/unknown_access.jsonl."""
import datetime
import json
import os

import numpy as np

from common.io import git_commit_hash, load_json, sha256_file
from common.paths import REPO_ROOT, storage_dir
from task4.data.cifar10 import decode, fetch_parquet

NEAR = ("bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel")
FAR = ("bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe")
LOCK = REPO_ROOT / "task4/results/final/lock.json"
ACCESS_LOG = REPO_ROOT / "task4/results/unknown_access.jsonl"


class UnknownAccessError(RuntimeError):
    pass


def verify_lock():
    if not LOCK.exists():
        raise UnknownAccessError("no lock: run `python -m task4.extract_outputs --stage lock` first")
    lock = load_json(LOCK)
    for run, h in lock["checkpoints"].items():
        if sha256_file(storage_dir("checkpoints", "task4") / run / "best.pt") != h:
            raise UnknownAccessError(f"checkpoint {run} changed after the lock")
    return lock


def load_unknowns(purpose: str):
    if os.environ.get("PA1_FORBID_UNKNOWN") == "1":
        raise UnknownAccessError("CIFAR-100 access is forbidden in this process")
    if purpose != "final_evaluation":
        raise UnknownAccessError("CIFAR-100 may only be loaded for the final evaluation")
    verify_lock()
    t = fetch_parquet("cifar100_test")
    names = json.loads(t.schema.metadata[b"huggingface"])["info"]["features"]["fine_label"]["names"]
    fine = np.asarray(t.column("fine_label").to_pylist())
    rows, cls, grp = [], [], []
    for group, classes in (("near", NEAR), ("far", FAR)):
        for c in classes:
            idx = np.flatnonzero(fine == names.index(c))
            assert len(idx) == 100, (c, len(idx))
            rows += idx.tolist(); cls += [c] * 100; grp += [group] * 100
    with open(ACCESS_LOG, "a") as f:
        f.write(json.dumps({"time": datetime.datetime.now().isoformat(timespec="seconds"), "purpose": purpose,
                            "git_commit": git_commit_hash(), "lock_sha256": sha256_file(LOCK),
                            "n_images": len(rows)}) + "\n")
    return {"images": decode(t, rows), "fine_class": np.asarray(cls), "group": np.asarray(grp),
            "hf_index": np.asarray(rows)}
