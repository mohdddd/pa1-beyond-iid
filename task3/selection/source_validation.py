"""Task 3 model selection — SOURCE validation only (no Sketch).

Checkpoints are selected by mean macro-F1 over the photo / art_painting / cartoon
validation splits (manual §3). During training the shared loop (task2/train.py)
calls ``source_val_report`` after every epoch and keeps the best checkpoint
(strict improvement, patience 5). This module is the Task 3 entry point for that
criterion and for locating / verifying the final checkpoint of every Task 3 run.
"""
from pathlib import Path

import torch

from common.checkpoint import load_checkpoint
from common.io import load_json
from common.paths import REPO_ROOT, storage_dir
from task2.evaluation.metrics import (domain_report, flat_row, predict,  # noqa: F401
                                      source_val_loader, source_val_report)
from task2.train import sha256
from task3.methods.erm import verify_erm_checkpoint
from task3.models.classifier_head import Net

SELECTION_METRIC = "mean_macro_f1"
MAIN = ["erm", "dan_dg", "sam"]
STUDY = {0.1: "dan_dg_l0p1", 1.0: "dan_dg", 10.0: "dan_dg_l10"}   # λ_DG = 1 is the main DAN-DG run
RUNS = MAIN + [r for r in STUDY.values() if r not in MAIN]
RES = REPO_ROOT / "task3" / "results"


def checkpoint(run: str, verify: bool = True) -> tuple[Path, str]:
    """(path, sha256) of a run's selected checkpoint; verified against the committed metadata."""
    if run == "erm":
        info = verify_erm_checkpoint() if verify else None
        from task3.methods.erm import erm_checkpoint
        path, exp = erm_checkpoint()
        return path, (info["sha256"] if info else exp)
    meta = load_json(RES / run / "run_meta.json")
    path = storage_dir("checkpoints", "task3") / run / "best.pt"
    if verify:
        got = sha256(path)
        if got != meta["best_sha256"]:
            raise RuntimeError(f"{run}: best.pt sha256 {got[:12]} != run_meta {meta['best_sha256'][:12]}")
    return path, meta["best_sha256"]


def available_runs() -> list[str]:
    return ["erm"] + [r for r in RUNS[1:] if (RES / r / "run_meta.json").exists()]


def load_model(run: str, device, verify: bool = True, path: Path | None = None) -> Net:
    path = path or checkpoint(run, verify)[0]
    net = Net().to(device)
    load_checkpoint(path, net, map_location=device, restore_rng=False)
    net.eval()
    return net
