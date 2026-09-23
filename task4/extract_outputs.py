"""Lock all Task 4 checkpoints, then extract logits / features once (the saved cache is the only input to every score).

  python -m task4.extract_outputs --stage lock --proser-main proser_cleanbn   # before any CIFAR-100 access
  python -m task4.extract_outputs --stage extract                            # CIFAR-100 loaded here only

Cache: task4/cache/<run>.npz (git-ignored). Known parts: val, test (+ unaugmented train for Vanilla, used by the
Mahalanobis fit). Unknown part: the fixed 1600 CIFAR-100 test images. Eval mode, fp32.
"""
import argparse
import datetime
import os
import subprocess

import numpy as np
import torch

from common.io import git_commit_hash, load_json, save_json, sha256_file
from common.paths import REPO_ROOT, storage_dir, task_dir
from task4.data.cifar10 import load_split, to_eval_tensor
from task4.data.cifar100_unknowns import FAR, LOCK, NEAR, load_unknowns, verify_lock
from task4.models.resnet_cifar import CifarResNet18

RUNS = ("vanilla", "gcsc", "proser", "proser_cleanbn")
SPLITS = "task4/data/splits/cifar10_seed6304.json"
CACHE = REPO_ROOT / "task4/cache"
LIMIT = int(os.environ.get("PA1_T4_LIMIT", 0))          # sandbox testing only


def lock(proser_main):
    assert proser_main in ("proser", "proser_cleanbn")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "task4", "common"], cwd=REPO_ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty and not LIMIT:
        raise SystemExit(f"commit/push code first; uncommitted changes:\n{dirty}")
    ck = storage_dir("checkpoints", "task4")
    hashes = {}
    for r in RUNS:
        h = sha256_file(ck / r / "best.pt")
        rec = load_json(REPO_ROOT / "task4/results" / r / "best.json")["best_pt_sha256"]
        assert h == rec, f"{r}: checkpoint differs from the one recorded at training time"
        hashes[r] = h
    save_json({"time": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit_hash(),
               "checkpoints": hashes, "proser_main": proser_main,
               "scores": ["msp", "mls", "energy", "mahalanobis(shared diag cov + 1e-6, unaugmented train feats)",
                          "proser_placeholder(T=1024, p_dummy - max p_known)"],
               "threshold": "95th percentile of CIFAR-10 val unknownness; accept iff u <= tau",
               "unknowns": {"near": NEAR, "far": FAR}}, LOCK)
    print("locked:", load_json(LOCK))


@torch.no_grad()
def run_model(model, x, dev, bs=1000):
    zs, ds, fs = [], [], []
    for i in range(0, len(x), bs):
        f = model.features(x[i:i + bs].to(dev))
        zs.append(model.fc(f).cpu()); fs.append(f.cpu())
        if model.dummy is not None:
            ds.append(model.dummy(f).cpu())
    cat = lambda a: torch.cat(a).numpy() if a else None
    return cat(zs), cat(ds), cat(fs)


def extract():
    lk = verify_lock()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    known = {p: load_split(SPLITS, p) for p in ("train", "val", "test")}
    unk = load_unknowns("final_evaluation")
    parts = {p: (to_eval_tensor(x[:LIMIT or None]), y[:LIMIT or None]) for p, (x, y, _) in known.items()}
    parts["unknown"] = (to_eval_tensor(unk["images"]), None)
    CACHE.mkdir(exist_ok=True)
    for r in RUNS:
        st = torch.load(storage_dir("checkpoints", "task4") / r / "best.pt", map_location="cpu", weights_only=False)
        model = CifarResNet18(10, st["config"]["model"].get("num_dummy", 0)).to(dev).eval()
        model.load_state_dict(st["model"])
        out = {"epoch": st["epoch"], "unknown_group": unk["group"], "unknown_class": unk["fine_class"],
               "unknown_hf_index": unk["hf_index"]}
        for p, (x, y) in parts.items():
            if p == "train" and r != "vanilla":
                continue
            z, d, f = run_model(model, x, dev)
            out[f"{p}_logits"] = z
            if d is not None:
                out[f"{p}_dummy"] = d
            if r == "vanilla":
                out[f"{p}_feats"] = f
                if p == "unknown":
                    out["unknown_images"] = unk["images"]           # for the failure figure
            if y is not None:
                out[f"{p}_y"] = y
        np.savez(CACHE / f"{r}.npz", **out)
        print(f"{r}: epoch {st['epoch']}, test acc {(out['test_logits'].argmax(1) == out['test_y']).mean():.4f}",
              flush=True)
    save_json({"lock_sha256": sha256_file(LOCK), "time": datetime.datetime.now().isoformat(timespec="seconds"),
               "runs": list(RUNS), "proser_main": lk["proser_main"]},
              task_dir("task4", "results", "final") / "extract_meta.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["lock", "extract"], required=True)
    ap.add_argument("--proser-main", choices=["proser", "proser_cleanbn"])
    a = ap.parse_args()
    lock(a.proser_main) if a.stage == "lock" else extract()
