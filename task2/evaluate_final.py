"""Final Task 2 evaluation — the ONLY place Sketch labels are used (manual Task 2 Step 5 + Step 6).

    python -m task2.evaluate_final --confirm-final          # extract (GPU) + analyze + figures
    python -m task2.evaluate_final --stage analyze          # recompute tables/figures from cache (CPU)

Order of operations
  1. lock: every run must be finished; each best.pt must match the SHA-256 recorded
     in its committed run_meta.json; the lock (commit + hashes) is written BEFORE
     any Sketch label is loaded. Labelled Sketch access is logged by shared.pacs.
  2. extract: predictions + 512-d features on the 3 source-val splits and on all Sketch
     images, for every run -> $PA1_STORAGE/cache/task2/final_<run>.npz
  3. analyze: source-val / target metrics, target Δacc vs Source-only, domain separability
     (source-val vs an equal-size seeded Sketch subset, identical for all runs), per-class
     changes, dominant confusions, controlled study (λ=1 is the main DAN run).
  4. figures: training curves, study plot, per-class Δ, confusion matrices, flip examples.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from common.checkpoint import load_checkpoint
from common.io import git_commit_hash, load_json, save_json
from common.metrics import classification_metrics
from common.paths import FIG_DIR, REPO_ROOT, TABLE_DIR, storage_dir
from common.plotting import setup_style
from common.seed import SEED
from shared.pacs import CLASSES, DOMAINS, PACSStore, decode
from shared.pacs_protocol import (SOURCES, load_splits, source_dataset,
                                  target_labeled_for_final_evaluation)
from task2.evaluation import curves
from task2.evaluation.class_analysis import (confusion, flip_examples, largest_changes,
                                             per_class_accuracy, top_confusions)
from task2.evaluation.domain_separability import equal_count_indices, separability_score
from task2.evaluation.metrics import domain_report, predict
from task2.models.classifier_head import Net
from task2.train import sha256

MAIN = ["source_only", "dan", "dann", "cdan"]
STUDY = {0.1: "dan_lmmd0p1", 1.0: "dan", 10.0: "dan_lmmd10"}      # λ=1 reuses the main DAN run
RUNS = MAIN + [r for r in STUDY.values() if r not in MAIN]
BASE = "source_only"
TGT = DOMAINS.index("sketch")
CODE_PATHS = ["task2/configs", "task2/methods", "task2/models", "task2/train.py", "shared"]


class Out:
    def __init__(self, debug: bool):
        root = Path(tempfile.gettempdir()) / "pa1_debug_task2" if debug else None
        self.res = (root / "final") if debug else REPO_ROOT / "task2" / "results" / "final"
        self.tab = (root / "tables") if debug else TABLE_DIR
        self.fig = (root / "figures") if debug else FIG_DIR / "task2"
        self.cache = (root / "cache") if debug else storage_dir("cache", "task2")
        for p in (self.res, self.tab, self.fig, self.cache):
            p.mkdir(parents=True, exist_ok=True)

    def savefig(self, fig, name):
        fig.savefig(self.fig / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.fig / f"{name}.png", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------- 1. lock
def lock(out: Out, debug: bool, relock: bool = False) -> dict:
    ck_root = storage_dir("checkpoints", "task2")
    runs = {}
    for r in RUNS:
        meta = load_json(REPO_ROOT / "task2" / "results" / r / "run_meta.json")
        status = ck_root / r / "status.json"
        if not status.exists() or not load_json(status).get("finished"):
            raise RuntimeError(f"run {r} is not finished")
        h = sha256(ck_root / r / "best.pt")
        if h != meta["best_sha256"] and not debug:
            raise RuntimeError(f"{r}: best.pt sha256 {h[:12]} != run_meta {meta['best_sha256'][:12]}")
        runs[r] = {"sha256": h, "best_epoch": meta["best_epoch"], "config": meta["config"]}
    dirty = subprocess.run(["git", "status", "--porcelain", *CODE_PATHS], cwd=REPO_ROOT,
                           capture_output=True, text=True).stdout.strip()
    if dirty and not debug:
        raise RuntimeError(f"uncommitted changes in training code/configs — commit first:\n{dirty}")
    lk = {"git_commit": git_commit_hash(), "runs": {r: {"sha256": v["sha256"], "best_epoch": v["best_epoch"]}
                                                     for r, v in runs.items()},
          "note": "All Task 2 checkpoints/settings fixed before Sketch labels were loaded."}
    path = out.res / "lock.json"
    if path.exists() and not debug:
        old = load_json(path)
        if old["runs"] != lk["runs"]:
            if not relock:
                raise RuntimeError("lock.json exists with different checkpoints — rerun with --relock "
                                   "(the previous lock is archived) if the runs were legitimately retrained")
            save_json(old, out.res / f"lock_previous_{old['git_commit'][:10]}.json")
            save_json(lk, path)
    else:
        save_json(lk, path)
    return runs


# ---------------------------------------------------------------- 2. extract
def extract(out: Out, runs: dict, debug_limit: int | None):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits()
    src = ConcatDataset([source_dataset(splits, d, "val") for d in SOURCES])
    tgt = target_labeled_for_final_evaluation(splits, task="task2")
    tgt_pos = np.arange(len(tgt))                     # position of each evaluated image in the sketch ID list
    if debug_limit:
        src = Subset(src, list(range(0, len(src), max(1, len(src) // debug_limit))))
        tgt_pos = np.arange(0, len(tgt), max(1, len(tgt) // debug_limit))
        tgt = Subset(tgt, tgt_pos.tolist())
    mk = lambda ds: DataLoader(ds, batch_size=128, shuffle=False, num_workers=2,
                               pin_memory=dev.type == "cuda")
    ls, lt = mk(src), mk(tgt)
    ck_root = storage_dir("checkpoints", "task2")
    for r in runs:
        net = Net().to(dev)
        load_checkpoint(ck_root / r / "best.pt", net, map_location=dev, restore_rng=False)
        a, b = predict(net, ls, dev, True, amp=False), predict(net, lt, dev, True, amp=False)
        np.savez(out.cache / f"final_{r}.npz",
                 **{f"src_{k}": v for k, v in a.items()}, **{f"tgt_{k}": v for k, v in b.items()}, tgt_pos=tgt_pos)
        print(f"[extract] {r}: source-val acc {(a['pred'] == a['label']).mean():.4f} | "
              f"target acc {(b['pred'] == b['label']).mean():.4f}")


# ---------------------------------------------------------------- 3. analyze
def analyze(out: Out) -> dict:
    data = {r: dict(np.load(out.cache / f"final_{r}.npz")) for r in RUNS}
    base_t = data[BASE]
    n_src = len(data[BASE]["src_label"])
    t_idx = equal_count_indices(len(base_t["tgt_label"]), min(n_src, len(base_t["tgt_label"])), SEED)
    res = {}
    for r, d in data.items():
        rep = domain_report(d["src_pred"], d["src_label"], d["src_domain"])
        committed = load_json(REPO_ROOT / "task2" / "results" / r / "source_val.json")
        if abs(rep["mean_macro_f1"] - committed["mean_macro_f1"]) > 1e-3:
            print(f"[warn] {r}: source-val F1 {rep['mean_macro_f1']:.4f} differs from training-time "
                  f"{committed['mean_macro_f1']:.4f}")
        tp, tl = d["tgt_pred"], d["tgt_label"]
        cm = confusion(tp, tl)
        tm = classification_metrics(tl, tp, len(CLASSES))
        sep = separability_score([d["src_feat"], d["tgt_feat"][t_idx]], seed=SEED)
        res[r] = {"source_val": rep, "target": {"accuracy": tm["accuracy"], "macro_f1": tm["macro_f1"],
                                                "per_class_accuracy": per_class_accuracy(tp, tl).tolist(),
                                                "confusion_matrix": cm.tolist()},
                  "domain_separability": sep,
                  "feature_norm": {"source_val": float(np.linalg.norm(d["src_feat"], axis=1).mean()),
                                   "target": float(np.linalg.norm(d["tgt_feat"], axis=1).mean())},
                  "prediction_histogram_target": np.bincount(tp, minlength=len(CLASSES)).tolist()}
    b_acc = res[BASE]["target"]["accuracy"]
    b_pc = np.array(res[BASE]["target"]["per_class_accuracy"])
    for r in res:
        pc = np.array(res[r]["target"]["per_class_accuracy"])
        res[r]["target"]["delta_accuracy_vs_source_only"] = res[r]["target"]["accuracy"] - b_acc
        res[r]["target"]["per_class_delta_vs_source_only"] = (pc - b_pc).tolist()
        if r != BASE:
            res[r]["target"]["largest_changes"] = largest_changes(pc - b_pc)
    save_json({"separability_target_subset_size": int(len(t_idx)), "runs": res}, out.res / "metrics.json")
    write_tables(out, res)
    return res


def write_tables(out: Out, res: dict):
    meta = {r: load_json(REPO_ROOT / "task2" / "results" / r / "run_meta.json") for r in RUNS}
    rows = []
    for r in MAIN:
        s, t, sep = res[r]["source_val"], res[r]["target"], res[r]["domain_separability"]
        row = {"method": r, "best_epoch": meta[r]["best_epoch"], "stop_reason": meta[r]["stop_reason"]}
        for d in SOURCES:
            row[f"val_{d}_acc"] = s[d]["accuracy"]; row[f"val_{d}_f1"] = s[d]["macro_f1"]
        row.update(val_mean_acc=s["mean_accuracy"], val_mean_f1=s["mean_macro_f1"],
                   val_worst_acc=s["worst_accuracy"], val_worst_f1=s["worst_macro_f1"],
                   target_acc=t["accuracy"], target_f1=t["macro_f1"],
                   target_delta_acc=t["delta_accuracy_vs_source_only"],
                   source_target_gap_acc=s["mean_accuracy"] - t["accuracy"],
                   domain_separability=sep["accuracy"],
                   feat_norm_source=res[r]["feature_norm"]["source_val"],
                   feat_norm_target=res[r]["feature_norm"]["target"])
        rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(out.tab / "task2_main_comparison.csv", index=False)

    pc = []
    for c, name in enumerate(CLASSES):
        row = {"class": name, "n_target": int(np.array(res[BASE]["target"]["confusion_matrix"])[c].sum())}
        for r in MAIN:
            row[f"{r}_acc"] = res[r]["target"]["per_class_accuracy"][c]
            if r != BASE:
                row[f"{r}_delta"] = res[r]["target"]["per_class_delta_vs_source_only"][c]
        pc.append(row)
    pd.DataFrame(pc).round(4).to_csv(out.tab / "task2_per_class_target.csv", index=False)

    conf = []
    for r in MAIN:
        cm = np.array(res[r]["target"]["confusion_matrix"])
        for c, name in enumerate(CLASSES):
            tc = top_confusions(cm, c, 2)
            conf.append({"method": r, "true_class": name, "class_acc": res[r]["target"]["per_class_accuracy"][c],
                         **{f"conf{i + 1}": f"{k} ({n}, {f:.2f})" for i, (k, n, f) in enumerate(tc)}})
    pd.DataFrame(conf).round(4).to_csv(out.tab / "task2_top_confusions.csv", index=False)

    st = []
    for lam, r in STUDY.items():
        e = pd.read_csv(REPO_ROOT / "task2" / "results" / r / "train_epochs.csv")
        be = e[e["epoch"] == meta[r]["best_epoch"]].iloc[0]
        s, t = res[r]["source_val"], res[r]["target"]
        st.append({"lambda_mmd": lam, "run": r, "best_epoch": meta[r]["best_epoch"],
                   "train_mmd2_best_epoch": be.get("train_mmd", np.nan),
                   "val_mean_acc": s["mean_accuracy"], "val_mean_macro_f1": s["mean_macro_f1"],
                   "val_worst_macro_f1": s["worst_macro_f1"],
                   "target_accuracy": t["accuracy"], "target_macro_f1": t["macro_f1"],
                   "target_delta_acc_vs_source_only": t["delta_accuracy_vs_source_only"],
                   "domain_separability": res[r]["domain_separability"]["accuracy"]})
    pd.DataFrame(st).round(4).to_csv(out.tab / "task2_controlled_study.csv", index=False)
    for f in ("task2_main_comparison.csv", "task2_per_class_target.csv", "task2_controlled_study.csv"):
        print(f"\n== {f}\n" + pd.read_csv(out.tab / f).to_string(index=False))


# ---------------------------------------------------------------- 4. figures
def figures(out: Out, res: dict):
    setup_style()
    out.savefig(curves.training_curves(), "task2_training_curves")
    out.savefig(curves.study_curves(pd.read_csv(out.tab / "task2_controlled_study.csv")), "task2_controlled_study")

    # per-class Δ target accuracy vs Source-only
    fig, ax = plt.subplots(figsize=(7.5, 3))
    w, x = 0.27, np.arange(len(CLASSES))
    for i, r in enumerate(["dan", "dann", "cdan"]):
        ax.bar(x + (i - 1) * w, res[r]["target"]["per_class_delta_vs_source_only"], w,
               label=curves.LABEL[r], color=curves.COLOR[r])
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Δ target accuracy vs Source-only"); ax.legend()
    out.savefig(fig, "task2_per_class_delta")

    # row-normalised target confusion matrices
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.8))
    for a, r in zip(ax, MAIN):
        cm = np.array(res[r]["target"]["confusion_matrix"], dtype=float)
        cmn = cm / cm.sum(1, keepdims=True)
        a.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                       color="white" if cmn[i, j] > 0.5 else "black")
        a.set_xticks(range(len(CLASSES))); a.set_xticklabels(CLASSES, rotation=90, fontsize=7)
        a.set_yticks(range(len(CLASSES))); a.set_yticklabels(CLASSES, fontsize=7)
        a.set_title(f"{curves.LABEL[r]} (acc {res[r]['target']['accuracy']:.3f})")
        a.set_xlabel("predicted")
    ax[0].set_ylabel("true (Sketch)")
    out.savefig(fig, "task2_confusion_target")

    # flip examples: images the method broke (most-degraded class) / fixed (most-improved class)
    splits = load_splits()
    ids = splits["target"]["ids"]
    store = PACSStore.get()
    base = np.load(out.cache / f"final_{BASE}.npz")
    methods = ["dan", "dann", "cdan"]
    n = 4
    fig, ax = plt.subplots(len(methods), 2 * n, figsize=(2 * n * 1.35, len(methods) * 1.65))
    rows = []
    for i, r in enumerate(methods):
        d = np.load(out.cache / f"final_{r}.npz")
        lc = res[r]["target"]["largest_changes"]
        lost = flip_examples(base["tgt_pred"], d["tgt_pred"], d["tgt_label"],
                             CLASSES.index(lc["most_degraded"]), False, n, SEED)
        gained = flip_examples(base["tgt_pred"], d["tgt_pred"], d["tgt_label"],
                               CLASSES.index(lc["most_improved"]), True, n, SEED)
        for j in range(2 * n):
            a = ax[i, j]; a.set_xticks([]); a.set_yticks([])
            k = j if j < n else j - n
            sel = lost if j < n else gained
            if k < len(sel):
                t = int(sel[k])
                sid = ids[int(d["tgt_pos"][t])]
                a.imshow(decode(store.bytes[sid]))
                tl, bp, mp = CLASSES[d["tgt_label"][t]], CLASSES[base["tgt_pred"][t]], CLASSES[d["tgt_pred"][t]]
                a.set_title(f"{tl}\nSO:{bp} → {mp}", fontsize=6, color="#b2182b" if j < n else "#1b7837")
                rows.append({"method": r, "type": "lost" if j < n else "gained", "sketch_id": sid,
                             "true": tl, "source_only_pred": bp, "method_pred": mp})
            else:
                a.axis("off")
            if j == 0:
                a.set_ylabel(curves.LABEL[r], fontsize=8)
    fig.suptitle("left: correct under Source-only, wrong after adaptation (most-degraded class) | "
                 "right: fixed by adaptation (most-improved class)", fontsize=8)
    fig.tight_layout()
    out.savefig(fig, "task2_flip_examples")
    with open(out.res / "flip_examples.csv", "w", newline="") as f:
        if rows:
            w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
    print(f"[figures] -> {out.fig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "analyze", "figures"], default="all")
    ap.add_argument("--confirm-final", action="store_true",
                    help="required for the extract stage: confirms all Task 2 decisions are fixed")
    ap.add_argument("--relock", action="store_true",
                    help="checkpoints changed since the previous lock (archives it and writes a new one)")
    ap.add_argument("--debug-limit", type=int, default=None, help="testing only: tiny subsets, temp outputs")
    a = ap.parse_args()
    debug = a.debug_limit is not None
    out = Out(debug)
    if a.stage == "all":
        if not a.confirm_final:
            raise SystemExit("Refusing to load Sketch labels without --confirm-final.")
        runs = lock(out, debug, a.relock)
        extract(out, runs, a.debug_limit)
    if a.stage in ("all", "analyze"):
        res = analyze(out)
    else:
        res = load_json(out.res / "metrics.json")["runs"]
    figures(out, res)


if __name__ == "__main__":
    main()
