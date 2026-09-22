"""Final Task 3 evaluation — the ONLY Task 3 code that loads Sketch (manual §3 Step 4/5).

    python -m task3.evaluate_sketch --confirm-final     # lock -> extract (GPU) -> analyze -> figures
    python -m task3.evaluate_sketch --stage analyze     # recompute tables / figures from the cache (CPU)

Order of operations
  1. lock (before any Sketch access): every Task 3 run finished; each best.pt matches the SHA-256
     in its committed run_meta.json (ERM: the Task 2 source_only file named in erm.yaml);
     source-side diagnostics exist for every run; no uncommitted training / selection code. The lock records the commit, checkpoint hashes and
     the hash of the source-side results.
  2. extract: predictions + 512-d features on all Sketch images for every run
     -> $PA1_STORAGE/cache/task3/final_<run>.npz. Labelled access is logged to
     task3/results/target_label_access.jsonl.
  3. analyze: Sketch accuracy / macro-F1, Δacc vs ERM, per-class changes and dominant confusions,
     controlled study, comparison with Task 2 (DAN / CDAN, matched λ). Source-side numbers are read
     from task3/results/source_side/diagnostics.json (computed without Sketch).
     Extra (analysis only, after the lock): source-val vs Sketch separability with the Task 2 probe,
     and the Task 3 three-way source separability of the Task 2 models (from the Task 2 cache).
  4. figures: per-class Δ, confusion matrices, controlled study, source-vs-Sketch predictiveness,
     flip examples.
"""
from __future__ import annotations

import os

if os.environ.get("PA1_FORBID_TARGET") == "1":
    raise SystemExit("PA1_FORBID_TARGET=1 is set: the final Sketch evaluation cannot run in this process.")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from common.io import git_commit_hash, load_json, save_json  # noqa: E402
from common.metrics import classification_metrics  # noqa: E402
from common.paths import FIG_DIR, REPO_ROOT, TABLE_DIR, storage_dir  # noqa: E402
from common.plotting import setup_style  # noqa: E402
from common.seed import SEED  # noqa: E402
from shared.pacs import CLASSES, PACSStore, decode  # noqa: E402
from shared.pacs_protocol import SOURCES, load_splits, target_labeled_for_final_evaluation  # noqa: E402
from task2.evaluation.class_analysis import (confusion, flip_examples, largest_changes,  # noqa: E402
                                             per_class_accuracy, top_confusions)
from task2.evaluation.domain_separability import equal_count_indices, separability_score  # noqa: E402
from task3.evaluation.curves import COLOR, LABEL  # noqa: E402
from task3.evaluation.domain_metrics import source_row  # noqa: E402
from task3.evaluation.source_domain_separability import source_separability  # noqa: E402
from task3.selection.source_validation import (MAIN, RES, RUNS, STUDY, checkpoint, load_model,  # noqa: E402
                                               predict)

BASE = "erm"
CODE_PATHS = ["task3/configs", "task3/methods", "task3/models", "task3/selection", "task3/evaluation",
              "task3/train.py", "task3/source_diagnostics.py", "task2/train.py", "task2/methods",
              "task2/models", "shared"]
T2_MAP = {0.1: "dan_lmmd0p1", 1.0: "dan", 10.0: "dan_lmmd10"}      # Task 2 DAN runs at matched λ


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Out:
    def __init__(self, debug: bool):
        root = Path(tempfile.gettempdir()) / "pa1_debug_task3"
        self.debug = debug
        self.res = root / "final" if debug else RES / "final"
        self.side = root / "source_side" if debug else RES / "source_side"
        self.tab = root / "tables" if debug else TABLE_DIR
        self.fig = root / "figures" if debug else FIG_DIR / "task3"
        self.cache = root / "cache" if debug else storage_dir("cache", "task3")
        self.cache2 = storage_dir("cache", "task2")
        for p in (self.res, self.tab, self.fig, self.cache):
            p.mkdir(parents=True, exist_ok=True)

    def savefig(self, fig, name):
        fig.savefig(self.fig / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(self.fig / f"{name}.png", bbox_inches="tight", dpi=150)
        plt.close(fig)


def _debug_ckpt(r: str) -> Path:
    s = Path(tempfile.gettempdir()) / "pa1_smoke"
    src = {"erm": s / "task2" / "smoke_source_only", "dan_dg": s / "task3" / "smoke_dan_dg",
           "sam": s / "task3" / "smoke_sam"}
    return src.get(r, src["dan_dg"]) / "best.pt"          # study runs -> dan_dg smoke in debug


# ---------------------------------------------------------------- 1. lock
def lock(out: Out) -> dict:
    runs = {}
    for r in RUNS:
        if out.debug:
            runs[r] = {"path": _debug_ckpt(r), "sha256": "debug", "best_epoch": None}
            continue
        if r != BASE:
            st = storage_dir("checkpoints", "task3") / r / "status.json"
            if not st.exists() or not load_json(st).get("finished"):
                raise RuntimeError(f"run {r} is not finished")
        path, h = checkpoint(r, verify=True)
        runs[r] = {"path": path, "sha256": h, "best_epoch": load_json(RES / r / "run_meta.json")["best_epoch"]}
    side = out.side / "diagnostics.json"
    prereg = RES / "study_preregistration.md"
    if not out.debug:
        if not side.exists():
            raise RuntimeError("source-side diagnostics missing — run `python -m task3.source_diagnostics` first")
        missing = set(RUNS) - set(load_json(side)["runs"])
        if missing:
            raise RuntimeError(f"source-side diagnostics missing for {sorted(missing)}")
        for r in RUNS:
            if load_json(side)["runs"][r]["sha256"] != runs[r]["sha256"]:
                raise RuntimeError(f"{r}: source-side diagnostics were computed on a different checkpoint")
        dirty = subprocess.run(["git", "status", "--porcelain", *CODE_PATHS], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            raise RuntimeError(f"uncommitted changes in training/selection code — commit first:\n{dirty}")
    lk = {"git_commit": git_commit_hash(),
          "runs": {r: {"sha256": v["sha256"], "best_epoch": v["best_epoch"]} for r, v in runs.items()},
          "source_side_sha256": _sha(side) if side.exists() else None,
          "preregistration_sha256": _sha(prereg) if prereg.exists() else None,
          "note": "All Task 3 checkpoints, settings and source-side diagnostics fixed before Sketch was loaded."}
    path = out.res / "lock.json"
    if path.exists() and not out.debug:
        if load_json(path)["runs"] != lk["runs"]:
            raise RuntimeError("lock.json exists with different checkpoints — investigate, do not overwrite")
    else:
        save_json(lk, path)
    print(f"[lock] {len(runs)} runs locked at {lk['git_commit'][:10]}")
    return runs


# ---------------------------------------------------------------- 2. extract
def extract(out: Out, runs: dict, limit: int | None):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgt = target_labeled_for_final_evaluation(load_splits(), task="task3")
    pos = np.arange(len(tgt))
    if limit:
        pos = np.arange(0, len(tgt), max(1, len(tgt) // limit))
        tgt = Subset(tgt, pos.tolist())
    lt = DataLoader(tgt, batch_size=128, shuffle=False, num_workers=2, pin_memory=dev.type == "cuda")
    for r, v in runs.items():
        net = load_model(r, dev, verify=False, path=v["path"])
        b = predict(net, lt, dev, return_features=True, amp=False)
        np.savez(out.cache / f"final_{r}.npz", **{f"tgt_{k}": x for k, x in b.items()}, tgt_pos=pos)
        print(f"[extract] {r}: Sketch acc {(b['pred'] == b['label']).mean():.4f}")


# ---------------------------------------------------------------- 3. analyze
def analyze(out: Out) -> dict:
    side = load_json(out.side / "diagnostics.json")["runs"]
    src_cache = out.cache if out.debug else storage_dir("cache", "task3")
    res = {}
    for r in RUNS:
        d = dict(np.load(out.cache / f"final_{r}.npz"))
        tp, tl = d["tgt_pred"], d["tgt_label"]
        cm = confusion(tp, tl)
        tm = classification_metrics(tl, tp, len(CLASSES))
        extra = {}
        sc = src_cache / f"source_{r if r in side else BASE}.npz"
        if sc.exists():                                   # Task 2 probe on Task 3 models (analysis only)
            s = np.load(sc)
            t_idx = equal_count_indices(len(tl), min(len(s["feat"]), len(tl)), SEED)
            extra["source_vs_sketch_separability"] = separability_score([s["feat"], d["tgt_feat"][t_idx]],
                                                                        seed=SEED)["accuracy"]
            extra["feat_norm_sketch"] = float(np.linalg.norm(d["tgt_feat"], axis=1).mean())
        sd = side.get(r, side[BASE])
        res[r] = {"source_val": sd["source_val"], "separability": sd["separability"]["accuracy"],
                  "sharpness": sd["sharpness"]["delta_sharp"], "best_epoch": sd.get("best_epoch"),
                  "feat_norm_source": sd.get("feat_norm_mean"),
                  "target": {"accuracy": tm["accuracy"], "macro_f1": tm["macro_f1"],
                             "per_class_accuracy": per_class_accuracy(tp, tl).tolist(),
                             "confusion_matrix": cm.tolist(),
                             "prediction_histogram": np.bincount(tp, minlength=len(CLASSES)).tolist()},
                  **extra}
    b_acc, b_pc = res[BASE]["target"]["accuracy"], np.array(res[BASE]["target"]["per_class_accuracy"])
    for r in res:
        pc = np.array(res[r]["target"]["per_class_accuracy"])
        res[r]["target"]["delta_accuracy_vs_erm"] = res[r]["target"]["accuracy"] - b_acc
        res[r]["target"]["per_class_delta_vs_erm"] = (pc - b_pc).tolist()
        if r != BASE:
            res[r]["target"]["largest_changes"] = largest_changes(pc - b_pc)

    t2 = _task2(out, res)
    save_json({"runs": res, "task2_comparison": t2}, out.res / "metrics.json")
    write_tables(out, res, t2)
    return {"runs": res, "task2_comparison": t2}


def _task2(out: Out, res: dict) -> dict:
    """Task 2 numbers for the cross-task comparison (read-only)."""
    p = REPO_ROOT / "task2" / "results" / "final" / "metrics.json"
    if not p.exists():
        return {}
    m = load_json(p)["runs"]
    so = m["source_only"]["target"]
    same = np.allclose(so["per_class_accuracy"], res[BASE]["target"]["per_class_accuracy"]) and \
        abs(so["accuracy"] - res[BASE]["target"]["accuracy"]) < 1e-9
    t2 = {"erm_equals_task2_source_only_on_sketch": bool(same), "runs": {}}
    for r in ("source_only", "dan", "dan_lmmd0p1", "dan_lmmd10", "cdan"):
        if r not in m:
            continue
        row = {"sketch_accuracy": m[r]["target"]["accuracy"], "sketch_macro_f1": m[r]["target"]["macro_f1"],
               "per_class_accuracy": m[r]["target"]["per_class_accuracy"],
               "source_vs_sketch_separability": m[r]["domain_separability"]["accuracy"],
               "val_mean_macro_f1": m[r]["source_val"]["mean_macro_f1"]}
        c = out.cache2 / f"final_{r}.npz"
        if c.exists() and not out.debug:          # Task 3 three-way probe on the Task 2 model's source-val features
            z = np.load(c)
            row["three_way_source_separability"] = source_separability(z["src_feat"], z["src_domain"])["accuracy"]
        t2["runs"][r] = row
    print(f"[task2] ERM Sketch predictions identical to Task 2 source_only: {same}")
    return t2


def write_tables(out: Out, res: dict, t2: dict):
    rows = []
    for r in MAIN:
        t = res[r]["target"]
        rows.append({"method": r, "best_epoch": res[r]["best_epoch"], **source_row(res[r]["source_val"]),
                     "sketch_acc": t["accuracy"], "sketch_f1": t["macro_f1"],
                     "sketch_delta_acc_vs_erm": t["delta_accuracy_vs_erm"],
                     "src_domain_separability": res[r]["separability"], "sharpness_delta": res[r]["sharpness"],
                     "source_vs_sketch_separability": res[r].get("source_vs_sketch_separability")})
    pd.DataFrame(rows).round(4).to_csv(out.tab / "task3_main_comparison.csv", index=False)

    t2r = t2.get("runs", {})
    pc = []
    for c, name in enumerate(CLASSES):
        row = {"class": name, "n_sketch": int(np.array(res[BASE]["target"]["confusion_matrix"])[c].sum()),
               "erm_acc": res[BASE]["target"]["per_class_accuracy"][c]}
        for r in ("dan_dg", "sam"):
            row[f"{r}_acc"] = res[r]["target"]["per_class_accuracy"][c]
            row[f"{r}_delta"] = res[r]["target"]["per_class_delta_vs_erm"][c]
        for r in ("dan", "cdan"):
            if r in t2r:
                row[f"task2_{r}_delta"] = t2r[r]["per_class_accuracy"][c] - row["erm_acc"]
        pc.append(row)
    pd.DataFrame(pc).round(4).to_csv(out.tab / "task3_per_class_sketch.csv", index=False)

    conf = []
    for r in MAIN:
        cm = np.array(res[r]["target"]["confusion_matrix"])
        for c, name in enumerate(CLASSES):
            conf.append({"method": r, "true_class": name, "class_acc": res[r]["target"]["per_class_accuracy"][c],
                         **{f"conf{i + 1}": f"{k} ({n}, {f:.2f})"
                            for i, (k, n, f) in enumerate(top_confusions(cm, c, 2))}})
    pd.DataFrame(conf).round(4).to_csv(out.tab / "task3_top_confusions.csv", index=False)

    st = []
    for lam, r in STUDY.items():
        e = RES / r / "train_epochs.csv"
        mmd = np.nan
        if e.exists() and res[r]["best_epoch"]:
            ee = pd.read_csv(e)
            mmd = ee.loc[ee["epoch"] == res[r]["best_epoch"], "train_mmd"].iloc[0]
        s, t = res[r]["source_val"], res[r]["target"]
        row = {"lambda_dg": lam, "run": r, "best_epoch": res[r]["best_epoch"], "train_mmd2_best_epoch": mmd,
               "val_mean_macro_f1": s["mean_macro_f1"], "val_worst_macro_f1": s["worst_macro_f1"],
               "src_domain_separability": res[r]["separability"], "sharpness_delta": res[r]["sharpness"],
               "sketch_acc": t["accuracy"], "sketch_f1": t["macro_f1"], "sketch_delta_acc_vs_erm": t["delta_accuracy_vs_erm"],
               "source_vs_sketch_separability": res[r].get("source_vs_sketch_separability")}
        k = T2_MAP[lam]
        if k in t2r:
            row.update(task2_dan_sketch_acc=t2r[k]["sketch_accuracy"],
                       task2_dan_val_mean_macro_f1=t2r[k]["val_mean_macro_f1"],
                       task2_dan_source_vs_sketch_separability=t2r[k]["source_vs_sketch_separability"],
                       task2_dan_three_way_source_separability=t2r[k].get("three_way_source_separability"))
        st.append(row)
    pd.DataFrame(st).round(4).to_csv(out.tab / "task3_controlled_study.csv", index=False)
    for f in ("task3_main_comparison.csv", "task3_per_class_sketch.csv", "task3_controlled_study.csv"):
        print(f"\n== {f}\n" + pd.read_csv(out.tab / f).to_string(index=False))


# ---------------------------------------------------------------- 4. figures
def figures(out: Out, R: dict):
    setup_style()
    res, t2r = R["runs"], R["task2_comparison"].get("runs", {})
    x, w = np.arange(len(CLASSES)), 0.2
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = [("dan_dg", res["dan_dg"]["target"]["per_class_delta_vs_erm"], COLOR["dan_dg"], None),
            ("sam", res["sam"]["target"]["per_class_delta_vs_erm"], COLOR["sam"], None)]
    base_pc = np.array(res[BASE]["target"]["per_class_accuracy"])
    for r, lab in (("dan", "Task 2 DAN (λ=1, sees Sketch)"), ("cdan", "Task 2 CDAN")):
        if r in t2r:
            bars.append((lab, np.array(t2r[r]["per_class_accuracy"]) - base_pc, "#bbbbbb" if r == "dan" else "#2ca02c", "//"))
    off = (np.arange(len(bars)) - (len(bars) - 1) / 2) * w
    for (lab, v, c, h), o in zip(bars, off):
        ax.bar(x + o, v, w, label=LABEL.get(lab, lab), color=c, hatch=h, edgecolor="k" if h else None, lw=0.4)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_ylabel("Δ Sketch accuracy vs ERM"); ax.legend(fontsize=7, ncol=2)
    out.savefig(fig, "task3_per_class_delta")

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for a, r in zip(ax, MAIN):
        cm = np.array(res[r]["target"]["confusion_matrix"], dtype=float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                       color="white" if cmn[i, j] > 0.5 else "black")
        a.set_xticks(range(len(CLASSES))); a.set_xticklabels(CLASSES, rotation=90, fontsize=7)
        a.set_yticks(range(len(CLASSES))); a.set_yticklabels(CLASSES, fontsize=7)
        a.set_title(f"{LABEL[r]} (acc {res[r]['target']['accuracy']:.3f})"); a.set_xlabel("predicted")
    ax[0].set_ylabel("true (Sketch)")
    out.savefig(fig, "task3_confusion_sketch")

    lams = list(STUDY)
    g = lambda f: [f(res[STUDY[l]]) for l in lams]
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.1))
    a = ax[0]
    a.plot(lams, g(lambda z: z["source_val"]["mean_macro_f1"]), "o-", color=COLOR["dan_dg"], label="mean")
    a.plot(lams, g(lambda z: z["source_val"]["worst_macro_f1"]), "s--", color=COLOR["dan_dg"], label="worst")
    a.axhline(res[BASE]["source_val"]["mean_macro_f1"], color=COLOR["erm"], ls=":", label="ERM mean")
    a.set_title("(a) source-val macro-F1"); a.legend(fontsize=7)
    a = ax[1]
    a.plot(lams, g(lambda z: z["separability"]), "o-", color=COLOR["dan_dg"], label="DAN-DG")
    a.axhline(res[BASE]["separability"], color=COLOR["erm"], ls=":", label="ERM")
    a.axhline(1 / 3, color="k", ls="--", lw=0.8, label="chance")
    a.set_ylim(0, 1.02); a.set_title("(b) 3-way source-domain separability"); a.legend(fontsize=7)
    a = ax[2]
    a.plot(lams, g(lambda z: z["target"]["accuracy"]), "o-", color=COLOR["dan_dg"], label="Task 3 DAN-DG (no Sketch)")
    t2acc = [t2r[T2_MAP[l]]["sketch_accuracy"] if T2_MAP[l] in t2r else np.nan for l in lams]
    a.plot(lams, t2acc, "s--", color="#888888", label="Task 2 DAN (unlabelled Sketch)")
    a.axhline(res[BASE]["target"]["accuracy"], color=COLOR["erm"], ls=":", label="ERM")
    a.set_title("(c) Sketch accuracy"); a.legend(fontsize=7)
    for a in ax:
        a.set_xscale("log"); a.set_xticks(lams); a.set_xticklabels([str(l) for l in lams]); a.set_xlabel("λ")
    fig.tight_layout()
    out.savefig(fig, "task3_controlled_study")

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
    for a, key, t in zip(ax, ("mean_macro_f1", "worst_macro_f1"), ("mean", "worst")):
        for r in RUNS:
            a.scatter(res[r]["source_val"][key], res[r]["target"]["accuracy"], color=COLOR[r], label=LABEL[r], s=30)
        a.set_xlabel(f"{t} source-val macro-F1"); a.set_ylabel("Sketch accuracy")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    out.savefig(fig, "task3_source_vs_sketch")

    ids = load_splits()["target"]["ids"]
    store = PACSStore.get()
    base = np.load(out.cache / f"final_{BASE}.npz")
    n, methods, rows = 4, ["dan_dg", "sam"], []
    fig, ax = plt.subplots(len(methods), 2 * n, figsize=(2 * n * 1.35, len(methods) * 1.65))
    for i, r in enumerate(methods):
        d = np.load(out.cache / f"final_{r}.npz")
        lc = res[r]["target"]["largest_changes"]
        lost = flip_examples(base["tgt_pred"], d["tgt_pred"], d["tgt_label"], CLASSES.index(lc["most_degraded"]), False, n, SEED)
        gained = flip_examples(base["tgt_pred"], d["tgt_pred"], d["tgt_label"], CLASSES.index(lc["most_improved"]), True, n, SEED)
        for j in range(2 * n):
            a = ax[i, j]; a.set_xticks([]); a.set_yticks([])
            sel, k = (lost, j) if j < n else (gained, j - n)
            if k < len(sel):
                t = int(sel[k]); sid = ids[int(d["tgt_pos"][t])]
                a.imshow(decode(store.bytes[sid]))
                tl, bp, mp = CLASSES[d["tgt_label"][t]], CLASSES[base["tgt_pred"][t]], CLASSES[d["tgt_pred"][t]]
                a.set_title(f"{tl}\nERM:{bp} → {mp}", fontsize=6, color="#b2182b" if j < n else "#1b7837")
                rows.append({"method": r, "type": "lost" if j < n else "gained", "sketch_id": sid,
                             "true": tl, "erm_pred": bp, "method_pred": mp})
            else:
                a.axis("off")
            if j == 0:
                a.set_ylabel(LABEL[r], fontsize=8)
    fig.suptitle("left: correct under ERM, wrong after the method (most-degraded class) | "
                 "right: fixed by the method (most-improved class)", fontsize=8)
    fig.tight_layout()
    out.savefig(fig, "task3_flip_examples")
    with open(out.res / "flip_examples.csv", "w", newline="") as f:
        if rows:
            w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
    print(f"[figures] -> {out.fig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "analyze", "figures"], default="all")
    ap.add_argument("--confirm-final", action="store_true",
                    help="required for the extract stage: confirms all Task 3 decisions are fixed")
    ap.add_argument("--debug-limit", type=int, default=None, help="sandbox testing only")
    a = ap.parse_args()
    out = Out(a.debug_limit is not None)
    if a.stage == "all":
        if not a.confirm_final:
            raise SystemExit("Refusing to load Sketch without --confirm-final.")
        runs = lock(out)
        extract(out, runs, a.debug_limit)
    R = analyze(out) if a.stage in ("all", "analyze") else load_json(out.res / "metrics.json")
    figures(out, R)


if __name__ == "__main__":
    main()
