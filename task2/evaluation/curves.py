"""Training-curve figures from the committed step/epoch logs (no target data involved)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.paths import REPO_ROOT

RES = REPO_ROOT / "task2" / "results"
MAIN = ["source_only", "dan", "dann", "cdan"]
LABEL = {"source_only": "Source-only", "dan": "DAN", "dann": "DANN", "cdan": "CDAN",
         "dan_lmmd0p1": "DAN λ=0.1", "dan_lmmd10": "DAN λ=10"}
COLOR = {"source_only": "#555555", "dan": "#1f77b4", "dann": "#d62728", "cdan": "#2ca02c",
         "dan_lmmd0p1": "#9ecae1", "dan_lmmd10": "#08306b"}


def steps(run: str) -> pd.DataFrame:
    return pd.read_csv(RES / run / "train_steps.csv")


def epochs(run: str) -> pd.DataFrame:
    return pd.read_csv(RES / run / "train_epochs.csv")


def _smooth(s: pd.Series, w: int = 5) -> pd.Series:
    return s.rolling(w, min_periods=1).mean()


def _best_marker(ax, e: pd.DataFrame, color):
    b = e[e["best"] == 1].iloc[-1]
    ax.plot(b["epoch"], b["val_mean_macro_f1"], marker="*", ms=10, color=color, zorder=5)


def training_curves():
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.4))
    a = ax[0, 0]
    for r in MAIN:
        s = steps(r)
        a.plot(s["step"], _smooth(s["cls_loss"]), label=LABEL[r], color=COLOR[r], lw=1)
    a.set_yscale("log"); a.set_title("(a) source classification loss"); a.set_xlabel("step"); a.legend()
    a = ax[0, 1]
    s = steps("dan")
    a.plot(s["step"], _smooth(s["mmd"]), color=COLOR["dan"], lw=1)
    a.set_title("(b) DAN: MMD² (λ=1)"); a.set_xlabel("step")
    a = ax[0, 2]
    for r in ("dann", "cdan"):
        s = steps(r)
        a.plot(s["step"], _smooth(s["domain_loss"]), label=LABEL[r], color=COLOR[r], lw=1)
    a.set_yscale("log"); a.set_title("(c) domain-classification loss"); a.set_xlabel("step"); a.legend()
    a = ax[1, 0]
    for r in ("dann", "cdan"):
        s = steps(r)
        a.plot(s["step"], _smooth(s["disc_acc"]), label=LABEL[r], color=COLOR[r], lw=1)
    a.axhline(0.5, ls="--", color="k", lw=0.8)
    a.set_ylim(0, 1); a.set_title("(d) discriminator accuracy (chance 0.5)"); a.set_xlabel("step"); a.legend()
    a = ax[1, 1]
    for r in ("dann", "cdan"):
        s = steps(r)
        a.plot(s["step"], s["alpha"], label=LABEL[r], color=COLOR[r], lw=1.5 if r == "dann" else 1,
               ls="-" if r == "dann" else "--")
    a.set_title("(e) GRL coefficient α(p) (shared schedule)"); a.set_xlabel("step"); a.legend()
    a = ax[1, 2]
    for r in MAIN:
        e = epochs(r)
        a.plot(e["epoch"], e["val_mean_macro_f1"], marker="o", ms=3, label=LABEL[r], color=COLOR[r], lw=1)
        _best_marker(a, e, COLOR[r])
    a.set_ylim(0, 1); a.set_title("(f) mean source-val macro-F1 (★ selected)"); a.set_xlabel("epoch"); a.legend()
    fig.tight_layout()
    return fig


def study_curves(study_rows: pd.DataFrame, runs=("dan_lmmd0p1", "dan", "dan_lmmd10")):
    lab = {"dan_lmmd0p1": "λ=0.1", "dan": "λ=1", "dan_lmmd10": "λ=10"}
    col = {"dan_lmmd0p1": COLOR["dan_lmmd0p1"], "dan": COLOR["dan"], "dan_lmmd10": COLOR["dan_lmmd10"]}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    a = ax[0]
    x = study_rows["lambda_mmd"]
    for c, name in (("val_mean_macro_f1", "source-val mean macro-F1"), ("target_accuracy", "target accuracy"),
                    ("target_macro_f1", "target macro-F1"), ("domain_separability", "domain separability")):
        a.plot(x, study_rows[c], marker="o", label=name)
    a.axhline(0.5, ls=":", color="k", lw=0.8)
    a.set_xscale("log"); a.set_xticks(list(x)); a.set_xticklabels([str(v) for v in x])
    a.set_ylim(0, 1.02); a.set_xlabel("λ_MMD"); a.set_title("(a) alignment strength"); a.legend(fontsize=7)
    a = ax[1]
    for r in runs:
        s = steps(r)
        a.plot(s["step"], _smooth(s["mmd"]), label=lab[r], color=col[r], lw=1)
    a.set_yscale("log"); a.set_xlabel("step"); a.set_title("(b) MMD² during training"); a.legend()
    a = ax[2]
    for r in runs:
        e = epochs(r)
        a.plot(e["epoch"], e["val_mean_macro_f1"], marker="o", ms=3, label=lab[r], color=col[r], lw=1)
        _best_marker(a, e, col[r])
    a.set_ylim(0, 1); a.set_xlabel("epoch"); a.set_title("(c) mean source-val macro-F1"); a.legend()
    fig.tight_layout()
    return fig
