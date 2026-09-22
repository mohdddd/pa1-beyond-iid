"""Task 3 training-curve figure from the committed step/epoch logs (no target data).
ERM curves are the Task 2 source_only logs (same run). Only runs that exist are drawn."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common.paths import REPO_ROOT

LABEL = {"erm": "ERM", "dan_dg": "DAN-DG (λ=1)", "sam": "SAM (ρ=0.05)",
         "dan_dg_l0p1": "DAN-DG λ=0.1", "dan_dg_l10": "DAN-DG λ=10"}
COLOR = {"erm": "#555555", "dan_dg": "#1f77b4", "sam": "#ff7f0e",
         "dan_dg_l0p1": "#9ecae1", "dan_dg_l10": "#08306b"}


def _dir(run: str, res3: Path, res2: Path) -> Path:
    return res2 / "source_only" if run == "erm" else res3 / run


def _load(run, res3, res2, kind):
    p = _dir(run, res3, res2) / f"train_{kind}.csv"
    return pd.read_csv(p) if p.exists() else None


def _sm(s, w=5):
    return s.rolling(w, min_periods=1).mean()


def training_curves(runs, res3: Path = REPO_ROOT / "task3" / "results",
                    res2: Path = REPO_ROOT / "task2" / "results"):
    S = {r: _load(r, res3, res2, "steps") for r in runs}
    E = {r: _load(r, res3, res2, "epochs") for r in runs}
    S = {r: v for r, v in S.items() if v is not None}
    E = {r: v for r, v in E.items() if v is not None}
    main = [r for r in ("erm", "dan_dg", "sam") if r in S]
    study = [r for r in ("dan_dg_l0p1", "dan_dg", "dan_dg_l10") if r in S]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.4))
    a = ax[0, 0]
    for r in main:
        a.plot(S[r]["step"], _sm(S[r]["cls_loss"]), label=LABEL[r], color=COLOR[r], lw=1)
    a.set_yscale("log"); a.set_title("(a) source classification loss at θ"); a.set_xlabel("step"); a.legend()
    a = ax[0, 1]
    if "dan_dg" in S:
        s = S["dan_dg"]
        a.plot(s["step"], _sm(s["mmd"]), color=COLOR["dan_dg"], lw=1.4, label="mean")
        for k, ls in (("mmd_PA", ":"), ("mmd_PC", "--"), ("mmd_AC", "-.")):
            if k in s:
                a.plot(s["step"], _sm(s[k]), color="k", lw=0.7, ls=ls, label=k[4:])
        a.legend()
    a.set_title("(b) DAN-DG: pairwise MMD² (λ=1)"); a.set_xlabel("step")
    a = ax[0, 2]
    if "sam" in S:
        s = S["sam"]
        a.plot(s["step"], _sm(s["sam_gap"]), color=COLOR["sam"], lw=1, label="L(θ+ε) − L(θ)")
        a.set_xlabel("step"); a.set_title("(c) SAM: loss gap and ||∇L||")
        b = a.twinx(); b.plot(s["step"], _sm(s["grad_norm"]), color="k", lw=0.7, ls=":", label="||∇L||")
        b.set_ylabel("||∇L||", fontsize=8)
        h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2)
    a = ax[1, 0]
    for r in main:
        e = E.get(r)
        if e is None:
            continue
        a.plot(e["epoch"], e["val_mean_macro_f1"], marker="o", ms=3, color=COLOR[r], label=LABEL[r])
        b = e[e["best"] == 1].iloc[-1]
        a.plot(b["epoch"], b["val_mean_macro_f1"], marker="*", ms=11, color=COLOR[r], zorder=5)
    a.set_title("(d) mean source-val macro-F1 (★ = selected)"); a.set_xlabel("epoch"); a.legend()
    a = ax[1, 1]
    for r in study:
        a.plot(S[r]["step"], _sm(S[r]["mmd"]), color=COLOR[r], lw=1, label=LABEL[r])
    a.set_yscale("log"); a.set_title("(e) study: mean pairwise MMD²"); a.set_xlabel("step")
    if study:
        a.legend()
    a = ax[1, 2]
    for r in study:
        e = E.get(r)
        if e is not None:
            a.plot(e["epoch"], e["val_mean_macro_f1"], marker="o", ms=3, color=COLOR[r], label=LABEL[r])
    a.set_title("(f) study: mean source-val macro-F1"); a.set_xlabel("epoch")
    if study:
        a.legend()
    fig.tight_layout()
    return fig
