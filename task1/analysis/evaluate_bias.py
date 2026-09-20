"""Evaluation of the four predictors on any condition, and comparisons between
conditions (accuracy change, prediction consistency, translation curve).

Per-image predictions are saved for every condition, so any pair of conditions
can be compared later without re-running a model.
"""
import numpy as np
import pandas as pd
import torch

from common.io import load_json, save_json
from common.metrics import classification_metrics, prediction_consistency
from common.plotting import savefig, setup_style
from task1.configs import BACKBONES, HEAD_OF, MODELS, RESULTS, shift_condition, split
from task1.models.backbones import load_features
from task1.models.heads import load_head, load_zeroshot


# ---------------- per-condition evaluation ----------------
@torch.no_grad()
def probabilities(condition: str, device=None) -> dict:
    """model name -> softmax probabilities [N, C] from the cached features."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    probs = {}
    for bb in BACKBONES:
        X = torch.as_tensor(load_features(bb, condition), device=device)
        probs[HEAD_OF[bb]] = torch.softmax(load_head(HEAD_OF[bb], device)(X), 1).cpu().numpy()
        if bb == "clip_vitb32":   # zero-shot: softmax over scaled cosine similarities
            W, scale = load_zeroshot(device)
            probs["clip_zeroshot"] = torch.softmax(scale * X @ W.T, 1).cpu().numpy()
    return probs


def evaluate_condition(condition: str, labels=None, device=None) -> dict:
    """labels default to the ground truth of the 500-image test subset."""
    s = split()
    y = np.asarray(s["test_subset_labels"] if labels is None else labels)
    probs = probabilities(condition, device)
    rows, metrics = {"row": np.arange(len(y)), "label": y}, {}
    for m in MODELS:
        pred, conf = probs[m].argmax(1), probs[m].max(1)
        rows[f"{m}_pred"], rows[f"{m}_conf"] = pred, conf
        met = classification_metrics(y, pred, len(s["classes"]))
        met["mean_max_confidence"] = float(conf.mean())
        metrics[m] = met
    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "predictions" / f"{condition}.csv", index=False)
    save_json(metrics, RESULTS / "metrics" / f"{condition}.json")
    return metrics


def load_predictions(condition: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / "predictions" / f"{condition}.csv")


def load_metrics(condition: str) -> dict:
    return load_json(RESULTS / "metrics" / f"{condition}.json")


# ---------------- comparisons between conditions ----------------
def consistency_vs(condition: str, reference: str = "test_clean") -> dict:
    """Fraction of images whose predicted class is unchanged by the intervention."""
    if condition == reference:
        return {m: 1.0 for m in MODELS}
    ref, cur = load_predictions(reference), load_predictions(condition)
    return {m: prediction_consistency(ref[f"{m}_pred"], cur[f"{m}_pred"]) for m in MODELS}


def condition_rows(condition: str, reference: str = "test_clean", label: str | None = None) -> list:
    met, ref = load_metrics(condition), load_metrics(reference)
    cons = consistency_vs(condition, reference)
    return [{"condition": label or condition, "model": m,
             "accuracy": met[m]["accuracy"],
             "delta_accuracy": met[m]["accuracy"] - ref[m]["accuracy"],
             "macro_f1": met[m]["macro_f1"],
             "mean_max_conf": met[m]["mean_max_confidence"],
             "consistency": cons[m]} for m in MODELS]


def summarize_conditions(conditions, reference: str = "test_clean", labels: dict | None = None) -> pd.DataFrame:
    labels = labels or {}
    rows = []
    for cd in conditions:
        rows += condition_rows(cd, reference, labels.get(cd))
    return pd.DataFrame(rows)


def translation_curve(shifts, directions) -> pd.DataFrame:
    """Accuracy and prediction consistency vs displacement, averaged over the four
    cardinal directions (delta = 0 is the clean condition)."""
    rows = []
    for d in shifts:
        for direction in directions:
            cd = "test_clean" if d == 0 else shift_condition(d, direction)
            met, cons = load_metrics(cd), consistency_vs(cd)
            for m in MODELS:
                rows.append({"shift": d, "direction": direction, "model": m,
                             "accuracy": met[m]["accuracy"], "consistency": cons[m]})
    df = pd.DataFrame(rows)
    return df.groupby(["model", "shift"], as_index=False)[["accuracy", "consistency"]].mean()


def plot_translation(curve: pd.DataFrame, name: str = "task1_translation") -> None:
    import matplotlib.pyplot as plt
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharex=True)
    for m in MODELS:
        sub = curve[curve.model == m].sort_values("shift")
        axes[0].plot(sub["shift"], sub["accuracy"], marker="o", label=m)
        axes[1].plot(sub["shift"], sub["consistency"], marker="o", label=m)
    axes[0].set_xlabel("displacement (px)"); axes[0].set_ylabel("top-1 accuracy")
    axes[1].set_xlabel("displacement (px)"); axes[1].set_ylabel("prediction consistency")
    axes[1].legend(frameon=False)
    fig.suptitle("Translation (mean over 4 cardinal directions)", fontsize=10)
    savefig(fig, name, subdir="task1")
