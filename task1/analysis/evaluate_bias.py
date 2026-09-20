"""Evaluation of the 4 predictors on any condition (clean, grayscale, ...).

Writes per-image predictions (results/predictions/<condition>.csv) and metrics
(results/metrics/<condition>.json). Later parts add prediction consistency
and shape-bias / coverage on top of these saved predictions.
"""
import numpy as np
import pandas as pd
import torch

from common.io import save_json
from common.metrics import classification_metrics
from task1.configs import BACKBONES, HEAD_OF, MODELS, RESULTS, split
from task1.models.backbones import load_features
from task1.models.heads import load_head, load_zeroshot


@torch.no_grad()
def probabilities(condition: str, device=None) -> dict:
    """model name -> softmax probabilities [N, C]."""
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
