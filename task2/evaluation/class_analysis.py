"""Per-class target analysis relative to Source-only (manual Task 2 Step 5, final stage only)."""
import numpy as np

from shared.pacs import CLASSES


def per_class_accuracy(pred, label, k: int = len(CLASSES)) -> np.ndarray:
    return np.array([(pred[label == c] == c).mean() if (label == c).any() else np.nan for c in range(k)])


def confusion(pred, label, k: int = len(CLASSES)) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (label, pred), 1)
    return cm


def top_confusions(cm: np.ndarray, cls: int, n: int = 2) -> list:
    """Most frequent wrong predictions for true class ``cls`` as (class, count, fraction)."""
    row = cm[cls].astype(float).copy()
    total = row.sum()
    row[cls] = -1
    order = [j for j in np.argsort(-row) if row[j] > 0][:n]
    return [(CLASSES[j], int(cm[cls, j]), float(cm[cls, j] / total)) for j in order]


def largest_changes(delta: np.ndarray) -> dict:
    return {"most_improved": CLASSES[int(np.nanargmax(delta))], "most_improved_delta": float(np.nanmax(delta)),
            "most_degraded": CLASSES[int(np.nanargmin(delta))], "most_degraded_delta": float(np.nanmin(delta))}


def flip_examples(pred_base, pred_run, label, cls: int, gained: bool, n: int, seed: int) -> np.ndarray:
    """Indices of class-``cls`` images that the method fixed (gained) or broke (not gained)
    relative to the baseline. Random but seeded selection."""
    ok_b, ok_r = pred_base == label, pred_run == label
    mask = (label == cls) & ((~ok_b & ok_r) if gained else (ok_b & ~ok_r))
    idx = np.flatnonzero(mask)
    if len(idx) > n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    return idx
