"""Cosine stability of each backbone's representation under an intervention.

    I_T = (1/N) sum_i  <f(x_i), f(T(x_i))> / (||f(x_i)|| ||f(T(x_i))||)

Each transformed image is paired with its own clean counterpart. For cue
conflicts the clean counterpart is the CONTENT image, i.e. the image whose
shape the conflict retains.
"""
import numpy as np

from common.io import load_json
from task1.configs import (BACKBONES, CUE_ACCEPTED, CUE_META, CUE_REVIEW, cfg,
                           shift_condition, split)
from task1.models.backbones import load_features

INTERVENTIONS = ["grayscale", "cue_conflict", "translation", "patch_shuffle"]


def cosine_stability(a: np.ndarray, b: np.ndarray) -> tuple:
    """Row-wise cosine similarity between paired feature matrices."""
    if a.shape != b.shape:
        raise ValueError(f"paired feature shapes differ {a.shape} vs {b.shape}; "
                         "re-run the stage that produced the transformed features")
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-12)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-12)
    cos = (a * b).sum(1)
    return float(cos.mean()), float(cos.std())


def paired_features(backbone: str, intervention: str, c=None):
    """Returns (clean_features, transformed_features, labels) for one intervention."""
    c = c or cfg()
    s = split()
    labels = np.asarray(s["test_subset_labels"])
    clean = load_features(backbone, "test_clean")
    if intervention == "grayscale":
        return clean, load_features(backbone, "test_gray"), labels
    if intervention == "patch_shuffle":
        from task1.configs import patch_condition
        return clean, load_features(backbone, patch_condition(c)), labels
    if intervention == "translation":
        d = c["representation"]["translation_delta"]
        feats = [load_features(backbone, shift_condition(d, dr)) for dr in c["translation"]["directions"]]
        return clean, feats, labels          # list: stability averaged over directions
    if intervention == "cue_conflict":
        meta, review = load_json(CUE_META), load_json(CUE_REVIEW)
        rows = [meta["rows"][i] for i in review["accepted"]]
        content_rows = np.array([r["content_row"] for r in rows])
        content_labels = np.array([r["content_label"] for r in rows])
        return clean[content_rows], load_features(backbone, CUE_ACCEPTED), content_labels
    raise ValueError(intervention)


def stability_table():
    """Cosine stability for every backbone x intervention."""
    import pandas as pd
    c = cfg()
    out = []
    for bb in BACKBONES:
        for iv in INTERVENTIONS:
            a, b, _ = paired_features(bb, iv, c)
            if isinstance(b, list):          # translation: mean over the four directions
                vals = [cosine_stability(a, bi) for bi in b]
                mean = float(np.mean([v[0] for v in vals]))
                sd = float(np.mean([v[1] for v in vals]))
            else:
                mean, sd = cosine_stability(a, b)
            out.append({"backbone": bb, "intervention": iv, "cosine_stability": mean, "std": sd})
    return pd.DataFrame(out)
