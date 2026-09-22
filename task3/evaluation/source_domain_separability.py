"""Task 3 source-domain separability (manual §3 Step 4) — no Sketch.

Frozen 512-d features of the three source VALIDATION splits, balanced to the size
of the smallest one (photo, 334) by a seeded subsample that is identical for every
model; stratified 70/30 split (seed 6304); StandardScaler + multinomial logistic
regression (C = 1, balanced class weights; lbfgs, multinomial by default in
scikit-learn) predicting photo / art_painting / cartoon. Held-out accuracy is the
score (chance 1/3). Uses the Task 2 probe (``separability_score``) unchanged, so
Task 2 and Task 3 scores are computed identically apart from the groups.
"""
import numpy as np

from common.seed import SEED
from shared.pacs import DOMAINS
from shared.pacs_protocol import SOURCES
from task2.evaluation.domain_separability import equal_count_indices, separability_score


def balanced_groups(feat: np.ndarray, domain: np.ndarray, seed: int = SEED):
    """Per-source feature groups of equal size (seeded, independent of the model)."""
    idx = [np.flatnonzero(domain == DOMAINS.index(d)) for d in SOURCES]
    n = min(len(i) for i in idx)
    keep = [i[equal_count_indices(len(i), n, seed)] for i in idx]
    return [feat[k] for k in keep], keep


def source_separability(feat: np.ndarray, domain: np.ndarray, seed: int = SEED) -> dict:
    groups, keep = balanced_groups(feat, domain, seed)
    out = separability_score(groups, seed=seed)
    out["groups"] = list(SOURCES)
    return out
