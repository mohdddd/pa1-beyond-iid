"""Domain separability probe (manual Task 2 Step 5; reused by Task 3 with 3 source domains).

Frozen features of G groups (Task 2: source-val vs target; Task 3: photo/art/cartoon),
equal counts per group, stratified 70/30 split with seed 6304, logistic regression
C = 1 with balanced class weights. Held-out accuracy = separability score
(chance = 1/G). Features are standardised with statistics of the 70% training split
(our choice: needed for convergence when feature scales differ by orders of
magnitude across methods, e.g. the collapsed DANN).
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.seed import SEED


def equal_count_indices(n_total: int, n_keep: int, seed: int = SEED) -> np.ndarray:
    """Sorted random subset of size n_keep (identical for every method given the seed)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n_keep, replace=False))


def separability_score(groups: list, seed: int = SEED, C: float = 1.0, test_size: float = 0.3) -> dict:
    X = np.concatenate([np.asarray(g, dtype=np.float64) for g in groups])
    y = np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)])
    finite = bool(np.isfinite(X).all())
    if not finite:
        X = np.nan_to_num(X, nan=0.0, posinf=1e30, neginf=-1e30)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {"accuracy": float((pred == yte).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
            "chance": 1.0 / len(groups), "n_per_group": [int(len(g)) for g in groups],
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "converged": not any(issubclass(x.category, ConvergenceWarning) for x in w),
            "features_finite": finite}
