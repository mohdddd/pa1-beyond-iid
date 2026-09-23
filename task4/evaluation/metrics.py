import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(u_known, u_unknown):
    """Unknown = positive class; larger unknownness = more novel."""
    y = np.r_[np.zeros(len(u_known)), np.ones(len(u_unknown))]
    return roc_auc_score(y, np.r_[u_known, u_unknown])


def roc(u_known, u_unknown):
    y = np.r_[np.zeros(len(u_known)), np.ones(len(u_unknown))]
    fpr, tpr, _ = roc_curve(y, np.r_[u_known, u_unknown])
    return fpr, tpr


def accuracy(logits, y):
    return float((np.asarray(logits).argmax(1) == y).mean())
