"""Shared classification metrics."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def classification_metrics(y_true, y_pred, num_classes: int) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1)
    per_class_acc = np.divide(np.diag(cm), support,
                              out=np.full(num_classes, np.nan), where=support > 0)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "per_class_accuracy": per_class_acc.tolist(),
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


def prediction_consistency(pred_a, pred_b) -> float:
    """Fraction of examples whose predicted class is unchanged between two conditions."""
    return float((np.asarray(pred_a) == np.asarray(pred_b)).mean())
