"""Unknown examples incorrectly accepted under a validation-calibrated threshold (manual §4 step 6)."""
import numpy as np
import pandas as pd

from task4.data.cifar10 import CLASSES


def accepted_unknowns(u_unk, pred, group, cls, hf_index, tau):
    df = pd.DataFrame({"group": group, "unknown_class": cls, "predicted": [CLASSES[p] for p in pred],
                       "score": u_unk, "threshold": tau, "hf_test_index": hf_index})
    return df[df.score <= tau].sort_values("score")


def most_confident_per_class(acc_df):
    """The most confidently accepted example (lowest unknownness) of every unknown class that has any."""
    return acc_df.groupby("unknown_class", sort=False).head(1).sort_values(["group", "score"],
                                                                           ascending=[False, True])


def absorption(acc_df, classes):
    """unknown class x predicted CIFAR-10 label counts among accepted unknowns."""
    t = pd.crosstab(acc_df.unknown_class, acc_df.predicted).reindex(index=classes, columns=CLASSES, fill_value=0)
    return t.fillna(0).astype(int)
