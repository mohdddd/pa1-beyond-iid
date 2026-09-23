import numpy as np


def mls(logits):
    """u_MLS = - max_k z_k"""
    return -np.asarray(logits, np.float64).max(1)
