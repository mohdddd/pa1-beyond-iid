import numpy as np
from scipy.special import logsumexp


def energy(logits):
    """u_Energy = - log sum_k exp(z_k)"""
    return -logsumexp(np.asarray(logits, np.float64), axis=1)
