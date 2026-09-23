import numpy as np
from scipy.special import softmax


def msp(logits):
    """u_MSP = 1 - max_k softmax(z)_k"""
    return 1.0 - softmax(np.asarray(logits, np.float64), axis=1).max(1)
