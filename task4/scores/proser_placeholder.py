"""PROSER placeholder-based detection score, as in the authors' reference evaluation (valdummy, CONF_DeltaP):
softmax over [10 known logits, max dummy logit] at temperature 1024; u = p_dummy - max_k p_known_k.
The rejection threshold is calibrated on CIFAR-10 validation data (95th percentile), which plays the role of the
reference code's dummy-logit bias."""
import numpy as np
from scipy.special import softmax

TEMPERATURE = 1024.0


def proser_placeholder(logits, dummy_logits):
    z = np.concatenate([np.asarray(logits, np.float64), np.asarray(dummy_logits, np.float64).max(1, keepdims=True)], 1)
    p = softmax(z / TEMPERATURE, axis=1)
    return p[:, -1] - p[:, :-1].max(1)
