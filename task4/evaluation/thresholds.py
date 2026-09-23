import numpy as np

PERCENTILE = 95.0


def val_threshold(u_val):
    """tau = 95th percentile of unknownness on CIFAR-10 validation (known data only); accept x iff u(x) <= tau."""
    return float(np.percentile(u_val, PERCENTILE))


def accept_rate(u, tau):
    return float((np.asarray(u) <= tau).mean())
