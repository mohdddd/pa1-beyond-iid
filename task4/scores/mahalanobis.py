import numpy as np


class Mahalanobis:
    """u_Mah = min_c (f - mu_c)^T Sigma^{-1} (f - mu_c) with class means and ONE shared diagonal covariance
    (pooled within-class variance, ML estimate) from unaugmented CIFAR-10 training features, + 1e-6 on the diagonal."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def fit(self, feats, labels, num_classes: int = 10):
        f = np.asarray(feats, np.float64)
        self.mu = np.stack([f[labels == c].mean(0) for c in range(num_classes)])
        self.var = ((f - self.mu[labels]) ** 2).mean(0) + self.eps
        return self

    def __call__(self, feats):
        f = np.asarray(feats, np.float64)
        w = 1.0 / self.var
        d = (f ** 2 * w).sum(1, keepdims=True) - 2 * f @ (self.mu * w).T + (self.mu ** 2 * w).sum(1)[None]
        return np.maximum(d, 0).min(1)
