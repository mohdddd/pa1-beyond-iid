"""Domain discriminator and gradient-reversal layer (DANN / CDAN).

Discriminator (manual): in_dim -> 256 -> ReLU -> Dropout(0.5) -> 2 logits
(0 = pooled source, 1 = target). GRL: identity forward, gradient * (-alpha) backward.
Schedule: alpha(p) = grl_max * (2 / (1 + exp(-10 p)) - 1), p = training progress in [0, 1]
(grl_max = 1 is the standard schedule; the controlled study may scale it).
"""
import math

import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None


def grad_reverse(x, alpha: float):
    return _GradReverse.apply(x, alpha)


def grl_alpha(p: float, grl_max: float = 1.0) -> float:
    return grl_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)
