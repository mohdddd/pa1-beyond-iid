"""Manifold mixup between examples of DIFFERENT classes (PROSER data placeholders, manual §4 step 4)."""
import torch


def different_class_partners(y: torch.Tensor) -> torch.Tensor:
    """For each i, a partner index j drawn uniformly among examples with y_j != y_i (global torch RNG)."""
    allowed = (y[:, None] != y[None, :]).float()
    if (allowed.sum(1) == 0).any():
        raise ValueError("a mixup half-batch contains a single class; cannot form different-class pairs")
    return torch.multinomial(allowed, 1).squeeze(1)


def manifold_mixup(h: torch.Tensor, y: torch.Tensor, alpha: float = 2.0):
    """h~ = lam * h_i + (1 - lam) * h_j with y_i != y_j and lam ~ Beta(alpha, alpha) per pair.
    Returns (mixed representation, partner indices, lambdas)."""
    j = different_class_partners(y)
    lam = torch.distributions.Beta(alpha, alpha).sample((len(y),)).to(h.device)
    lam_ = lam.view(-1, *([1] * (h.dim() - 1))).to(h.dtype)
    return lam_ * h + (1 - lam_) * h[j], j, lam
