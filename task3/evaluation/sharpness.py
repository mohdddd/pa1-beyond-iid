"""Common local sharpness proxy (manual §3 Step 4) — no Sketch.

Fixed validation batch: 32 images from each source validation split (seed 6304,
chosen once, identical for every model), eval transform. The model is in eval mode
(BN uses its frozen ImageNet statistics; no dropout in ResNet-18), fp32.

    Δ_sharp = L_val(θ + ε) − L_val(θ),   ε = 0.05 · ∇θ L_val / ||∇θ L_val||_2

L_val = mean cross-entropy over the 96 images; the gradient and ε cover ALL
trainable parameters (backbone incl. BN γ/β and the head) — the same scope as the
SAM perturbation. Parameters are restored exactly afterwards. One normalised
ascent step: a standardised local diagnostic, not a global flatness measure.
"""
import numpy as np
import torch
import torch.nn.functional as F

from common.seed import SEED
from shared.pacs_protocol import SOURCES, source_dataset

PER_SOURCE = 32
RADIUS = 0.05


def fixed_batch(splits, per_source: int = PER_SOURCE, seed: int = SEED):
    """(x, y, ids) — the same 96 validation images for every model."""
    rng = np.random.default_rng(seed)
    xs, ys, ids = [], [], []
    for d in SOURCES:
        ds = source_dataset(splits, d, "val")
        pick = np.sort(rng.choice(len(ds), per_source, replace=False))
        for i in pick:
            x, y, _, _ = ds[int(i)]
            xs.append(x); ys.append(y); ids.append(ds.ids[int(i)])
    return torch.stack(xs), torch.tensor(ys), ids


def sharpness(net, x, y, radius: float = RADIUS) -> dict:
    dev = next(net.parameters()).device
    x, y = x.to(dev), y.to(dev)
    net.eval()
    params = [p for p in net.parameters() if p.requires_grad]
    net.zero_grad(set_to_none=True)
    loss = F.cross_entropy(net(x)[1].float(), y)
    loss.backward()
    grads = [p.grad for p in params if p.grad is not None]
    gnorm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
    scale = float(radius / (gnorm + 1e-12))
    with torch.no_grad():
        saved = [p.detach().clone() for p in params]
        for p in params:
            if p.grad is not None:
                p.add_(p.grad, alpha=scale)
        loss_pert = F.cross_entropy(net(x)[1].float(), y)
        for p, s in zip(params, saved):
            p.copy_(s)
    net.zero_grad(set_to_none=True)
    return {"loss": loss.item(), "loss_perturbed": loss_pert.item(),
            "delta_sharp": loss_pert.item() - loss.item(), "grad_norm": gnorm.item(),
            "radius": radius, "n": int(len(y)), "n_params": int(sum(p.numel() for p in params))}
