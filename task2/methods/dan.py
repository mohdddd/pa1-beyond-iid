"""DAN — multi-kernel MMD alignment (manual Task 2 Step 2; Long et al., 2015).

L = L_cls + lambda_mmd * MMD^2(F(x_s), F(x_t)) on the 512-d feature.
Kernel: sum of three RBF kernels k(a,b) = exp(-||a-b||^2 / (m * med)),
m in {0.5, 1, 2}, med = median pairwise squared distance in the current
combined batch (off-diagonal pairs, detached). Biased (V-statistic) estimate of
||E_s phi - E_t phi||^2_H via the kernel trick.
``mmd2`` is reused unchanged by Task 3 DAN-DG (same implementation required).
"""
import torch

from task2.methods.base import Method


def mmd2(x: torch.Tensor, y: torch.Tensor, mults=(0.5, 1.0, 2.0),
         detach_median: bool = True) -> torch.Tensor:
    z = torch.cat([x, y]).float()
    d2 = torch.cdist(z, z).pow(2)
    n = z.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=z.device)
    med = d2[off].median()
    med = (med.detach() if detach_median else med).clamp_min(1e-8)
    k = sum(torch.exp(-d2 / (m * med)) for m in mults)
    nx = x.shape[0]
    kxx, kyy, kxy = k[:nx, :nx], k[nx:, nx:], k[:nx, nx:]
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()


class DAN(Method):
    name = "dan"
    uses_target = True

    def __init__(self, cfg, feat_dim, num_classes):
        super().__init__(cfg, feat_dim, num_classes)
        self.lam = float(cfg["method"]["lambda_mmd"])
        self.mults = tuple(cfg["method"]["kernel_mults"])

    def loss(self, feats, logits, y, dom, progress):
        src, tgt = self.split(dom)
        cls = self.source_ce(logits, y, src)
        fa = self.align_feats(feats)
        mmd = mmd2(fa[src], fa[tgt], self.mults, detach_median=self.detach_mmd_median)
        return cls + self.lam * mmd, {"cls_loss": cls.item(), "mmd": mmd.item(),
                                      "feat_norm": feats.norm(dim=1).mean().item()}
