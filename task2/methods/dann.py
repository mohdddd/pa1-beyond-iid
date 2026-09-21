"""DANN — adversarial marginal alignment (manual Task 2 Step 3; Ganin et al., 2016).

Discriminator on the 512-d feature through a GRL with alpha(p).
L = L_cls(source only) + w * L_dom(source + target), w = 1.
Domain labels: 0 for every pooled source example, 1 for target (24 vs 24 per batch).
"""
import torch
import torch.nn.functional as F

from task2.methods.base import Method
from task2.models.domain_discriminator import DomainDiscriminator, grad_reverse, grl_alpha


class DANN(Method):
    name = "dann"
    uses_target = True

    def __init__(self, cfg, feat_dim, num_classes):
        super().__init__(cfg, feat_dim, num_classes)
        m = cfg["method"]
        self.grl_max = float(m["grl_max"])
        self.w = float(m["domain_loss_weight"])
        self.modules_["disc"] = DomainDiscriminator(self.disc_in(feat_dim, num_classes),
                                                    m["disc_hidden"], m["disc_dropout"])

    def disc_in(self, feat_dim, num_classes):
        return feat_dim

    def disc_input(self, feats, logits):
        return feats

    def loss(self, feats, logits, y, dom, progress):
        src, tgt = self.split(dom)
        cls = self.source_ce(logits, y, src)
        alpha = grl_alpha(progress, self.grl_max)
        d_logits = self.modules_["disc"](grad_reverse(self.disc_input(feats, logits), alpha))
        d_true = tgt.long()
        dl = F.cross_entropy(d_logits, d_true)
        d_acc = (d_logits.argmax(1) == d_true).float().mean()
        return cls + self.w * dl, {"cls_loss": cls.item(), "domain_loss": dl.item(),
                                   "disc_acc": d_acc.item(), "alpha": alpha}
