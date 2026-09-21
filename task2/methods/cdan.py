"""CDAN — class-conditional adversarial alignment (manual Task 2 Step 4; Long et al., 2018).

Discriminator input g(x) = vec(f ⊗ p), f = 512-d feature, p = softmax(C(f)) (7-d),
i.e. a 3584-d multilinear map. Same discriminator width / dropout / GRL schedule /
unit loss weight as DANN. No entropy conditioning; neither f nor p is detached,
so the reversed gradient reaches the backbone through f and through p (and head).
"""
import torch

from task2.methods.dann import DANN


class CDAN(DANN):
    name = "cdan"

    def disc_in(self, feat_dim, num_classes):
        return feat_dim * num_classes

    def disc_input(self, feats, logits):
        p = logits.softmax(1)
        return torch.bmm(p.unsqueeze(2), feats.unsqueeze(1)).flatten(1)   # (B, C*D)
