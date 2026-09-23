"""PROSER — Learning Placeholders for Open-Set Recognition (Zhou et al., CVPR 2021), manual §4 step 4.

Written from the paper; the authors' code (github.com/zhoudw-zdw/CVPR21-Proser, no licence) was consulted only.
K = 10 known classes, 5 dummy classifiers. As in the paper, the dummies act as ONE extra class K+1 whose logit is
the maximum dummy response:  z_hat(x) = [W^T phi(x), max_k w_hat_k^T phi(x)]  (11 logits).

Each mini-batch is split in two equal halves (manual): first half -> classifier placeholders, second -> data
placeholders.
  classifier placeholders:  CE(z_hat(x), y) + beta * CE(z_hat(x) with logit y removed, K+1)
      the true class stays the top response, and once it is excluded the strongest dummy must win.
  data placeholders:  h~ = lam*phi_pre(x_i) + (1-lam)*phi_pre(x_j), y_i != y_j, lam ~ Beta(2,2), mixed after
      layer2 (before layer3); gamma * CE(z_hat(phi_post(h~)), K+1)  -- proxy unknowns go to the dummy class.
  total = mean_A[CE + beta*CE_masked] + gamma * mean_B[CE_mix]  with beta = 1, gamma = 0.1.
Differences of the reference code (not used): known-class CE over all 15 logits, mixup term weighted 0.01 and
targeted at the first dummy only, one lambda per batch, no different-class constraint, WideResNet, 10 epochs.
"""
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.io import sha256_file
from common.paths import storage_dir
from task4.methods.manifold_mixup import manifold_mixup
from task4.methods.vanilla import Vanilla

MASK = -1e9


@contextmanager
def no_running_stat_update(*modules):
    """BN layers normalise with batch statistics but leave running mean/var unchanged (momentum 0)."""
    bns = [m for mod in modules for m in mod.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, mo in zip(bns, saved):
            m.momentum = mo


def placeholder_logits(model, f):
    """11 logits: 10 known-class logits + the strongest dummy response (class K+1)."""
    return torch.cat([model.fc(f), model.dummy(f).max(1, keepdim=True).values], 1).float()


class PROSER(Vanilla):
    name = "proser"

    def setup(self, model, device):
        p = self.cfg["proser"]
        path = storage_dir("checkpoints", "task4") / p["init_run"] / "best.pt"
        st = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(st["model"], strict=False)
        assert not unexpected and set(missing) == {"dummy.weight", "dummy.bias"}, (missing, unexpected)
        self.meta = {"init_checkpoint": str(path), "init_sha256": sha256_file(path),
                     "init_epoch": st["epoch"], "init_val_acc": st["val_acc"]}
        self.beta, self.gamma, self.alpha = p["beta"], p["gamma"], p["mixup_alpha"]
        self.K = self.cfg["model"]["num_classes"]
        # "update" (default, as in the reference code): mixed features also update BN running stats of layer3/4.
        # "frozen": the mixed pass uses batch statistics but does not change the running statistics.
        self.mixed_bn = p.get("mixed_bn_stats", "update")
        assert self.mixed_bn in ("update", "frozen")
        return model

    def loss(self, model, x, y):
        n = len(y) // 2
        xa, ya, xb, yb = x[:n], y[:n], x[n:], y[n:]
        K = torch.full_like(ya, self.K)
        # classifier placeholders (first half)
        f = model.features(xa)
        zhat = placeholder_logits(model, f)
        l_known = F.cross_entropy(zhat, ya)
        masked = zhat.masked_fill(F.one_hot(ya, self.K + 1).bool(), MASK)
        l_cp = F.cross_entropy(masked, K)
        # data placeholders (second half): manifold mixup after layer2
        h_mix, _, _ = manifold_mixup(model.pre(xb), yb, self.alpha)
        if self.mixed_bn == "frozen":
            with no_running_stat_update(model.layer3, model.layer4):
                zmix = placeholder_logits(model, model.post(h_mix))
        else:
            zmix = placeholder_logits(model, model.post(h_mix))
        l_dp = F.cross_entropy(zmix, torch.full_like(yb, self.K))
        loss = l_known + self.beta * l_cp + self.gamma * l_dp
        with torch.no_grad():
            logs = {"l_known": l_known.detach(), "l_classifier_ph": l_cp.detach(), "l_data_ph": l_dp.detach(),
                    "mix_to_dummy": (zmix.argmax(1) == self.K).float().mean(),      # mixed examples assigned K+1
                    "known_dummy_2nd": (masked.argmax(1) == self.K).float().mean()}  # dummy strongest after y
        return loss, logs, zhat[:, : self.K], ya

    def known_logits(self, model, x):
        return model(x)              # CSA / checkpoint selection use only the 10 known-class logits
