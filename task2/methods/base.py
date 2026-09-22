"""Common method interface.

A method receives, for one training step, the features f = F(x) and logits
C(f) of the WHOLE batch (sources first, then unlabeled target if present),
the labels (target labels are -1 and never used) and the domain indices.
It returns the total loss and a dict of scalar logs.

Methods may own extra trainable modules (e.g. a domain discriminator); they are
exposed via ``self.modules_`` so the trainer optimises and checkpoints them
with the same AdamW settings as the network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.pacs import DOMAINS, TARGET_DOMAIN

TARGET_IDX = DOMAINS.index(TARGET_DOMAIN)


class Method:
    name = "base"
    uses_target = False

    def __init__(self, cfg: dict, feat_dim: int, num_classes: int):
        self.cfg = cfg
        self.modules_ = nn.ModuleDict()      # extra trainable modules (may stay empty)
        al = cfg.get("align", {})
        # Stabilisation switches (see task2/README.md "Alignment-branch stabilisation").
        # l2_normalize: the feature COPY used by the alignment loss / discriminator is projected
        # onto the unit sphere; the classifier still sees the raw 512-d feature, so Source-only
        # ERM is bit-identical either way.
        self.l2_align = bool(al.get("l2_normalize", False))
        self.detach_mmd_median = bool(al.get("detach_mmd_median", True))

    def align_feats(self, feats):
        """Feature used by the alignment branch (MMD / domain discriminator)."""
        return F.normalize(feats, dim=1) if self.l2_align else feats

    @staticmethod
    def split(dom: torch.Tensor):
        """Boolean masks (source, target) over the batch."""
        tgt = dom == TARGET_IDX
        return ~tgt, tgt

    @staticmethod
    def source_ce(logits, y, src):
        ys = y[src]
        assert (ys >= 0).all(), "a source example has no label"
        return F.cross_entropy(logits[src].float(), ys)

    def loss(self, feats, logits, y, dom, progress: float):
        raise NotImplementedError
