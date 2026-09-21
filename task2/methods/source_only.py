"""Source-only ERM (manual Task 2 Step 1): cross-entropy on the three labeled
source domains with domain-balanced 8+8+8 batches. Reused unchanged as the
Task 3 ERM baseline:  L_ERM = 1/3 * sum_e R_e(theta)  (equal per-domain counts
in every batch make the batch-mean CE equal to the domain-averaged risk)."""
from task2.methods.base import Method


class SourceOnly(Method):
    name = "source_only"
    uses_target = False

    def loss(self, feats, logits, y, dom, progress):
        src, tgt = self.split(dom)
        assert not tgt.any(), "source-only batch contains target examples"
        cls = self.source_ce(logits, y, src)
        return cls, {"cls_loss": cls.item()}
