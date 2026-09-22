"""Task 3 Step 2 — DAN-DG: pairwise source-domain MMD alignment (no target).

L = L_ERM + lambda_dg / 3 * sum_{e<e'} MMD^2(F(X_e), F(X_e'))
on the 512-d feature before the head, over the pairs (photo, art), (photo, cartoon),
(art, cartoon) of every 8+8+8 batch. ``mmd2`` is the same Task 2 DAN implementation
(biased estimate, sum of RBF kernels with bandwidths {0.5, 1, 2} x median pairwise
squared distance). Called per pair, the median is taken over that pair's 16 features,
as the manual specifies.

The Task 2 alignment-branch stabilisation applies here unchanged (align.l2_normalize,
align.detach_mmd_median in task2/configs/base.yaml): the MMD sees the L2-normalised
feature copy and a differentiable median bandwidth, while the classifier keeps the raw
feature. Without it the penalty is minimised by changing feature scale and the model
collapses to a single predicted class (see task2/results/diagnostics/).
"""
from itertools import combinations

from shared.pacs import DOMAINS
from shared.pacs_protocol import SOURCES
from task2.methods.base import Method
from task2.methods.dan import mmd2

PAIRS = list(combinations(SOURCES, 2))
SHORT = {"photo": "P", "art_painting": "A", "cartoon": "C"}


class DANDG(Method):
    name = "dan_dg"
    uses_target = False

    def __init__(self, cfg, feat_dim, num_classes):
        super().__init__(cfg, feat_dim, num_classes)
        self.lam = float(cfg["method"]["lambda_dg"])
        self.mults = tuple(cfg["method"]["kernel_mults"])

    def loss(self, feats, logits, y, dom, progress):
        src, tgt = self.split(dom)
        assert not tgt.any(), "DAN-DG batch contains target examples"
        cls = self.source_ce(logits, y, src)
        fa = self.align_feats(feats)
        pair = {f"mmd_{SHORT[a]}{SHORT[b]}": mmd2(fa[dom == DOMAINS.index(a)],
                                                   fa[dom == DOMAINS.index(b)], self.mults,
                                                   detach_median=self.detach_mmd_median)
                for a, b in PAIRS}
        mmd = sum(pair.values()) / len(pair)
        logs = {"cls_loss": cls.item(), "mmd": mmd.item(), "feat_norm": feats.norm(dim=1).mean().item(),
                **{k: v.item() for k, v in pair.items()}}
        return cls + self.lam * mmd, logs
