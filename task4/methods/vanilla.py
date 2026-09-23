"""Vanilla closed-set baseline: ten-class cross-entropy (manual §4 step 1)."""
import torch.nn.functional as F


class Vanilla:
    """Method interface used by task4/train.py:
    setup(model, device) -> model ; loss(model, x, y) -> (loss, {name: detached tensor}, known_logits, known_y);
    known_logits(model, x) -> 10 known-class logits (for CSA / checkpoint selection)."""
    name = "vanilla"

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self, model, device):
        return model

    def loss(self, model, x, y):
        logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        return loss, {"ce": loss.detach()}, logits, y

    def known_logits(self, model, x):
        return model(x)
