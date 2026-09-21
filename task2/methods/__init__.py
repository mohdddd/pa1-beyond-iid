"""Method registry. Every method implements the same interface (see base.py)
so all methods run through the single training loop in task2/train.py."""
from task2.methods.cdan import CDAN
from task2.methods.dan import DAN
from task2.methods.dann import DANN
from task2.methods.source_only import SourceOnly

METHODS = {"source_only": SourceOnly, "dan": DAN, "dann": DANN, "cdan": CDAN}


def build_method(cfg: dict, feat_dim: int, num_classes: int):
    name = cfg["method"]["name"]
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; available: {sorted(METHODS)}")
    return METHODS[name](cfg, feat_dim, num_classes)
