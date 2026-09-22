"""Task 3 method registry. Trained methods (DAN-DG, SAM) run through the single
training loop in task2/train.py (same pipeline as ERM); ERM itself is reused."""
from task3.methods.dan_dg import DANDG
from task3.methods.erm import ERM
from task3.methods.sam import SAM

METHODS = {"dan_dg": DANDG, "sam": SAM}


def build_method(cfg: dict, feat_dim: int, num_classes: int):
    name = cfg["method"]["name"]
    if name == "erm":
        raise RuntimeError("ERM is not trained in Task 3 — it reuses the Task 2 source_only checkpoint")
    if name not in METHODS:
        raise KeyError(f"unknown Task 3 method {name!r}; available: {sorted(METHODS)}")
    return METHODS[name](cfg, feat_dim, num_classes)
