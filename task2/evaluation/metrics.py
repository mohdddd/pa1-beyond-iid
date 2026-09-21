"""Prediction + per-domain metrics shared by training-time model selection
(source validation only) and final evaluation (Part 4).

Source-validation report: accuracy / macro-F1 for each source domain, then the
mean and worst over the three domains. Checkpoint selection uses
``mean_macro_f1`` (manual: mean macro-F1 across the three source val splits).
"""
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from common.metrics import classification_metrics
from shared.pacs import DOMAINS
from shared.pacs_protocol import NUM_CLASSES, SOURCES, source_dataset


@torch.no_grad()
def predict(net, loader, device, return_features: bool = False, amp: bool = True):
    """Returns dict of numpy arrays: pred, label, domain, prob[, feat]."""
    net.eval()
    out = {"pred": [], "label": [], "domain": [], "prob": [], "feat": []}
    for x, y, dom, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            f, logits = net(x)
        p = logits.float().softmax(1)
        out["pred"].append(p.argmax(1).cpu()); out["prob"].append(p.cpu())
        out["label"].append(y); out["domain"].append(dom)
        if return_features:
            out["feat"].append(f.float().cpu())
    res = {k: torch.cat(v).numpy() for k, v in out.items() if v}
    return res


def source_val_loader(splits, batch_size: int = 128, num_workers: int = 2):
    ds = ConcatDataset([source_dataset(splits, d, "val") for d in SOURCES])
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def domain_report(pred, label, domain, domains=SOURCES) -> dict:
    """Per-domain classification metrics + mean / worst over ``domains``."""
    rep = {}
    for d in domains:
        m = domain == DOMAINS.index(d)
        rep[d] = classification_metrics(label[m], pred[m], NUM_CLASSES)
    accs = [rep[d]["accuracy"] for d in domains]
    f1s = [rep[d]["macro_f1"] for d in domains]
    rep["mean_accuracy"] = float(np.mean(accs))
    rep["mean_macro_f1"] = float(np.mean(f1s))
    rep["worst_accuracy"] = float(np.min(accs))
    rep["worst_macro_f1"] = float(np.min(f1s))
    rep["worst_domain_by_f1"] = domains[int(np.argmin(f1s))]
    return rep


def source_val_report(net, loader, device, amp: bool = True) -> dict:
    r = predict(net, loader, device, amp=amp)
    return domain_report(r["pred"], r["label"], r["domain"])


def flat_row(rep: dict, prefix: str = "val_") -> dict:
    """Compact scalar row for CSV logs."""
    row = {}
    for d in SOURCES:
        row[f"{prefix}{d}_acc"] = rep[d]["accuracy"]
        row[f"{prefix}{d}_f1"] = rep[d]["macro_f1"]
    for k in ("mean_accuracy", "mean_macro_f1", "worst_accuracy", "worst_macro_f1"):
        row[prefix + k] = rep[k]
    return row
