"""CIFAR-10-only diagnostic for the PROSER validation drop (no CIFAR-100 access).

For each checkpoint: val accuracy (10 known logits) in eval mode with the stored BN running statistics, and after
re-estimating the BN running statistics on clean, unaugmented CIFAR-10 training images (cumulative average).
If re-estimation restores accuracy, the drop is caused by BN running statistics, not by the learned weights.
  python -m task4.evaluation.bn_diagnostic
"""
import copy
import os

os.environ["PA1_FORBID_UNKNOWN"] = "1"

import torch
import torch.nn as nn

from common.io import save_json
from common.paths import storage_dir, task_dir
from task4.data.cifar10 import load_split, to_eval_tensor
from task4.models.resnet_cifar import CifarResNet18

SPLITS = "task4/data/splits/cifar10_seed6304.json"


@torch.no_grad()
def acc(model, x, y, dev, bs=1000):
    model.eval()
    c = 0
    for i in range(0, len(y), bs):
        c += (model(x[i:i + bs].to(dev)).argmax(1).cpu() == y[i:i + bs]).sum().item()
    return c / len(y)


@torch.no_grad()
def reestimate_bn(model, x, dev, bs=500):
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            m.momentum = None                      # cumulative average over all batches
    model.train()
    for i in range(0, len(x), bs):
        model(x[i:i + bs].to(dev))
    model.eval()


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xtr, _, _ = load_split(SPLITS, "train")
    xva, yva, _ = load_split(SPLITS, "val")
    xtr, xva, yva = to_eval_tensor(xtr), to_eval_tensor(xva), torch.from_numpy(yva)
    ck = storage_dir("checkpoints", "task4")
    rows = []
    for run, f in [("vanilla", "best.pt"), ("proser", "best.pt"), ("proser", "last.pt"),
                   ("proser_cleanbn", "best.pt"), ("proser_cleanbn", "last.pt")]:
        p = ck / run / f
        if not p.exists():
            continue
        st = torch.load(p, map_location="cpu", weights_only=False)
        model = CifarResNet18(10, 5 if run.startswith("proser") else 0).to(dev)
        model.load_state_dict(st["model"])
        a0 = acc(model, xva, yva, dev)
        m2 = copy.deepcopy(model)
        reestimate_bn(m2, xtr, dev)
        a1 = acc(m2, xva, yva, dev)
        r = {"run": run, "file": f, "epoch": st["epoch"], "val_acc_stored_bn": a0, "val_acc_reestimated_bn": a1}
        rows.append(r)
        print(r, flush=True)
    save_json(rows, task_dir("task4", "results", "diagnostics") / "proser_bn_diagnostic.json")


if __name__ == "__main__":
    main()
