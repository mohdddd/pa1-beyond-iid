"""Training-stability probe (diagnostic only; not part of the main comparison).

Runs the first N training steps of a method exactly as task2/train.py does
(same seed, init, batches, optimiser, frozen-BN policy) with or without fp16
autocast, printing losses and the peak |activation| of each ResNet stage every
25 steps. Used to diagnose the non-finite DANN/CDAN losses under fp16.

    PYTHONPATH=. python -m task2.evaluation.stability_probe dann fp32 300
    PYTHONPATH=. python -m task2.evaluation.stability_probe dann fp32 300 align.l2_normalize=true
"""
import math
import sys

import torch
import torch.nn as nn

from common.seed import set_seed
from shared.pacs_protocol import load_splits, train_loader
from task2.methods import build_method
from task2.models.backbone import freeze_bn_stats
from task2.models.classifier_head import Net
from task2.train import load_config


def main(name: str, amp: bool, steps: int, overrides=()):
    cfg = load_config(f"task2/configs/{name}.yaml", overrides)
    print(f"[probe] {name} amp={amp} steps={steps} overrides={list(overrides)}", flush=True)
    dev = torch.device("cuda")
    set_seed(6304)
    net = Net().to(dev)
    m = build_method(cfg, 512, 7)
    m.modules_.to(dev)
    c = nn.ModuleDict({"net": net, "method": m.modules_})
    opt = torch.optim.AdamW(c.parameters(), lr=cfg["optim"]["lr"], weight_decay=cfg["optim"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    peak = {}
    for n, mod in net.backbone.net.named_children():
        if n.startswith(("layer", "avgpool")):
            mod.register_forward_hook(
                lambda mod, i, o, n=n: peak.__setitem__(n, max(peak.get(n, 0.0), o.detach().abs().max().item())))
    total = cfg["train"]["max_epochs"] * cfg["data"]["steps_per_epoch"]
    loader, _ = train_loader(load_splits(), m.uses_target, num_workers=2, steps=steps)
    c.train(); freeze_bn_stats(net)
    fmt = lambda v: "inf" if not math.isfinite(v) else round(v)
    for s, (x, y, d, _) in enumerate(loader, 1):
        x, y, d = x.to(dev), y.to(dev), d.to(dev)
        with torch.autocast("cuda", torch.float16, enabled=amp):
            f, l = net(x)
        loss, logs = m.loss(f.float(), l.float(), y, d, (s - 1) / total)
        bad = not torch.isfinite(loss)
        if bad or s % 25 == 0:
            print(s, "loss %.3f" % loss.item(), {k: round(v, 3) for k, v in logs.items()},
                  "| peak:", {k: fmt(v) for k, v in peak.items()}, flush=True)
            peak.clear()
        if bad:
            print("NON-FINITE at step", s)
            break
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    print("finished", "amp" if amp else "fp32")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] == "amp", int(sys.argv[3]), sys.argv[4:])
