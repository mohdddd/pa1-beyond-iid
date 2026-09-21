"""Single training loop for every Task 2 method (and reused by Task 3).

    python -m task2.train --config task2/configs/source_only.yaml
    python -m task2.train --config task2/configs/source_only.yaml --smoke   # quick test

Protocol (manual §2): ResNet-18 IMAGENET1K_V1 + 7-class head, full fine-tuning,
frozen BN running statistics, AdamW(lr 1e-4, wd 1e-4), <= 30 source epochs,
early stopping after 5 epochs without improved mean source-val macro-F1,
seed 6304, domain-balanced batches (8 per source [+ 24 unlabeled target]).
Only SOURCE validation data is evaluated here — the target is never scored.

Storage
  PA1_STORAGE/checkpoints/task2/<run>/  last.pt (resume; deleted when finished),
                                        best.pt (network only), logs, status.json
  task2/results/<run>/                  train_steps.csv, train_epochs.csv,
                                        source_val.json, run_meta.json (committed)
A finished run is never retrained unless --fresh is given (protects the
Source-only checkpoint that Task 3 must reuse unchanged).
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from common.checkpoint import load_checkpoint, save_checkpoint
from common.io import load_json, load_yaml, save_json, save_run_metadata
from common.logging_utils import CSVLogger
from common.paths import REPO_ROOT, storage_dir
from common.seed import set_seed
from shared.pacs_protocol import load_splits, train_loader
from task2.evaluation.metrics import flat_row, source_val_loader, source_val_report
from task2.methods import build_method
from task2.models.backbone import freeze_bn_stats
from task2.models.classifier_head import Net

CONFIG_DIR = REPO_ROOT / "task2" / "configs"


# ---------------------------------------------------------------- config
def _deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        a[k] = _deep_update(a.get(k, {}), v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def load_config(path, overrides=(), base=CONFIG_DIR / "base.yaml") -> dict:
    cfg = _deep_update(copy.deepcopy(load_yaml(base)), load_yaml(path))
    for ov in overrides:                     # e.g. method.lambda_mmd=0.1
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


# ---------------------------------------------------------------- helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def trim_csv(path: Path, max_epoch: int) -> None:
    """Drop rows logged after the last completed (checkpointed) epoch."""
    if not path.exists():
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    keep = [r for r in rows if int(r["epoch"]) <= max_epoch]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(keep)


def bn_stats_match_imagenet(net: Net) -> bool:
    """Verify BN running mean/var equal the pretrained ImageNet values."""
    from torchvision.models import ResNet18_Weights, resnet18
    ref = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    own = net.backbone.net.state_dict()
    keys = [k for k in ref if k.endswith(("running_mean", "running_var"))]
    return all(torch.equal(own[k].cpu(), ref[k]) for k in keys)


# ---------------------------------------------------------------- main
def run(cfg: dict, smoke: bool = False, fresh: bool = False, keep_last: bool = False,
        task: str = "task2") -> dict:
    name = cfg["run_name"]
    if smoke:
        name = f"smoke_{name}"
        run_dir = Path(tempfile.gettempdir()) / "pa1_smoke" / task / name
        res_dir = Path(tempfile.gettempdir()) / "pa1_smoke" / task / "results" / name
        cfg["train"]["max_epochs"] = 2
        cfg["data"]["steps_per_epoch"] = 5
    else:
        run_dir = storage_dir("checkpoints", task) / name
        res_dir = REPO_ROOT / task / "results" / name
    run_dir.mkdir(parents=True, exist_ok=True); res_dir.mkdir(parents=True, exist_ok=True)
    last_pt, best_pt, status = run_dir / "last.pt", run_dir / "best.pt", run_dir / "status.json"

    if status.exists() and load_json(status).get("finished") and not fresh:
        print(f"[{name}] already finished (best.pt sha256 {load_json(status)['best_sha256'][:12]}). "
              f"Not retraining. Use --fresh to override.")
        return load_json(status)
    if fresh:
        for p in (last_pt, best_pt, status, run_dir / "train_steps.csv", run_dir / "train_epochs.csv"):
            p.unlink(missing_ok=True)

    set_seed(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"]["amp"]) and dev.type == "cuda"
    splits = load_splits()

    # identical initialisation across methods: net is built right after seeding
    net = Net(cfg["model"]["num_classes"]).to(dev)
    method = build_method(cfg, cfg["model"]["feat_dim"], cfg["model"]["num_classes"])
    method.modules_.to(dev)
    container = nn.ModuleDict({"net": net, "method": method.modules_})
    opt = torch.optim.AdamW(container.parameters(), lr=cfg["optim"]["lr"],
                            weight_decay=cfg["optim"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    spe = cfg["data"]["steps_per_epoch"]
    loader, sampler = train_loader(splits, with_target=method.uses_target,
                                   num_workers=cfg["data"]["num_workers"], seed=cfg["seed"], steps=spe)
    vl = source_val_loader(splits, num_workers=cfg["data"]["num_workers"])
    if smoke:
        n = [len(d) for d in vl.dataset.datasets]
        idx = [o + i for o, k in zip([0, n[0], n[0] + n[1]], n) for i in range(min(16, k))]
        vl = DataLoader(Subset(vl.dataset, idx), batch_size=48, num_workers=0)

    st = {"epoch": 0, "global_step": 0, "best": -1.0, "best_epoch": 0, "bad": 0, "train_time": 0.0}
    if last_pt.exists():
        ck = load_checkpoint(last_pt, container, opt, None, scaler, map_location=dev)
        st = ck["extra"]["state"]
        sampler.load_state_dict(ck["extra"]["sampler"])
        print(f"[{name}] resumed after epoch {st['epoch']} (best {st['best']:.4f} @ epoch {st['best_epoch']})")
    for f in ("train_steps.csv", "train_epochs.csv"):
        trim_csv(run_dir / f, st["epoch"])
    step_log = CSVLogger(run_dir / "train_steps.csv")
    epoch_log = CSVLogger(run_dir / "train_epochs.csv")

    max_ep, patience = cfg["train"]["max_epochs"], cfg["train"]["patience"]
    total_steps = max_ep * spe
    n_bn = freeze_bn_stats(net)
    print(f"[{name}] device={dev} amp={amp} method={method.name} uses_target={method.uses_target} "
          f"steps/epoch={spe} max_epochs={max_ep} BN modules frozen={n_bn}")

    stop_reason = "max_epochs"
    while st["epoch"] < max_ep:
        if st["bad"] >= patience:
            stop_reason = "early_stopping"
            break
        ep = st["epoch"] + 1
        container.train(); freeze_bn_stats(net)          # BN policy, every epoch
        t0, acc, cnt = time.time(), {}, 0
        for x, y, dom, _ in loader:
            x, y, dom = (x.to(dev, non_blocking=True), y.to(dev, non_blocking=True),
                         dom.to(dev, non_blocking=True))
            progress = st["global_step"] / total_steps
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                feats, logits = net(x)
            loss, logs = method.loss(feats.float(), logits.float(), y, dom, progress)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {st['global_step']}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            st["global_step"] += 1; cnt += 1
            logs["loss"] = loss.item()
            for k, v in logs.items():
                acc[k] = acc.get(k, 0.0) + v
            if st["global_step"] % cfg["train"]["log_every"] == 0:
                step_log.log({"epoch": ep, "step": st["global_step"], "progress": round(progress, 5),
                              **{k: round(v, 6) for k, v in logs.items()}})
        st["train_time"] += time.time() - t0

        rep = source_val_report(net, vl, dev, amp=amp)
        metric = rep["mean_macro_f1"]
        improved = metric > st["best"]
        if improved:
            st.update(best=metric, best_epoch=ep, bad=0)
            save_checkpoint(best_pt, net, epoch=ep, best_metric=metric,
                            extra={"config": cfg, "val_report": rep})
        else:
            st["bad"] += 1
        st["epoch"] = ep
        epoch_log.log({"epoch": ep, "step": st["global_step"],
                       **{f"train_{k}": round(v / cnt, 6) for k, v in acc.items()},
                       **{k: round(v, 6) for k, v in flat_row(rep).items()},
                       "best": int(improved), "epoch_time_s": round(time.time() - t0, 1)})
        save_checkpoint(last_pt, container, opt, None, scaler, epoch=ep, best_metric=st["best"],
                        extra={"state": st, "sampler": sampler.state_dict(), "config": cfg})
        print(f"[{name}] ep {ep:2d} | loss {acc['loss'] / cnt:.4f} | val F1 "
              + " ".join(f"{d[:3]} {rep[d]['macro_f1']:.3f}" for d in ("photo", "art_painting", "cartoon"))
              + f" | mean {metric:.4f}{' *' if improved else ''} | {time.time() - t0:.0f}s")
    else:
        if st["bad"] >= patience:
            stop_reason = "early_stopping"

    # ------------------------------------------------ finalize
    ck = load_checkpoint(best_pt, net, map_location=dev, restore_rng=False)
    rep = source_val_report(net, vl, dev, amp=amp)
    assert abs(rep["mean_macro_f1"] - ck["best_metric"]) < 1e-9, "best checkpoint re-evaluation mismatch"
    bn_ok = bn_stats_match_imagenet(net)
    digest = sha256(best_pt)
    save_json({"epoch": ck["epoch"], **rep}, res_dir / "source_val.json")
    for f in ("train_steps.csv", "train_epochs.csv"):
        if (run_dir / f).exists():
            shutil.copyfile(run_dir / f, res_dir / f)
    info = {"run_name": name, "method": method.name, "best_epoch": st["best_epoch"],
            "best_val_mean_macro_f1": st["best"], "epochs_run": st["epoch"], "stop_reason": stop_reason,
            "global_steps": st["global_step"], "train_time_s": round(st["train_time"], 1),
            "best_ckpt": str(best_pt), "best_sha256": digest,
            "bn_running_stats_equal_imagenet": bn_ok, "bn_modules_frozen": n_bn}
    save_run_metadata(res_dir / "run_meta.json", cfg, **info)
    save_json({"finished": True, **info}, status)
    if not keep_last:
        last_pt.unlink(missing_ok=True)
    print(f"[{name}] done: best epoch {st['best_epoch']} mean val F1 {st['best']:.4f} "
          f"(worst {rep['worst_macro_f1']:.4f}, {rep['worst_domain_by_f1']}) | stop: {stop_reason} | "
          f"BN stats unchanged: {bn_ok} | sha256 {digest[:12]}")
    print(f"[{name}] results -> {res_dir}")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. method.lambda_mmd=0.1")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", action="store_true", help="2 epochs x 5 steps, tiny val, temp dirs")
    ap.add_argument("--fresh", action="store_true", help="discard an existing run and retrain")
    ap.add_argument("--keep-last", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.run_name:
        cfg["run_name"] = a.run_name
    run(cfg, smoke=a.smoke, fresh=a.fresh, keep_last=a.keep_last)


if __name__ == "__main__":
    main()
