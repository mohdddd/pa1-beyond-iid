"""Task 4 training (Vanilla, GCSC, PROSER). CIFAR-10 only; CIFAR-100 access is blocked in this process.

  python -m task4.train --config task4/configs/vanilla.yaml          # resumes automatically after a disconnect
  python -m task4.train --config task4/configs/vanilla.yaml --smoke  # tiny end-to-end check (throw-away dirs)

Checkpoints: $PA1_STORAGE/checkpoints/task4/<run>/{last.pt (full state), best.pt (weights), status.json}.
Results:     task4/results/<run>/{train_epochs.csv, run_meta.json}.
Selection:   highest CIFAR-10 validation accuracy on the 10 known-class logits (ties -> earliest epoch).
"""
import argparse
import copy
import math
import os
import shutil
import time

os.environ["PA1_FORBID_UNKNOWN"] = "1"          # no CIFAR-100 image may reach training/selection

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from common.checkpoint import load_checkpoint, save_checkpoint
from common.io import load_json, load_yaml, save_json, save_run_metadata, sha256_file
from common.logging_utils import CSVLogger
from common.paths import REPO_ROOT, storage_dir, task_dir
from common.seed import SEED, seed_worker, set_seed
from task4.data.cifar10 import ArrayDataset, load_split, to_eval_tensor, train_transform
from task4.methods import build_method
from task4.models.resnet_cifar import CifarResNet18

CONFIG_DIR = REPO_ROOT / "task4/configs"


def load_config(path, overrides=()):
    cfg = load_yaml(REPO_ROOT / path if not os.path.isabs(path) else path)
    if "inherit" in cfg:
        base = load_config(CONFIG_DIR / cfg.pop("inherit"))
        cfg = _merge(base, cfg)
    for ov in overrides:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@torch.no_grad()
def evaluate(method, model, x_eval, y_eval, device, amp, bs, channels_last):
    model.eval()
    correct, loss = 0, 0.0
    for i in range(0, len(y_eval), bs):
        x = x_eval[i:i + bs].to(device, non_blocking=True)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = y_eval[i:i + bs].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            z = method.known_logits(model, x).float()
        loss += F.cross_entropy(z, y, reduction="sum").item()
        correct += (z.argmax(1) == y).sum().item()
    return correct / len(y_eval), loss / len(y_eval)


def run(cfg, smoke=False, fresh=False):
    run_name = cfg["run_name"] + ("_smoke" if smoke else "")
    ckpt_dir = storage_dir("checkpoints", "task4") / run_name
    res_dir = task_dir("task4", "results", run_name)
    if fresh or smoke:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        shutil.rmtree(res_dir, ignore_errors=True)
        res_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    status_path = ckpt_dir / "status.json"
    if status_path.exists() and load_json(status_path).get("finished"):
        print(f"{run_name} already finished (use --fresh to retrain).")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    cl = bool(cfg["train"]["channels_last"])
    set_seed(cfg["seed"], deterministic=False)     # seeds fixed; cudnn autotuning allowed for speed
    torch.backends.cudnn.benchmark = device.type == "cuda"

    # ---- data (CIFAR-10 train/val splits only)
    xtr, ytr, _ = load_split(cfg["data"]["splits"], "train")
    xva, yva, _ = load_split(cfg["data"]["splits"], "val")
    train_ds = ArrayDataset(xtr, ytr, train_transform(cfg["data"].get("randaugment")))
    sub = 512 if smoke else cfg["data"].get("train_subset")      # testing only; null in all configs
    if sub:
        train_ds = Subset(train_ds, range(sub))
        xva, yva = xva[: sub // 2], yva[: sub // 2]
    x_val, y_val = to_eval_tensor(xva), torch.from_numpy(yva)

    # ---- model + method (model built right after seeding -> identical init across runs)
    model = CifarResNet18(cfg["model"]["num_classes"])
    method = build_method(cfg)
    model = method.setup(model, device).to(device)
    if cl:
        model = model.to(memory_format=torch.channels_last)
    o = cfg["optim"]
    opt = torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"])
    epochs = 2 if smoke else cfg["train"]["epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    start, best = 1, -1.0
    if (ckpt_dir / "last.pt").exists():
        st = load_checkpoint(ckpt_dir / "last.pt", model, opt, sched, scaler, map_location=device)
        start, best = st["epoch"] + 1, st["best_metric"]
        print(f"resumed after epoch {st['epoch']} (best val acc {best:.4f})")
    logger = CSVLogger(res_dir / "train_epochs.csv", resume=start > 1)
    save_run_metadata(res_dir / "run_meta.json", cfg, device=str(device), amp=amp, smoke=smoke)

    bs, nw = cfg["data"]["batch_size"], (0 if smoke else cfg["data"]["num_workers"])
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        g = torch.Generator().manual_seed(SEED * 1000 + epoch)      # per-epoch order -> exact resume
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, generator=g,
                            worker_init_fn=seed_worker, pin_memory=device.type == "cuda", drop_last=False)
        model.train()
        sums, n, correct, nk = {}, 0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if cl:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss, logs, zk, yk = method.loss(model, x, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if not amp and not math.isfinite(loss.item()):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            b = len(y)
            for k, v in {"loss": loss.detach(), **logs}.items():
                sums[k] = sums.get(k, 0.0) + v.float() * b
            n += b
            correct += (zk.argmax(1) == yk).sum()
            nk += len(yk)
        lr = sched.get_last_lr()[0]
        sched.step()
        val_acc, val_loss = evaluate(method, model, x_val, y_val, device, amp, cfg["train"]["eval_batch"], cl)
        row = {"epoch": epoch, "lr": lr, **{f"train_{k}": (v / n).item() for k, v in sums.items()},
               "train_acc": (correct / nk).item(), "val_acc": val_acc, "val_loss": val_loss,
               "seconds": round(time.time() - t0, 1)}
        improved = val_acc > best
        if improved:
            best = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc,
                        "config": cfg}, ckpt_dir / "best.tmp")
            os.replace(ckpt_dir / "best.tmp", ckpt_dir / "best.pt")
        logger.log({**row, "best": int(improved)})
        save_checkpoint(ckpt_dir / "last.pt", model, opt, sched, scaler, epoch=epoch, best_metric=best)
        if os.environ.get("PA1_TEST_STOP_AFTER") == str(epoch):     # resume test hook
            raise SystemExit("test stop")
        print(f"ep {epoch:3d}/{epochs}  loss {row['train_loss']:.4f}  train {row['train_acc']:.4f}  "
              f"val {val_acc:.4f}{' *' if improved else ''}  {row['seconds']}s", flush=True)

    best_state = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    summary = {"run": run_name, "best_epoch": best_state["epoch"], "best_val_acc": best_state["val_acc"],
               "best_pt_sha256": sha256_file(ckpt_dir / "best.pt"), "epochs": epochs}
    save_json(summary, res_dir / "best.json")
    save_json({"finished": True, **summary}, status_path)
    print("finished:", summary)
    if smoke:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        shutil.rmtree(res_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    run(load_config(a.config, a.set), smoke=a.smoke, fresh=a.fresh)


if __name__ == "__main__":
    main()
