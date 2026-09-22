"""Task 3 training entry point (no Sketch data can be loaded by this process).

    python -m task3.train --config task3/configs/erm.yaml       # verify + re-evaluate reused ERM (no training)
    python -m task3.train --config task3/configs/dan_dg.yaml    # DAN-DG, lambda_dg = 1
    python -m task3.train --config task3/configs/sam.yaml       # SAM, rho = 0.05
    python -m task3.train --config task3/configs/dan_dg.yaml --set method.lambda_dg=0.1 --run-name dan_dg_l0p1
    python -m task3.train --config task3/configs/dan_dg.yaml --set method.lambda_dg=10  --run-name dan_dg_l10
    add --smoke for a 2 x 5-step test in temporary folders

Trained runs use the exact Task 2 training loop (task2/train.py) with the Task 3
method registry: same splits, 8+8+8 batches, augmentation seeds, initialisation,
AdamW, fp32, frozen BN statistics, <= 30 epochs, patience 5 on mean source-val
macro-F1. Checkpoints -> $PA1_STORAGE/checkpoints/task3/<run>/, results ->
task3/results/<run>/.
"""
import os

os.environ["PA1_FORBID_TARGET"] = "1"          # must precede any PACS access

import argparse  # noqa: E402

import torch  # noqa: E402

from common.io import save_json, save_run_metadata  # noqa: E402
from common.paths import REPO_ROOT  # noqa: E402
from shared.pacs import PACSStore  # noqa: E402
from shared.pacs_protocol import load_splits  # noqa: E402
from task2.train import load_config, run  # noqa: E402
from task3.methods import build_method  # noqa: E402
from task3.methods.erm import verify_erm_checkpoint  # noqa: E402
from task3.selection.source_validation import load_model, source_val_loader, source_val_report  # noqa: E402


def register_erm(cfg: dict) -> dict:
    """ERM = Task 2 source_only best.pt, unchanged: verify its hash, re-evaluate on source val,
    check the value matches the one recorded when it was selected, and write task3/results/erm/."""
    info = verify_erm_checkpoint()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_model("erm", dev, verify=False, path=info["path"])
    rep = source_val_report(net, source_val_loader(load_splits()), dev, amp=False)
    recorded = info["task2_meta"]["best_val_mean_macro_f1"]
    diff = abs(rep["mean_macro_f1"] - recorded)
    res = REPO_ROOT / "task3" / "results" / "erm"
    res.mkdir(parents=True, exist_ok=True)
    save_json({"epoch": info["task2_meta"]["best_epoch"], **rep}, res / "source_val.json")
    save_run_metadata(res / "run_meta.json", cfg, run_name="erm", method="erm",
                      reused_from="task2/results/source_only", best_ckpt=info["path"],
                      best_sha256=info["sha256"], best_epoch=info["task2_meta"]["best_epoch"],
                      best_val_mean_macro_f1=rep["mean_macro_f1"],
                      task2_recorded_val_mean_macro_f1=recorded, abs_diff_vs_task2=diff)
    print(f"[erm] sha256 {info['sha256'][:12]} verified (Task 2 source_only best.pt, epoch "
          f"{info['task2_meta']['best_epoch']})")
    print("[erm] val F1 " + " ".join(f"{d[:3]} {rep[d]['macro_f1']:.4f}" for d in ("photo", "art_painting", "cartoon"))
          + f" | mean {rep['mean_macro_f1']:.4f} | worst {rep['worst_macro_f1']:.4f} ({rep['worst_domain_by_f1']})"
          + f" | |Δ vs Task 2 record| = {diff:.2e}")
    if diff > 1e-6:
        print("[erm] WARNING: re-evaluation differs from the Task 2 record — report this")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. method.lambda_dg=0.1")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--keep-last", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.run_name:
        cfg["run_name"] = a.run_name
    assert PACSStore.get().target_excluded, "Sketch rows were loaded — PA1_FORBID_TARGET not effective"
    if cfg["method"]["name"] == "erm":
        register_erm(cfg)
        return
    run(cfg, smoke=a.smoke, fresh=a.fresh, keep_last=a.keep_last, task="task3", build=build_method)


if __name__ == "__main__":
    main()
