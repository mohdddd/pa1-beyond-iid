"""Task 3 source-side diagnostics — everything that is computed WITHOUT Sketch
(this process cannot load Sketch: PA1_FORBID_TARGET=1 and PACSStore drops its rows).

    python -m task3.source_diagnostics           # all finished runs
    python -m task3.source_diagnostics --debug   # sandbox test on smoke checkpoints

For every Task 3 model (ERM, DAN-DG, SAM, study runs):
  * verified checkpoint (SHA-256 vs committed run_meta / erm.yaml);
  * source-val accuracy / macro-F1 per domain, mean, worst (fp32);
  * source-domain separability (3-way, balanced, chance 1/3);
  * sharpness proxy on the fixed 3 x 32 validation batch.
Outputs: task3/results/source_side/{diagnostics.json, sharpness_batch_ids.json},
report/tables/task3_source_side.csv, report/figures/task3/task3_training_curves.pdf|png,
features cached in $PA1_STORAGE/cache/task3/source_<run>.npz.
Run this BEFORE the Sketch lock; evaluate_sketch.py records its file hash in the lock.
"""
import os

os.environ["PA1_FORBID_TARGET"] = "1"

import argparse  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from common.io import git_commit_hash, load_json, save_json  # noqa: E402
from common.paths import FIG_DIR, REPO_ROOT, TABLE_DIR, storage_dir  # noqa: E402
from common.seed import SEED, set_seed  # noqa: E402
from shared.pacs import PACSStore  # noqa: E402
from shared.pacs_protocol import load_splits  # noqa: E402
from task3.evaluation.curves import training_curves  # noqa: E402
from task3.evaluation.domain_metrics import domain_report, source_row  # noqa: E402
from task3.evaluation.sharpness import fixed_batch, sharpness  # noqa: E402
from task3.evaluation.source_domain_separability import source_separability  # noqa: E402
from task3.selection.source_validation import (RES, available_runs, checkpoint, load_model,  # noqa: E402
                                               predict, source_val_loader)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--debug", action="store_true", help="smoke checkpoints, subset, temp outputs")
    a = ap.parse_args()
    assert PACSStore.get().target_excluded
    set_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits()

    if a.debug:
        root = Path(tempfile.gettempdir()) / "pa1_debug_task3"
        out_res, out_tab, out_fig, cache = root / "source_side", root / "tables", root / "figures", root / "cache"
        smoke = Path(tempfile.gettempdir()) / "pa1_smoke"
        ck = {"erm": smoke / "task2" / "smoke_source_only" / "best.pt",
              **{r: smoke / "task3" / f"smoke_{r}" / "best.pt" for r in ("dan_dg", "sam")}}
        runs = [r for r in (a.runs or ck) if ck[r].exists()]
        res3, res2 = smoke / "task3" / "results", smoke / "task2" / "results"
        curve_names = {r: (f"smoke_{r}" if r != "erm" else "erm") for r in runs}
    else:
        out_res, out_tab, out_fig = RES / "source_side", TABLE_DIR, FIG_DIR / "task3"
        cache = storage_dir("cache", "task3")
        runs = a.runs or available_runs()
        res3, res2 = RES, REPO_ROOT / "task2" / "results"
    for p in (out_res, out_tab, out_fig, cache):
        p.mkdir(parents=True, exist_ok=True)

    vl = source_val_loader(splits, num_workers=2)
    if a.debug:
        n = [len(d) for d in vl.dataset.datasets]
        idx = [o + i for o, k in zip(np.cumsum([0] + n[:-1]), n) for i in range(min(40, k))]
        vl = DataLoader(Subset(vl.dataset, idx), batch_size=64, num_workers=0)
    xb, yb, ids = fixed_batch(splits)
    save_json({"seed": SEED, "per_source": 32, "ids": ids}, out_res / "sharpness_batch_ids.json")

    diag, rows = {}, []
    for r in runs:
        if a.debug:
            net, digest = load_model(r, dev, verify=False, path=ck[r]), "debug"
        else:
            path, digest = checkpoint(r, verify=True)
            net = load_model(r, dev, verify=False, path=path)
        pr = predict(net, vl, dev, return_features=True, amp=False)
        np.savez(cache / f"source_{r}.npz", **pr)
        rep = domain_report(pr["pred"], pr["label"], pr["domain"])
        sep = source_separability(pr["feat"], pr["domain"])
        sh = sharpness(net, xb, yb)
        meta = {} if a.debug else load_json(RES / r / "run_meta.json")
        diag[r] = {"sha256": digest, "best_epoch": meta.get("best_epoch"), "source_val": rep,
                   "separability": sep, "sharpness": sh,
                   "feat_norm_mean": float(np.linalg.norm(pr["feat"], axis=1).mean())}
        rows.append({"method": r, "best_epoch": meta.get("best_epoch"), **source_row(rep),
                     "src_domain_separability": sep["accuracy"], "separability_converged": sep["converged"],
                     "sharpness_delta": sh["delta_sharp"], "sharpness_loss": sh["loss"],
                     "sharpness_grad_norm": sh["grad_norm"], "feat_norm_mean": diag[r]["feat_norm_mean"]})
        print(f"[{r}] val mean F1 {rep['mean_macro_f1']:.4f} worst {rep['worst_macro_f1']:.4f} "
              f"({rep['worst_domain_by_f1']}) | separability {sep['accuracy']:.3f} (chance .333) | "
              f"Δsharp {sh['delta_sharp']:.4f} (L {sh['loss']:.4f}, ||g|| {sh['grad_norm']:.3f})")

    save_json({"git_commit": git_commit_hash(), "runs": diag}, out_res / "diagnostics.json")
    pd.DataFrame(rows).to_csv(out_tab / "task3_source_side.csv", index=False, float_format="%.4f")
    fig = training_curves(runs if not a.debug else list(curve_names.values()), res3, res2) if not a.debug \
        else None
    if fig is not None:
        fig.savefig(out_fig / "task3_training_curves.pdf", bbox_inches="tight")
        fig.savefig(out_fig / "task3_training_curves.png", bbox_inches="tight", dpi=150)
    print(f"[source-side] -> {out_res / 'diagnostics.json'}, {out_tab / 'task3_source_side.csv'}")


if __name__ == "__main__":
    main()
