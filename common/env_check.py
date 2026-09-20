"""Smoke test: `python -m common.env_check` (run from the repo root)."""
import importlib
import platform

import torch

from common.paths import CACHE_DIR, CKPT_DIR, DATA_DIR, STORAGE_ROOT, ensure_dirs
from common.seed import SEED, set_seed


def main():
    print(f"Python      : {platform.python_version()}")
    print(f"torch       : {torch.__version__} (CUDA {torch.version.cuda})")
    for pkg in ["torchvision", "open_clip", "sklearn", "umap", "datasets", "pandas", "matplotlib"]:
        try:
            m = importlib.import_module(pkg)
            print(f"{pkg:<12}: {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            print(f"{pkg:<12}: MISSING ({e.__class__.__name__})")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU         : {p.name}, {p.total_memory / 1e9:.1f} GB")
    else:
        print("GPU         : NOT AVAILABLE -> Runtime > Change runtime type > T4 GPU")
    ensure_dirs()
    for d in (STORAGE_ROOT, DATA_DIR, CKPT_DIR, CACHE_DIR):
        probe = d / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        print(f"writable    : {d}")
    set_seed(SEED)
    print(f"seed        : {SEED} -> torch.rand(1) = {torch.rand(1).item():.6f}")
    print("Environment OK.")


if __name__ == "__main__":
    main()
