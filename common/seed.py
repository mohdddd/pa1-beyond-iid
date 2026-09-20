"""Seeding utilities. Every experiment in PA1 uses seed 6304."""
import os
import random

import numpy as np
import torch

SEED = 6304


def set_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Pass as DataLoader(worker_init_fn=seed_worker) so augmentation is reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = SEED) -> torch.Generator:
    """Pass as DataLoader(generator=...) so shuffling order is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
