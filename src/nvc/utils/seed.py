"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python's random, NumPy, and PyTorch (CPU and, if present, all GPUs).

    deterministic=True additionally asks PyTorch to use deterministic
    algorithms where available. This can be slower, and some ops without a
    deterministic implementation will raise - so it's opt-in, not the
    default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if no GPU is present

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
