"""CPU/CUDA device selection."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available torch.device, printing what was selected.

    Prefers CUDA when available; falls back to CPU otherwise. Nothing in
    this project requires CUDA - this just lets callers use a GPU
    automatically when one is present.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[DEVICE] GPU available: {name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        print("[DEVICE] No CUDA GPU available - using CPU")
    return device
