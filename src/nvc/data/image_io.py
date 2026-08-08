"""Single-image read/write in the project's tensor convention.

One place defines what "an image as a tensor" means here - `[3, H, W]`
float32 in `[0, 1]`, RGB channel order - so FrameDataset and the codec CLIs
cannot drift apart on it.

OpenCV is used for the actual file I/O, consistent with the rest of
`src/nvc/data/`. cv2 always works in BGR in memory regardless of the file's
real colors, so reads convert BGR->RGB and writes convert back.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from nvc.data.validation import DatasetValidationError


def read_image_as_tensor(path: str | Path) -> torch.Tensor:
    """Read an image file as a [3, H, W] float32 RGB tensor in [0, 1]."""
    path = Path(path)
    image = cv2.imread(str(path))
    if image is None:
        raise DatasetValidationError(
            f"Could not read image (missing, corrupt, or unsupported): {path}"
        )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0


def write_tensor_as_image(tensor: torch.Tensor, path: str | Path) -> Path:
    """Write a [3, H, W] or [1, 3, H, W] float tensor in [0, 1] as an image file.

    Values are clamped to [0, 1] before scaling to uint8: a decoder output
    can sit a hair outside the range through floating-point error, and
    silently wrapping around would produce speckled garbage pixels.
    """
    if tensor.dim() == 4:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Expected a single image, got a batch of {tensor.shape[0]}"
            )
        tensor = tensor[0]
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        raise ValueError(
            f"Expected a [3, H, W] RGB tensor, got shape {tuple(tensor.shape)}"
        )

    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor((array * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise DatasetValidationError(f"Could not write image to {path}")
    return path
