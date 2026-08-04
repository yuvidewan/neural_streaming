"""Train/eval transform pipelines for FrameDataset.

Kept deliberately minimal: video frames have "natural" up-is-up,
forward-is-forward statistics that most classic image augmentations don't
respect. Only RandomHorizontalFlip (mirroring a scene left-right is a
realistic viewpoint) and an optional fixed-size RandomCrop are used for
training - nothing else. No vertical flip, color jitter, rotation, or
perspective warp: those would teach the model to reconstruct frames that
don't look like real video, which defeats the point of a reconstruction
task.

Both functions operate on tensors (not PIL images), matching what
FrameDataset produces.
"""

from __future__ import annotations

import torchvision.transforms as T


def build_train_transform(crop_size: int | None = None) -> T.Compose:
    """Random crop (if configured) + random horizontal flip. Non-deterministic."""
    ops: list = []
    if crop_size is not None:
        ops.append(T.RandomCrop(crop_size))
    ops.append(T.RandomHorizontalFlip(p=0.5))
    return T.Compose(ops)


def build_eval_transform(crop_size: int | None = None) -> T.Compose:
    """Deterministic counterpart to build_train_transform: a center crop to
    the same output size (if configured), and nothing else - no flipping,
    so validation/test results are reproducible run to run.
    """
    ops: list = []
    if crop_size is not None:
        ops.append(T.CenterCrop(crop_size))
    return T.Compose(ops)
