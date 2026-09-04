"""Project-wide configuration for the Neural Video Compression engine.

A plain dataclass with placeholder defaults. Values here are NOT
experimentally tuned - they exist so later modules (data loading, model
training, evaluation) have a single, typed source of configuration to
import from instead of scattering magic numbers across the codebase.

Defaults can be overridden by editing configs/default.json; unknown keys
in that file raise an error rather than being silently ignored.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# src/nvc/utils/config.py -> parents[3] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


@dataclass
class Config:
    """Typed container for settings needed across the pipeline."""

    # --- Frame / video parameters ---
    frame_width: int = 256
    frame_height: int = 256

    # --- Model parameters ---
    latent_dim: int = 128
    latent_channels: int = 64  # BaselineAutoencoder's spatial latent channel count

    # --- Latent quantization parameters (Milestone 5) ---
    quantization_bits: int = 8
    quantization_mode: str = "per_channel"  # "global" or "per_channel"

    # --- Quantization-aware training / noise relaxation (Milestone 8A) ---
    # Distortion-only training-time latent perturbation; see
    # nvc.training.quantization_noise. Disabled by default so existing
    # training behavior is unchanged unless explicitly opted into.
    qat_enabled: bool = False
    qat_bits: int = 4  # bit depth the noise relaxation targets during training
    qat_mode: str = "per_channel"  # must match the calibration file's mode
    # Frozen calibration artifact (scripts/calibrate_quantizer.py output,
    # Vimeo TRAIN split only) supplying the training-time noise scale. None
    # is only valid when qat_enabled is False.
    qat_calibration_path: Path | None = None

    # --- Differentiable rate estimation / D + lambda*R training (Milestone 9A) ---
    # See nvc.training.rate_estimator.RateEstimator. Disabled by default so
    # existing training behavior (including QAT-only training) is unchanged
    # unless explicitly opted into.
    rate_enabled: bool = False
    # D + lambda * R. Must be >= 0 (see trainer._check_lambda_rate); 0.0 is
    # valid and reproduces the distortion-only objective exactly.
    rate_lambda: float = 0.0
    # Learning rate for the rate estimator's OWN parameters (loc/log_scale),
    # kept separate from `learning_rate` above (Milestone 9C.1). M9C measured
    # that these two need very different step sizes: fitting the latent needs
    # loc/log_scale to move by O(1)-O(5), but at the model's 1e-4 a 500-step
    # run can move a parameter by at most ~0.05, so the estimator stayed at
    # its initialization (scale 1.00 -> 1.02-1.05) and the rate term degraded
    # into an L1 penalty on latent magnitude instead of a fitted entropy
    # model. See MILESTONE_9_PLAN.md's M9C.1 section. Used only when
    # rate_enabled is True; must be finite and > 0.
    rate_lr: float = 1e-2

    # Calibration file supplying the rate estimator's bin width, used ONLY
    # when rate_enabled is True and qat_enabled is False. When both QAT and
    # rate training are enabled, the rate estimator instead reuses
    # quantization_noise.scale directly (see scripts/train_autoencoder.py) -
    # this path is for rate training WITHOUT QAT. None is only valid when
    # rate_enabled is False, or when qat_enabled is also True.
    rate_calibration_path: Path | None = None

    # --- Training parameters ---
    batch_size: int = 8
    learning_rate: float = 1e-4
    epochs: int = 50

    # --- Dataset and checkpoint paths (relative to the project root) ---
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    frames_dir: Path = PROJECT_ROOT / "data" / "frames"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    checkpoint_dir: Path = PROJECT_ROOT / "outputs" / "checkpoints"
    visualizations_dir: Path = PROJECT_ROOT / "outputs" / "visualizations"
    metrics_dir: Path = PROJECT_ROOT / "outputs" / "metrics"
    benchmarks_dir: Path = PROJECT_ROOT / "outputs" / "benchmarks"

    # --- Vimeo-90K large-scale training dataset (Milestone 6.5) ---
    # Not committed and not auto-downloaded - see README.md, "Large-Scale
    # Training Dataset", for how to obtain and place it.
    vimeo_root: Path = PROJECT_ROOT / "data" / "external" / "vimeo_septuplet"
    vimeo_manifest_path: Path = PROJECT_ROOT / "data" / "processed" / "vimeo_manifest.json"

    # --- Dataset preprocessing parameters ---
    every_n_frames: int = 1
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    supported_video_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    supported_image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")

    # --- PyTorch data pipeline parameters ---
    num_workers: int = 0
    random_crop_size: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        """Build a Config from a JSON file, overriding only given keys."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            overrides = json.load(f)

        valid_keys = {f.name for f in fields(cls)}
        unknown_keys = set(overrides) - valid_keys
        if unknown_keys:
            raise ValueError(
                f"Unknown config key(s) in {path}: {sorted(unknown_keys)}"
            )

        config = cls()
        path_fields = {
            "raw_data_dir", "frames_dir", "processed_data_dir",
            "checkpoint_dir", "visualizations_dir", "metrics_dir", "benchmarks_dir",
            "vimeo_root", "vimeo_manifest_path",
        }
        # Unlike path_fields above, these are Optional - None is their
        # documented default (required only when the corresponding
        # *_enabled flag is True), so they must not be forced through Path().
        nullable_path_fields = {"qat_calibration_path", "rate_calibration_path"}
        tuple_fields = {"supported_video_extensions", "supported_image_extensions"}
        for key, value in overrides.items():
            if key in path_fields:
                value = Path(value)
            elif key in nullable_path_fields:
                value = Path(value) if value is not None else None
            elif key in tuple_fields:
                value = tuple(value)
            setattr(config, key, value)
        return config

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (Path fields become strings)."""
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }


def load_default_config() -> Config:
    """Load configs/default.json if present, otherwise return built-in defaults."""
    if DEFAULT_CONFIG_PATH.exists():
        return Config.from_json(DEFAULT_CONFIG_PATH)
    return Config()
