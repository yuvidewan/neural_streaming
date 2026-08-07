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
            "checkpoint_dir", "visualizations_dir",
        }
        tuple_fields = {"supported_video_extensions", "supported_image_extensions"}
        for key, value in overrides.items():
            if key in path_fields:
                value = Path(value)
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
