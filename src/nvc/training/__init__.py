"""Training loop, checkpointing, and resume logic for BaselineAutoencoder.

Kept separate from src/nvc/models/ so a later milestone can swap in a
different model (e.g. a VAE, or a quantized codec) without rewriting the
epoch loop or checkpoint format.

- trainer:            train_one_epoch / validate_one_epoch, and their
                       Milestone 9A D+lambda*R counterparts
                       train_one_epoch_with_rate / validate_one_epoch_with_rate
- checkpoint:          save_checkpoint / load_checkpoint / resume_training_state
- quantization_noise:  QuantizationNoise, the Milestone 8A QAT relaxation
                       (distortion-only; disabled unless explicitly attached
                       to a model - see BaselineAutoencoder)
- rate_estimator:      RateEstimator, the Milestone 9A differentiable rate
                       proxy (training-only - see its own module docstring
                       for the distinction from the deployed entropy model)
"""

from .checkpoint import (
    load_checkpoint,
    load_model_from_checkpoint,
    resume_training_state,
    save_checkpoint,
)
from .quantization_noise import QuantizationNoise
from .rate_estimator import RateEstimator
from .trainer import (
    train_one_epoch,
    train_one_epoch_with_rate,
    validate_one_epoch,
    validate_one_epoch_with_rate,
)

__all__ = [
    "train_one_epoch",
    "validate_one_epoch",
    "train_one_epoch_with_rate",
    "validate_one_epoch_with_rate",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_from_checkpoint",
    "resume_training_state",
    "QuantizationNoise",
    "RateEstimator",
]
