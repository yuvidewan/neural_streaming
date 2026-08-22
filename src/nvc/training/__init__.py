"""Training loop, checkpointing, and resume logic for BaselineAutoencoder.

Kept separate from src/nvc/models/ so a later milestone can swap in a
different model (e.g. a VAE, or a quantized codec) without rewriting the
epoch loop or checkpoint format.

- trainer:            train_one_epoch / validate_one_epoch
- checkpoint:          save_checkpoint / load_checkpoint / resume_training_state
- quantization_noise:  QuantizationNoise, the Milestone 8A QAT relaxation
                       (distortion-only; disabled unless explicitly attached
                       to a model - see BaselineAutoencoder)
"""

from .checkpoint import (
    load_checkpoint,
    load_model_from_checkpoint,
    resume_training_state,
    save_checkpoint,
)
from .quantization_noise import QuantizationNoise
from .trainer import train_one_epoch, validate_one_epoch

__all__ = [
    "train_one_epoch",
    "validate_one_epoch",
    "save_checkpoint",
    "load_checkpoint",
    "load_model_from_checkpoint",
    "resume_training_state",
    "QuantizationNoise",
]
