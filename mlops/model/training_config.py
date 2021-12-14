"""Contains the TrainingConfig class."""

from typing import Any
from dataclasses import dataclass
from tensorflow.keras.callbacks import History

# TODO test script


@dataclass
class TrainingConfig:
    """Contains training configuration and results.

    history: The model's training history.
    train_args: The training arguments.
    """
    history: History
    train_args: dict[str, Any]
