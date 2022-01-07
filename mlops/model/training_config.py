"""Contains the TrainingConfig class."""
# pylint: disable = no-name-in-module

from typing import Any, Dict
from dataclasses import dataclass
from tensorflow.keras.callbacks import History


@dataclass
class TrainingConfig:
    """Contains training configuration and results.

    history: The model's training history.
    train_args: The training arguments.
    """
    history: History
    train_args: Dict[str, Any]
