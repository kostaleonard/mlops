"""Tests training_config.py."""

from tensorflow.keras.callbacks import History
from mlops.model.training_config import TrainingConfig


def test_training_config_init() -> None:
    """Tests that TrainingConfig objects are initialized correctly."""
    history = History()
    config = TrainingConfig(history, {})
    assert isinstance(config.history, History)
    assert isinstance(config.train_args, dict)
