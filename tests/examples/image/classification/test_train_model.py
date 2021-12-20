"""Tests train_model.py."""

import os
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.examples.image.classification.publish_dataset import \
    publish_dataset, DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION
from mlops.examples.image.classification import train_model

TEST_DATASET_PUBLICATION_PATH_LOCAL = '/tmp/test_train_model/datasets'
TEST_CHECKPOINT_PATH = '/tmp/test_train_model/models/checkpoints'
TEST_MODEL_PUBLICATION_PATH_LOCAL = '/tmp/test_train_model/models/versioned'


def _create_dataset() -> None:
    """Creates the dataset files."""
    try:
        shutil.rmtree(TEST_DATASET_PUBLICATION_PATH_LOCAL)
    except FileNotFoundError:
        pass
    publish_dataset(TEST_DATASET_PUBLICATION_PATH_LOCAL)


def test_get_baseline_model_correct_shapes() -> None:
    """Tests that the baseline model has the correct input and output shapes."""
    _create_dataset()
    dataset = VersionedDataset(os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                            DATASET_VERSION))
    model = train_model.get_baseline_model(dataset)
    assert model.input_shape[1:] == dataset.X_train.shape[1:]
    assert model.output_shape[1:] == dataset.y_train.shape[1:]


def test_train_model_creates_checkpoints() -> None:
    """Tests that train_model creates model checkpoints."""
    _create_dataset()
    try:
        shutil.rmtree(TEST_CHECKPOINT_PATH)
    except FileNotFoundError:
        pass
    dataset = VersionedDataset(os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                            DATASET_VERSION))
    model = Sequential([
        Flatten(input_shape=dataset.X_train.shape[1:]),
        Dense(dataset.y_train.shape[1])])
    model.compile('adam', loss='mse')
    train_kwargs = {'epochs': 3, 'batch_size': 8}
    model_checkpoint_filename = os.path.join(TEST_CHECKPOINT_PATH, 'model.h5')
    _ = train_model.train_model(
        model,
        dataset,
        model_checkpoint_filename=model_checkpoint_filename,
        **train_kwargs)
    assert os.path.exists(model_checkpoint_filename)


def test_train_model_returns_correct_training_config() -> None:
    """Tests that train_model returns a TrainingConfig object with the correct
    information."""
    # TODO
    assert False


def test_publish_model_creates_files() -> None:
    """Tests that publish_model creates the published model files."""
    # TODO
    assert False
