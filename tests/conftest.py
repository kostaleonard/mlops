"""Contains test fixtures."""
# pylint: disable=no-name-in-module

import os
import pytest
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.training_config import TrainingConfig
from tests.dataset.test_versioned_dataset import _publish_test_dataset_local, \
    TEST_PUBLICATION_PATH_LOCAL as TEST_DATASET_PUBLICATION_PATH_LOCAL


@pytest.fixture(name='dataset')
def fixture_dataset() -> VersionedDataset:
    """Returns the versioned dataset fixture for testing.

    :return: The versioned dataset fixture.
    """
    _publish_test_dataset_local()
    return VersionedDataset(os.path.join(TEST_DATASET_PUBLICATION_PATH_LOCAL,
                                         'v1'))


@pytest.fixture(name='model')
def fixture_model(dataset: VersionedDataset) -> Model:
    """Returns the model fixture for testing.

    :param dataset: The versioned dataset.
    :return: The model fixture.
    """
    mod = Sequential([Dense(dataset.y_train.shape[1],
                            input_shape=dataset.X_train.shape[1:])])
    mod.compile('adam', loss='mse')
    return mod


@pytest.fixture(name='training_config')
def fixture_training_config(dataset: VersionedDataset,
                            model: Model) -> TrainingConfig:
    """Returns the training configuration fixture for testing.

    :param dataset: The versioned dataset.
    :param model: The model.
    :return: The training configuration fixture.
    """
    train_kwargs = {'epochs': 5,
                    'batch_size': 8}
    history = model.fit(x=dataset.X_train,
                        y=dataset.y_train,
                        **train_kwargs)
    return TrainingConfig(history, train_kwargs)
