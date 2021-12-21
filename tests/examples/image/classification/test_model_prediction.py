"""Tests model_prediction.py."""
# pylint: disable=no-name-in-module,no-member

import os
import json
import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model import VersionedModel
from mlops.examples.image.classification.publish_dataset import \
    DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION
from mlops.examples.image.classification.train_model import train_model, \
    publish_model
from mlops.examples.image.classification import model_prediction
from mlops.examples.image.classification.pokemon_classification_data_processor \
    import DEFAULT_DATASET_PRED_PATH
from mlops.examples.image.classification.errors import NoModelPathsSuppliedError
from tests.examples.image.classification.test_train_model import _create_dataset

TEST_MODEL_PUBLICATION_PATH_LOCAL = ('/tmp/test_model_prediction/models/'
                                     'versioned')
EXPECTED_PRED_SHAPE = (3, 2)
FAKE_TRAIN_HISTORIES = [{'loss': [1, 2, 3], 'val_loss': [4, 5, 6]},
                        {'loss': [9, 5, 1], 'val_loss': [7, 7, 7]},
                        {'loss': [2, 2, 2], 'val_loss': [4, 5, 1]}]
BEST_VAL_IDX = 2


def _create_model() -> str:
    """Creates a test model.

    :return: The model's publication path.
    """
    dataset = VersionedDataset(os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                            DATASET_VERSION))
    model = Sequential([
        Flatten(input_shape=dataset.X_train.shape[1:]),
        Dense(dataset.y_train.shape[1])])
    model.compile('adam', loss='mse')
    train_kwargs = {'epochs': 3, 'batch_size': 8}
    training_config = train_model(
        model,
        dataset,
        **train_kwargs)
    return publish_model(model,
                         dataset,
                         training_config,
                         TEST_MODEL_PUBLICATION_PATH_LOCAL)


def test_model_evaluate_returns_valid_loss() -> None:
    """Tests that model evaluation returns a loss value that is valid."""
    _create_dataset()
    publication_path = _create_model()
    dataset = VersionedDataset(os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                            DATASET_VERSION))
    model = VersionedModel(publication_path)
    assert model_prediction.model_evaluate(dataset, model) >= 0


def test_model_predict_correct_shape() -> None:
    """Tests that model predictions are of the correct shape."""
    _create_dataset()
    publication_path = _create_model()
    dataset = VersionedDataset(os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                            DATASET_VERSION))
    model = VersionedModel(publication_path)
    features = dataset.data_processor.get_preprocessed_features(
        DEFAULT_DATASET_PRED_PATH)['X_pred']
    predictions = model_prediction.model_predict(features, dataset, model)
    assert predictions.shape == EXPECTED_PRED_SHAPE


def test_get_best_model_with_no_models_raises_error() -> None:
    """Tests that get_best_model raises an error when an empty path collection
    is supplied."""
    with pytest.raises(NoModelPathsSuppliedError):
        _ = model_prediction.get_best_model([])


def test_get_best_model_gets_lowest_val_error() -> None:
    """Tests that get_best_model gets the model with the lowest validation
    error."""
    _create_dataset()
    publication_paths = []
    for fake_train_history in FAKE_TRAIN_HISTORIES:
        publication_path = _create_model()
        publication_paths.append(publication_path)
        with open(os.path.join(publication_path, 'meta.json'), 'r',
                  encoding='utf-8') as infile:
            contents = json.loads(infile.read())
        contents['history'] = fake_train_history
        with open(os.path.join(publication_path, 'meta.json'), 'w',
                  encoding='utf-8') as outfile:
            outfile.write(json.dumps(contents))
    best_model = model_prediction.get_best_model(publication_paths)
    assert best_model.path == publication_paths[BEST_VAL_IDX]
