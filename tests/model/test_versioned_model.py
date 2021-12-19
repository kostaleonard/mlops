"""Tests versioned_model.py."""

import os
import pytest
from tensorflow.keras.models import Model
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.model.training_config import TrainingConfig
from mlops.model.versioned_model import VersionedModel
from tests.model.test_versioned_model_builder import \
    _remove_test_directories_local, _remove_test_directories_s3, \
    TEST_MODEL_PUBLICATION_PATH_LOCAL, TEST_MODEL_PUBLICATION_PATH_S3, \
    dataset, model, training_config

EXPECTED_ATTRIBUTES = {'model', 'md5'}


def _publish_test_model_local(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Publishes the versioned model to the local filesystem.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, 'v1')


def _publish_test_model_s3(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Publishes the versioned model to S3.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_s3()
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_S3, 'v1')


def test_init_loads_model_local(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that the object loads the saved model from the local filesystem.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _publish_test_model_local(dataset, model, training_config)
    model_path = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, 'v1')
    versioned_model = VersionedModel(model_path)
    for attribute in EXPECTED_ATTRIBUTES:
        assert hasattr(versioned_model, attribute)


@pytest.mark.awstest
def test_init_loads_model_s3(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that the object loads the saved model from S3.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _publish_test_model_s3(dataset, model, training_config)
    model_path = os.path.join(TEST_MODEL_PUBLICATION_PATH_S3, 'v1')
    versioned_model = VersionedModel(model_path)
    for attribute in EXPECTED_ATTRIBUTES:
        assert hasattr(versioned_model, attribute)


def test_versioned_models_from_same_files_are_equal(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that two versioned models loaded from the same files are considered
    equal in comparisons.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _publish_test_model_local(dataset, model, training_config)
    model_path = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, 'v1')
    versioned_model1 = VersionedModel(model_path)
    versioned_model2 = VersionedModel(model_path)
    assert versioned_model1 == versioned_model2


def test_hashcode_is_hash_of_md5_digest(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that the hashcode of the model object is the hash of the loaded MD5
    digest.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _publish_test_model_local(dataset, model, training_config)
    model_path = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, 'v1')
    versioned_model = VersionedModel(model_path)
    assert hash(versioned_model) == hash(versioned_model.md5)
