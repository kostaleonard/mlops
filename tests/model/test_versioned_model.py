"""Tests versioned_model.py."""
# pylint: disable=no-name-in-module

import os
import shutil
import pytest
from tensorflow.keras.models import Model
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.model.training_config import TrainingConfig
from mlops.model.versioned_model import VersionedModel
from tests.model.test_versioned_model_builder import \
    _remove_test_directories_local, _remove_test_directories_s3, \
    TEST_MODEL_PUBLICATION_PATH_LOCAL, TEST_MODEL_PUBLICATION_PATH_S3

EXPECTED_ATTRIBUTES = {'path', 'name', 'version', 'model', 'md5'}
TEST_REPUBLICATION_PATH_LOCAL = '/tmp/test_versioned_model/models'


def _remove_republication_directories_local() -> None:
    """Removes the republication paths from the local filesystem."""
    try:
        shutil.rmtree(TEST_REPUBLICATION_PATH_LOCAL)
    except FileNotFoundError:
        pass


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
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version='v1')


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
    builder.publish(TEST_MODEL_PUBLICATION_PATH_S3, version='v1')


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


def test_republish_creates_files(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that republish creates the expected files.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_republication_directories_local()
    _publish_test_model_local(dataset, model, training_config)
    model_path = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, 'v1')
    versioned_model = VersionedModel(model_path)
    versioned_model.republish(TEST_REPUBLICATION_PATH_LOCAL)
    republication_path = os.path.join(TEST_REPUBLICATION_PATH_LOCAL, 'v1')
    assert os.path.exists(republication_path)


def test_versioned_model_has_dataset_path(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that VersionedModel has a dataset path attribute.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_republication_directories_local()
    _publish_test_model_local(dataset, model, training_config)
    model_path = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, 'v1')
    versioned_model = VersionedModel(model_path)
    assert hasattr(versioned_model, 'dataset_path')
    loaded_dataset = VersionedDataset(versioned_model.dataset_path)
    assert loaded_dataset.md5 == dataset.md5
