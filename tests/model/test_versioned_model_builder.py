"""Tests versioned_model_builder.py."""

import os
import shutil
from datetime import datetime
import pytest
import json
from s3fs import S3FileSystem
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.model.training_config import TrainingConfig
from mlops.errors import PublicationPathAlreadyExistsError
from tests.dataset.test_versioned_dataset import _publish_test_dataset_local, \
    TEST_PUBLICATION_PATH_LOCAL as TEST_DATASET_PUBLICATION_PATH_LOCAL

TEST_MODEL_PUBLICATION_PATH_LOCAL = '/tmp/test_versioned_model_builder/model'
TEST_MODEL_PUBLICATION_PATH_S3 = ('s3://kosta-mlops/'
                                  'test_versioned_model_builder/model')


def _remove_test_directories_local() -> None:
    """Removes the local test directories."""
    try:
        shutil.rmtree(TEST_MODEL_PUBLICATION_PATH_LOCAL)
    except FileNotFoundError:
        pass


def _remove_test_directories_s3() -> None:
    """Removes the S3 test directories."""
    fs = S3FileSystem()
    try:
        fs.rm(TEST_MODEL_PUBLICATION_PATH_S3, recursive=True)
    except FileNotFoundError:
        pass


@pytest.fixture
def dataset() -> VersionedDataset:
    """Returns the versioned dataset fixture for testing.

    :return: The versioned dataset fixture.
    """
    _publish_test_dataset_local()
    return VersionedDataset(os.path.join(TEST_DATASET_PUBLICATION_PATH_LOCAL,
                                         'v1'))


@pytest.fixture
def model(dataset: VersionedDataset) -> Model:
    """Returns the model fixture for testing.

    :param dataset: The versioned dataset.
    :return: The model fixture.
    """
    mod = Sequential([Dense(dataset.y_train.shape[1],
                            input_shape=dataset.X_train.shape[1:])])
    mod.compile('adam', loss='mse')
    return mod


@pytest.fixture
def training_config(dataset: VersionedDataset,
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


def test_publish_appends_explicit_version(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that publish appends the version string to the path.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = 'v2'
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)
    expected_filename = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)


def test_publish_appends_version_timestamp(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that publish appends the timestamp to the path when no version is
    given.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    start = datetime.now()
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL)
    end = datetime.now()
    assert len(os.listdir(TEST_MODEL_PUBLICATION_PATH_LOCAL)) == 1
    dirname = os.listdir(TEST_MODEL_PUBLICATION_PATH_LOCAL)[0]
    publication_time = datetime.fromisoformat(dirname)
    assert start < publication_time < end


def test_publish_local_path_creates_expected_files(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that publish on a local path creates the expected
    files/directories on the local filesystem.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = 'v2'
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)
    assert len(os.listdir(TEST_MODEL_PUBLICATION_PATH_LOCAL)) == 1
    assert os.listdir(TEST_MODEL_PUBLICATION_PATH_LOCAL)[0] == version
    publication_dir = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, version)
    assert set(os.listdir(publication_dir)) == {'model.h5', 'meta.json'}


@pytest.mark.awstest
def test_publish_s3_path_creates_expected_files(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that publish on an S3 path creates the expected files/directories
    on S3.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_s3()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = 'v2'
    builder.publish(TEST_MODEL_PUBLICATION_PATH_S3, version=version)
    fs = S3FileSystem()
    expected_filename = os.path.join(TEST_MODEL_PUBLICATION_PATH_S3, version)
    # Remove 's3://' from latter paths.
    assert set(fs.ls(expected_filename)) == {
        os.path.join(expected_filename, 'model.h5')[5:],
        os.path.join(expected_filename, 'meta.json')[5:]}
    assert fs.isdir(expected_filename)


def test_publish_local_path_raises_path_already_exists_error(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that publish on a local path that already exists raises a
    PublicationPathAlreadyExistsError.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = 'v2'
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)
    with pytest.raises(PublicationPathAlreadyExistsError):
        builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)


@pytest.mark.awstest
def test_publish_s3_path_raises_path_already_exists_error(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that publish on an S3 path that already exists raises a
    PublicationPathAlreadyExistsError.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_s3()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = 'v2'
    builder.publish(TEST_MODEL_PUBLICATION_PATH_S3, version=version)
    with pytest.raises(PublicationPathAlreadyExistsError):
        builder.publish(TEST_MODEL_PUBLICATION_PATH_S3, version=version)


def test_publish_includes_expected_metadata(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that publish creates a file meta.json with the expected
    metadata.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = 'v2'
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)
    meta_path = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, version,
                             'meta.json')
    with open(meta_path, 'r', encoding='utf-8') as infile:
        contents = json.loads(infile.read())
    assert set(contents.keys()) == {
        'version',
        'hash',
        'dataset',
        'history',
        'train_args',
        'created_at',
        'tags'}


def test_publish_timestamps_match(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that all 3 timestamps match if no version string is supplied:
    metadata.json's version and created_at fields, and the final directory
    of the published path.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL)
    assert len(os.listdir(TEST_MODEL_PUBLICATION_PATH_LOCAL)) == 1
    dirname = os.listdir(TEST_MODEL_PUBLICATION_PATH_LOCAL)[0]
    meta_path = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, dirname,
                             'meta.json')
    with open(meta_path, 'r', encoding='utf-8') as infile:
        contents = json.loads(infile.read())
    version_time = contents['version']
    created_at_time = contents['created_at']
    assert dirname == version_time == created_at_time


def test_publish_accepts_path_with_trailing_slash(
        dataset: VersionedDataset,
        model: Model,
        training_config: TrainingConfig) -> None:
    """Tests that publish accepts a path with (potentially many) trailing
    slashes and creates the files as if the trailing slashes were absent.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = 'v2'
    # One trailing slash.
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL + '/', version)
    expected_filename = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)
    _remove_test_directories_local()
    # Many trailing slashes.
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL + '///', version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)


def test_publish_local_with_trailing_slash() -> None:
    """Tests that publishing to a local path with a trailing slash works
    properly."""
    # TODO
    assert False


@pytest.mark.awstest
def test_publish_s3_with_trailing_slash() -> None:
    """Tests that publishing to an S3 path with a trailing slash works
    properly."""
    # TODO
    assert False


def test_same_models_have_same_hashes() -> None:
    """Tests that the hash values from two models that have identical files are
    the same."""
    # TODO
    assert False


def test_different_models_have_different_hashes() -> None:
    """Tests that the hash values from two models that have different files are
    different."""
    # TODO
    assert False


@pytest.mark.awstest
def test_publish_local_and_s3_create_same_model() -> None:
    """Tests that publishing locally or remotely on S3 produces the same model.
    Verifies identity by comparing model hashes."""
    # TODO
    assert False


def test_metadata_includes_training_history() -> None:
    """Tests that the metadata includes the training history dictionary, and
    that it is consistent with the training results."""
    # TODO
    assert False


def test_metadata_includes_training_args() -> None:
    """Tests that the metadata includes any kwargs supplied to the training
    function."""
    # TODO
    assert False


def test_metadata_includes_dataset_link() -> None:
    """Tests that the metadata includes a link to the dataset."""
    # TODO
    assert False


def test_metadata_includes_expected_tags() -> None:
    """Tests that the metadata includes the expected tags."""
    # TODO
    assert False


def test_published_model_performance_matches_trained_model() -> None:
    """Tests that the published model has the same performance as the model
    supplied during object instantiation."""
    # TODO
    assert False
