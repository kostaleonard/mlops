"""Tests versioned_model_builder.py."""
# pylint: disable=redefined-outer-name,no-name-in-module

import os
import shutil
from datetime import datetime
import json
import time
import dateutil
import numpy as np
import pytest
from s3fs import S3FileSystem
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.model.training_config import TrainingConfig
from mlops.errors import PublicationPathAlreadyExistsError

TEST_MODEL_PUBLICATION_PATH_LOCAL = "/tmp/test_versioned_model_builder/model"
TEST_MODEL_PUBLICATION_PATH_S3 = (
    "s3://kosta-mlops/" "test_versioned_model_builder/model"
)


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


def test_publish_appends_explicit_version(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that publish appends the version string to the path.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = "v2"
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)
    expected_filename = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, version
    )
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)


def test_publish_appends_version_timestamp(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
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
    publication_time = dateutil.parser.parse(dirname)
    assert start < publication_time < end


def test_publish_local_path_creates_expected_files(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that publish on a local path creates the expected
    files/directories on the local filesystem.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = "v2"
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)
    assert len(os.listdir(TEST_MODEL_PUBLICATION_PATH_LOCAL)) == 1
    assert os.listdir(TEST_MODEL_PUBLICATION_PATH_LOCAL)[0] == version
    publication_dir = os.path.join(TEST_MODEL_PUBLICATION_PATH_LOCAL, version)
    assert set(os.listdir(publication_dir)) == {"model.h5", "meta.json"}


@pytest.mark.mockedawstest
def test_publish_s3_path_creates_expected_files(
    dataset: VersionedDataset,
    model: Model,
    training_config: TrainingConfig,
    mocked_s3: None,
) -> None:
    """Tests that publish on an S3 path creates the expected files/directories
    on S3.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_s3()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = "v2"
    builder.publish(TEST_MODEL_PUBLICATION_PATH_S3, version=version)
    fs = S3FileSystem()
    expected_filename = os.path.join(TEST_MODEL_PUBLICATION_PATH_S3, version)
    # Remove 's3://' from latter paths.
    assert set(fs.ls(expected_filename)) == {
        os.path.join(expected_filename, "model.h5")[5:],
        os.path.join(expected_filename, "meta.json")[5:],
    }
    assert fs.isdir(expected_filename)


def test_publish_local_path_raises_path_already_exists_error(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that publish on a local path that already exists raises a
    PublicationPathAlreadyExistsError.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = "v2"
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)
    with pytest.raises(PublicationPathAlreadyExistsError):
        builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)


@pytest.mark.mockedawstest
def test_publish_s3_path_raises_path_already_exists_error(
    dataset: VersionedDataset,
    model: Model,
    training_config: TrainingConfig,
    mocked_s3: None,
) -> None:
    """Tests that publish on an S3 path that already exists raises a
    PublicationPathAlreadyExistsError.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_s3()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = "v2"
    builder.publish(TEST_MODEL_PUBLICATION_PATH_S3, version=version)
    with pytest.raises(PublicationPathAlreadyExistsError):
        builder.publish(TEST_MODEL_PUBLICATION_PATH_S3, version=version)


def test_publish_includes_expected_metadata(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that publish creates a file meta.json with the expected
    metadata.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = "v2"
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)
    meta_path = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, version, "meta.json"
    )
    with open(meta_path, "r", encoding="utf-8") as infile:
        contents = json.loads(infile.read())
    assert set(contents.keys()) == {
        "name",
        "version",
        "hash",
        "dataset",
        "history",
        "train_args",
        "created_at",
        "tags",
    }


def test_publish_timestamps_match(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
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
    meta_path = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, dirname, "meta.json"
    )
    with open(meta_path, "r", encoding="utf-8") as infile:
        contents = json.loads(infile.read())
    version_time = contents["version"]
    created_at_time = contents["created_at"]
    assert dirname == version_time == created_at_time


def test_publish_local_with_trailing_slash(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that publishing to a local path with a trailing slash works
    properly.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = "v2"
    # One trailing slash.
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL + "/", version=version)
    expected_filename = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, version
    )
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)
    _remove_test_directories_local()
    # Many trailing slashes.
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL + "///", version=version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)


@pytest.mark.mockedawstest
def test_publish_s3_with_trailing_slash(
    dataset: VersionedDataset,
    model: Model,
    training_config: TrainingConfig,
    mocked_s3: None,
) -> None:
    """Tests that publishing to an S3 path with a trailing slash works
    properly.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_s3()
    builder = VersionedModelBuilder(dataset, model, training_config)
    version = "v2"
    # One trailing slash.
    builder.publish(TEST_MODEL_PUBLICATION_PATH_S3 + "/", version=version)
    fs = S3FileSystem()
    expected_filename = os.path.join(TEST_MODEL_PUBLICATION_PATH_S3, version)
    assert fs.exists(expected_filename)
    assert fs.isdir(expected_filename)
    # Many trailing slashes.
    _remove_test_directories_s3()
    builder.publish(TEST_MODEL_PUBLICATION_PATH_S3 + "///", version=version)
    fs = S3FileSystem()
    expected_filename = os.path.join(TEST_MODEL_PUBLICATION_PATH_S3, version)
    assert fs.exists(expected_filename)
    assert fs.isdir(expected_filename)


def test_same_models_have_same_hashes(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that the hash values from two models that have identical files are
    the same.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v1")
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v2")
    meta_path1 = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v1", "meta.json"
    )
    meta_path2 = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v2", "meta.json"
    )
    with open(meta_path1, "r", encoding="utf-8") as infile:
        contents1 = json.loads(infile.read())
    with open(meta_path2, "r", encoding="utf-8") as infile:
        contents2 = json.loads(infile.read())
    assert contents1["created_at"] != contents2["created_at"]
    assert contents1["hash"] == contents2["hash"]


@pytest.mark.slowtest
def test_same_models_have_same_hashes_different_timestamps(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that the hash values from two models that have identical files are
    the same even when they have different timestamps. In some libraries, e.g.,
    h5py (older versions), timestamp data is stored at second-level
    granularity.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v1")
    time.sleep(2)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v2")
    meta_path1 = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v1", "meta.json"
    )
    meta_path2 = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v2", "meta.json"
    )
    with open(meta_path1, "r", encoding="utf-8") as infile:
        contents1 = json.loads(infile.read())
    with open(meta_path2, "r", encoding="utf-8") as infile:
        contents2 = json.loads(infile.read())
    assert contents1["created_at"] != contents2["created_at"]
    assert contents1["hash"] == contents2["hash"]


def test_different_models_have_different_hashes(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that the hash values from two models that have different files are
    different.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder1 = VersionedModelBuilder(dataset, model, training_config)
    builder1.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v1")
    # model2 has an extra layer.
    model2 = Sequential(
        [
            Dense(5, input_shape=dataset.X_train.shape[1:]),
            Dense(dataset.y_train.shape[1]),
        ]
    )
    model2.compile("adam", loss="mse")
    train_args = {"epochs": 1, "batch_size": 2}
    history = model2.fit(x=dataset.X_train, y=dataset.y_train, **train_args)
    training_config2 = TrainingConfig(history, train_args)
    builder2 = VersionedModelBuilder(dataset, model2, training_config2)
    builder2.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v2")
    meta_path1 = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v1", "meta.json"
    )
    meta_path2 = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v2", "meta.json"
    )
    with open(meta_path1, "r", encoding="utf-8") as infile:
        contents1 = json.loads(infile.read())
    with open(meta_path2, "r", encoding="utf-8") as infile:
        contents2 = json.loads(infile.read())
    assert contents1["created_at"] != contents2["created_at"]
    assert contents1["hash"] != contents2["hash"]


@pytest.mark.mockedawstest
def test_publish_local_and_s3_create_same_model(
    dataset: VersionedDataset,
    model: Model,
    training_config: TrainingConfig,
    mocked_s3: None,
) -> None:
    """Tests that publishing locally or remotely on S3 produces the same model.
    Verifies identity by comparing model hashes.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _remove_test_directories_s3()
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v1")
    builder.publish(TEST_MODEL_PUBLICATION_PATH_S3, version="v1")
    meta_path1 = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v1", "meta.json"
    )
    meta_path2 = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_S3, "v1", "meta.json"
    )
    with open(meta_path1, "r", encoding="utf-8") as infile:
        contents1 = json.loads(infile.read())
    fs = S3FileSystem()
    with fs.open(meta_path2, "r", encoding="utf-8") as infile:
        contents2 = json.loads(infile.read())
    assert contents1["created_at"] != contents2["created_at"]
    assert contents1["hash"] == contents2["hash"]


def test_metadata_includes_training_history(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that the metadata includes the training history dictionary, and
    that it is consistent with the training results.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v1")
    meta_path = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v1", "meta.json"
    )
    with open(meta_path, "r", encoding="utf-8") as infile:
        contents = json.loads(infile.read())
    meta_history = contents["history"]
    assert meta_history == training_config.history.history


def test_metadata_includes_training_args(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that the metadata includes any kwargs supplied to the training
    function.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v1")
    meta_path = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v1", "meta.json"
    )
    with open(meta_path, "r", encoding="utf-8") as infile:
        contents = json.loads(infile.read())
    meta_args = contents["train_args"]
    assert meta_args == training_config.train_args


def test_metadata_includes_dataset_link(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that the metadata includes a link to the dataset.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v1")
    meta_path = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v1", "meta.json"
    )
    with open(meta_path, "r", encoding="utf-8") as infile:
        contents = json.loads(infile.read())
    link = contents["dataset"]
    assert link == dataset.path


def test_metadata_includes_expected_tags(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that the metadata includes the expected tags.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    tags = ["hello", "world"]
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v1", tags=tags)
    meta_path = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v1", "meta.json"
    )
    with open(meta_path, "r", encoding="utf-8") as infile:
        contents = json.loads(infile.read())
    meta_tags = contents["tags"]
    assert meta_tags == tags


def test_published_model_performance_matches_trained_model(
    dataset: VersionedDataset, model: Model, training_config: TrainingConfig
) -> None:
    """Tests that the published model has the same performance as the model
    supplied during object instantiation.

    :param dataset: The versioned dataset.
    :param model: The model.
    :param training_config: The training configuration.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version="v1")
    saved_model_path = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, "v1", "model.h5"
    )
    loaded_model = load_model(saved_model_path)
    val_err = model.evaluate(x=dataset.X_val, y=dataset.y_val)
    loaded_val_err = loaded_model.evaluate(x=dataset.X_val, y=dataset.y_val)
    assert np.isclose(val_err, loaded_val_err)


def test_publish_without_training_config(
    dataset: VersionedDataset, model: Model
) -> None:
    """Tests that a model can be published without a training config.

    :param dataset: The versioned dataset.
    :param model: The model.
    """
    _remove_test_directories_local()
    builder = VersionedModelBuilder(dataset, model)
    version = "v2"
    builder.publish(TEST_MODEL_PUBLICATION_PATH_LOCAL, version=version)
    expected_filename = os.path.join(
        TEST_MODEL_PUBLICATION_PATH_LOCAL, version
    )
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)
