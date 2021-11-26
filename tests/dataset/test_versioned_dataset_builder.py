"""Tests versioned_dataset_builder.py."""

import os
import shutil
from pathlib import Path
from datetime import datetime
import pytest
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder, \
    STRATEGY_COPY, STRATEGY_LINK
from mlops.errors import PublicationPathAlreadyExistsError
from tests.dataset.preset_data_processor import PresetDataProcessor

TEST_DATASET_PATH_LOCAL = '/tmp/test_versioned_dataset_builder/dataset'
TEST_PUBLICATION_PATH_LOCAL = '/tmp/test_versioned_dataset_builder/publish'
TEST_DATASET_FILENAMES = ['file0.txt', 'file1.txt', 'file2.txt']


def _remove_test_directories_local() -> None:
    """Removes the local test directories."""
    for dirname in (TEST_DATASET_PATH_LOCAL, TEST_PUBLICATION_PATH_LOCAL):
        try:
            shutil.rmtree(dirname)
        except FileNotFoundError:
            pass


def _create_test_dataset_local() -> None:
    """Creates a preset raw dataset at the local test dataset path."""
    path = Path(TEST_DATASET_PATH_LOCAL)
    path.mkdir(parents=True)
    for filename in TEST_DATASET_FILENAMES:
        with open(os.path.join(TEST_DATASET_PATH_LOCAL, filename),
                  'w',
                  encoding='utf-8') as outfile:
            outfile.write(filename)


def test_publish_appends_explicit_version() -> None:
    """Tests that publish appends the version string to the path."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version)
    expected_filename = os.path.join(TEST_DATASET_PATH_LOCAL, version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)


def test_publish_appends_version_timestamp() -> None:
    """Tests that publish appends the timestamp to the path when no version is
    given."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    start = datetime.now()
    builder.publish(TEST_PUBLICATION_PATH_LOCAL)
    end = datetime.now()
    assert len(os.listdir(TEST_PUBLICATION_PATH_LOCAL)) == 1
    dirname = os.listdir(TEST_PUBLICATION_PATH_LOCAL)[0]
    publication_time = datetime.fromisoformat(dirname)
    assert start < publication_time < end


def test_publish_local_path_creates_expected_files() -> None:
    """Tests that publish on a local path creates the expected
    files/directories on the local filesystem."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version)
    assert len(os.listdir(TEST_PUBLICATION_PATH_LOCAL)) == 1
    assert os.listdir(TEST_PUBLICATION_PATH_LOCAL)[0] == version
    publication_dir = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version)
    expected_features = {'X_train', 'X_val', 'X_test'}
    expected_labels = {'y_train', 'y_val', 'y_test'}
    assert set(os.listdir(publication_dir)) == expected_features.union(
        expected_labels).union(
        {'data_processor.pkl', 'meta.json', 'raw'})


def test_publish_s3_path_creates_expected_files() -> None:
    """Tests that publish on an S3 path creates the expected files/directories
    on the local filesystem."""
    # TODO
    assert False


def test_publish_local_path_raises_path_already_exists_error() -> None:
    """Tests that publish on a local path that already exists raises a
    PublicationPathAlreadyExistsError."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version)
    with pytest.raises(PublicationPathAlreadyExistsError):
        builder.publish(TEST_PUBLICATION_PATH_LOCAL, version)


def test_publish_s3_path_raises_path_already_exists_error() -> None:
    """Tests that publish on an S3 path that already exists raises a
    PublicationPathAlreadyExistsError."""
    # TODO
    assert False


def test_publish_copies_raw_dataset() -> None:
    """Tests that publish copies the entire raw dataset when the copy strategy
    is STRATEGY_COPY."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL,
                    version,
                    dataset_copy_strategy=STRATEGY_COPY)
    raw_dataset_dir = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version, 'raw')
    assert set(os.listdir(raw_dataset_dir)) == set(TEST_DATASET_FILENAMES)


def test_publish_includes_raw_dataset_link() -> None:
    """Tests that publish includes a link to the raw dataset when the copy
    strategy is STRATEGY_LINK."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL,
                    version,
                    dataset_copy_strategy=STRATEGY_LINK)
    raw_dataset_dir = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version, 'raw')
    link_filename = 'link.txt'
    assert set(os.listdir(raw_dataset_dir)) == {link_filename}
    with open(os.path.join(raw_dataset_dir, link_filename),
              'r',
              encoding='uft-8') as infile:
        assert infile.read() == TEST_DATASET_PATH_LOCAL


def test_publish_includes_expected_metadata() -> None:
    """Tests that publish creates a file meta.json with the expected
    metadata."""
    # TODO
    assert False


def test_publish_timestamps_match() -> None:
    """Tests that all 3 timestamps match if no version string is supplied:
    metadata.json's version and created_at fields, and the final directory
    of the published path."""
    # TODO
    assert False


def test_publish_accepts_path_with_trailing_slash() -> None:
    """Tests that publish accepts a path with (potentially many) trailing
    slashes and creates the files as if the trailing slashes were absent."""
    # TODO
    assert False


def test_same_datasets_have_same_hashes() -> None:
    """Tests that the hash values from two datasets that have identical files
    are the same."""
    # TODO
    assert False


def test_different_datasets_have_different_hashes() -> None:
    """Tests that the hash values from two datasets that have different files
    are different."""
    # TODO
    assert False
