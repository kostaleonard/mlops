"""Tests versioned_dataset_builder.py."""

import os
import shutil
from pathlib import Path
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder
from tests.dataset.preset_data_processor import PresetDataProcessor

TEST_DATASET_PATH_LOCAL = '/tmp/test_versioned_dataset_builder/dataset'
TEST_PUBLICATION_PATH_LOCAL = '/tmp/test_versioned_dataset_builder/publish'
TEST_DATASET_FILENAMES = ['file0.txt', 'file1.txt', 'file2.txt']


def _remove_test_directories() -> None:
    """Removes the test directories."""
    for dirname in (TEST_DATASET_PATH_LOCAL, TEST_PUBLICATION_PATH_LOCAL):
        try:
            shutil.rmtree(dirname)
        except FileNotFoundError:
            pass


def _create_test_dataset() -> None:
    """Creates a preset raw dataset at the test dataset path."""
    path = Path(TEST_DATASET_PATH_LOCAL)
    path.mkdir(parents=True)
    for filename in TEST_DATASET_FILENAMES:
        with open(os.path.join(TEST_DATASET_PATH_LOCAL, filename),
                  'w',
                  encoding='utf-8') as outfile:
            outfile.write(filename)


def test_publish_appends_explicit_version() -> None:
    """Tests that publish appends the version string to the path."""
    _remove_test_directories()
    _create_test_dataset()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version)
    expected_filename = os.path.join(TEST_DATASET_PATH_LOCAL, version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)
    _remove_test_directories()


def test_publish_appends_version_timestamp() -> None:
    """Tests that publish appends the timestamp to the path when no version is
    given."""
    # TODO
    assert False


def test_publish_local_path_creates_expected_files() -> None:
    """Tests that publish on a local path creates the expected
    files/directories on the local filesystem."""
    # TODO
    assert False


def test_publish_s3_path_creates_expected_files() -> None:
    """Tests that publish on an S3 path creates the expected files/directories
    on the local filesystem."""
    # TODO
    assert False


def test_publish_local_path_raises_path_already_exists_error() -> None:
    """Tests that publish on a local path that already exists raises a
    PublicationPathAlreadyExistsError."""
    # TODO
    assert False


def test_publish_s3_path_raises_path_already_exists_error() -> None:
    """Tests that publish on an S3 path that already exists raises a
    PublicationPathAlreadyExistsError."""
    # TODO
    assert False


def test_publish_copies_raw_dataset() -> None:
    """Tests that publish copies the entire raw dataset when the copy strategy
    is STRATEGY_COPY."""
    # TODO
    assert False


def test_publish_includes_raw_dataset_link() -> None:
    """Tests that publish includes a link to the raw dataset when the copy
    strategy is STRATEGY_LINK."""
    # TODO
    assert False


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
