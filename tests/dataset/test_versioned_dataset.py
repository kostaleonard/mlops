"""Tests versioned_dataset.py."""

import os
import pytest
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder
from mlops.dataset.versioned_dataset import VersionedDataset
from tests.dataset.test_versioned_dataset_builder import \
    _remove_test_directories_local, _create_test_dataset_local, \
    _remove_test_directories_s3, \
    TEST_DATASET_PATH_LOCAL, TEST_PUBLICATION_PATH_LOCAL, \
    TEST_PUBLICATION_PATH_S3
from tests.dataset.preset_data_processor import PresetDataProcessor

EXPECTED_ATTRIBUTES = {'X_train', 'X_val', 'X_test',
                       'y_train', 'y_val', 'y_test',
                       'name',
                       'version',
                       'md5',
                       'data_processor'}


def _publish_test_dataset_local() -> None:
    """Publishes the versioned dataset to the local filesystem."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version=version)


def _publish_test_dataset_s3() -> None:
    """Publishes the versioned dataset to S3."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    _remove_test_directories_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_S3, version=version)


def test_init_creates_expected_attributes_local() -> None:
    """Tests that init creates all feature/label tensors, the hash, and the
    data processor as attributes when the dataset is on the local filesystem."""
    _publish_test_dataset_local()
    dataset = VersionedDataset(os.path.join(TEST_PUBLICATION_PATH_LOCAL, 'v1'))
    for attribute in EXPECTED_ATTRIBUTES:
        assert hasattr(dataset, attribute)


@pytest.mark.awstest
def test_init_creates_expected_attributes_s3() -> None:
    """Tests that init creates all feature/label tensors, the hash, and the
    data processor as attributes when the dataset is on S3."""
    _publish_test_dataset_s3()
    dataset = VersionedDataset(os.path.join(TEST_PUBLICATION_PATH_S3, 'v1'))
    for attribute in EXPECTED_ATTRIBUTES:
        assert hasattr(dataset, attribute)


def test_versioned_datasets_from_same_files_are_equal() -> None:
    """Tests that two versioned datasets loaded from the same files are
    considered equal in comparisons."""
    _publish_test_dataset_local()
    dataset_path = os.path.join(TEST_PUBLICATION_PATH_LOCAL, 'v1')
    dataset1 = VersionedDataset(dataset_path)
    dataset2 = VersionedDataset(dataset_path)
    assert dataset1 == dataset2


def test_hashcode_is_hash_of_md5_digest() -> None:
    """Tests that the hashcode of the dataset object is the hash of the loaded
    MD5 digest."""
    _publish_test_dataset_local()
    dataset = VersionedDataset(os.path.join(TEST_PUBLICATION_PATH_LOCAL, 'v1'))
    assert hash(dataset) == hash(dataset.md5)
