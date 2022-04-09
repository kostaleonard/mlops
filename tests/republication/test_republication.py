"""Tests republication.py."""

import os
import shutil
import pytest
from s3fs import S3FileSystem
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.errors import PublicationPathAlreadyExistsError
from tests.dataset.test_versioned_dataset import _publish_test_dataset_local, \
    _publish_test_dataset_s3, TEST_PUBLICATION_PATH_LOCAL, \
    TEST_PUBLICATION_PATH_S3

TEST_REPUBLICATION_PATH_LOCAL = '/tmp/test_republication/datasets'
TEST_REPUBLICATION_PATH_S3 = 's3://kosta-mlops/test_republication/datasets'


def _remove_republication_directories_local() -> None:
    """Removes the republication paths from the local filesystem."""
    try:
        shutil.rmtree(TEST_REPUBLICATION_PATH_LOCAL)
    except FileNotFoundError:
        pass


def _remove_republication_directories_s3() -> None:
    """Removes the republication paths from S3."""
    fs = S3FileSystem()
    try:
        fs.rm(TEST_REPUBLICATION_PATH_S3, recursive=True)
    except FileNotFoundError:
        pass


def test_republish_copies_files_local_to_local() -> None:
    """Tests that republish copies files from local to local."""
    _remove_republication_directories_local()
    _publish_test_dataset_local()
    dataset = VersionedDataset(os.path.join(TEST_PUBLICATION_PATH_LOCAL, 'v1'))
    dataset.republish(TEST_REPUBLICATION_PATH_LOCAL)
    republication_path = os.path.join(TEST_REPUBLICATION_PATH_LOCAL, 'v1')
    assert os.path.exists(republication_path)


@pytest.mark.mockedawstest
def test_republish_copies_files_s3_to_local(mocked_s3: None) -> None:
    """Tests that republish copies files from S3 to local.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    _remove_republication_directories_local()
    _publish_test_dataset_s3()
    dataset = VersionedDataset(os.path.join(TEST_PUBLICATION_PATH_S3, 'v1'))
    dataset.republish(TEST_REPUBLICATION_PATH_LOCAL)
    republication_path = os.path.join(TEST_REPUBLICATION_PATH_LOCAL, 'v1')
    assert os.path.exists(republication_path)


@pytest.mark.mockedawstest
def test_republish_copies_files_local_to_s3(mocked_s3: None) -> None:
    """Tests that republish copies files from local to S3.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    _remove_republication_directories_s3()
    _publish_test_dataset_local()
    dataset = VersionedDataset(os.path.join(TEST_PUBLICATION_PATH_LOCAL, 'v1'))
    dataset.republish(TEST_REPUBLICATION_PATH_S3)
    republication_path = os.path.join(TEST_REPUBLICATION_PATH_S3, 'v1')
    fs = S3FileSystem()
    assert fs.exists(republication_path)


@pytest.mark.mockedawstest
def test_republish_copies_files_s3_to_s3(mocked_s3: None) -> None:
    """Tests that republish copies files from S3 to S3.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    _remove_republication_directories_s3()
    _publish_test_dataset_s3()
    dataset = VersionedDataset(os.path.join(TEST_PUBLICATION_PATH_S3, 'v1'))
    dataset.republish(TEST_REPUBLICATION_PATH_S3)
    republication_path = os.path.join(TEST_REPUBLICATION_PATH_S3, 'v1')
    fs = S3FileSystem()
    assert fs.exists(republication_path)


def test_republish_raises_publication_path_exists_error() -> None:
    """Tests that republish raises a PublicationPathAlreadyExistsError when the
    publication path already exists."""
    _remove_republication_directories_local()
    _publish_test_dataset_local()
    dataset = VersionedDataset(os.path.join(TEST_PUBLICATION_PATH_LOCAL, 'v1'))
    dataset.republish(TEST_REPUBLICATION_PATH_LOCAL)
    with pytest.raises(PublicationPathAlreadyExistsError):
        dataset.republish(TEST_REPUBLICATION_PATH_LOCAL)
