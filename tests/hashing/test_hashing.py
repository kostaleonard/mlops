"""Tests hashing.py."""

import os
import pytest
from mlops.hashing import hashing
from tests.dataset.test_versioned_dataset_builder import \
    _remove_test_directories_local, _create_test_dataset_local, \
    _remove_test_directories_s3, _create_test_dataset_s3, \
    TEST_DATASET_PATH_LOCAL, TEST_DATASET_FILENAMES, \
    TEST_DATASET_PATH_S3


def test_hash_local_is_reproducible() -> None:
    """Tests that hashing of local files is reproducible."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    files_to_hash_forward = [os.path.join(TEST_DATASET_PATH_LOCAL, filename)
                             for filename in TEST_DATASET_FILENAMES]
    files_to_hash_reverse = [os.path.join(TEST_DATASET_PATH_LOCAL, filename)
                             for filename in TEST_DATASET_FILENAMES[::-1]]
    hash_forward = hashing.get_hash_local(files_to_hash_forward)
    hash_reverse = hashing.get_hash_local(files_to_hash_reverse)
    assert hash_forward
    assert hash_reverse
    assert hash_forward == hash_reverse


@pytest.mark.mockedawstest
def test_hash_s3_is_reproducible(mocked_s3: None) -> None:
    """Tests that hashing of remote files is reproducible.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    _remove_test_directories_s3()
    _create_test_dataset_s3()
    files_to_hash_forward = [os.path.join(TEST_DATASET_PATH_S3, filename)
                             for filename in TEST_DATASET_FILENAMES]
    files_to_hash_reverse = [os.path.join(TEST_DATASET_PATH_S3, filename)
                             for filename in TEST_DATASET_FILENAMES[::-1]]
    hash_forward = hashing.get_hash_s3(files_to_hash_forward)
    hash_reverse = hashing.get_hash_s3(files_to_hash_reverse)
    assert hash_forward
    assert hash_reverse
    assert hash_forward == hash_reverse


def test_hash_different_if_file_content_different() -> None:
    """Tests that the output of hashing is different if the content of files is
    different."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    files_to_hash = [os.path.join(TEST_DATASET_PATH_LOCAL, filename)
                     for filename in TEST_DATASET_FILENAMES]
    hash_before = hashing.get_hash_local(files_to_hash)
    with open(os.path.join(TEST_DATASET_PATH_LOCAL, TEST_DATASET_FILENAMES[0]),
              'w',
              encoding='utf-8') as outfile:
        outfile.write('new data')
    hash_after = hashing.get_hash_local(files_to_hash)
    assert hash_before != hash_after
