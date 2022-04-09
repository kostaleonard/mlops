"""Tests versioned_artifact.py."""

import json
import pytest
from s3fs import S3FileSystem
from mlops.dataset.versioned_dataset import VersionedDataset


def test_update_metadata_overwrites_local_file(
        dataset: VersionedDataset) -> None:
    """Tests that update_metadata overwrites the previous metadata file on the
    local filesystem.

    :param dataset: The versioned dataset.
    """
    with open(dataset.metadata_path, 'r', encoding='utf-8') as infile:
        original_metadata = json.loads(infile.read())
    new_name = 'new_dataset_name'
    assert original_metadata['name'] != new_name
    dataset.update_metadata({'name': new_name})
    with open(dataset.metadata_path, 'r', encoding='utf-8') as infile:
        new_metadata = json.loads(infile.read())
    assert new_metadata['name'] == new_name


@pytest.mark.mockedawstest
def test_update_metadata_overwrites_s3_file(
        dataset_s3: VersionedDataset,
        mocked_s3: None) -> None:
    """Tests that update_metadata overwrites the previous metadata file on the
    S3 filesystem.

    :param dataset_s3: The versioned dataset loaded from S3.
    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    fs = S3FileSystem()
    with fs.open(dataset_s3.metadata_path, 'r', encoding='utf-8') as infile:
        original_metadata = json.loads(infile.read())
    new_name = 'new_dataset_name'
    assert original_metadata['name'] != new_name
    dataset_s3.update_metadata({'name': new_name})
    with fs.open(dataset_s3.metadata_path, 'r', encoding='utf-8') as infile:
        new_metadata = json.loads(infile.read())
    assert new_metadata['name'] == new_name


def test_update_metadata_reflected_in_loaded_artifact(
        dataset: VersionedDataset) -> None:
    """Tests that changes made in update_metadata are reflected when the object
    is loaded.

    :param dataset: The versioned dataset.
    """
    new_name = 'new_dataset_name'
    assert dataset.name != new_name
    dataset.update_metadata({'name': new_name})
    new_dataset = VersionedDataset(dataset.path)
    assert new_dataset.name == new_name
