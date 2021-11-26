"""Tests versioned_dataset_builder.py."""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json
from urllib.parse import urlparse
import pytest
import boto3
from botocore.exceptions import ClientError
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder, \
    STRATEGY_COPY, STRATEGY_LINK
from mlops.errors import PublicationPathAlreadyExistsError
from tests.dataset.preset_data_processor import PresetDataProcessor

TEST_DATASET_PATH_LOCAL = '/tmp/test_versioned_dataset_builder/raw_dataset'
TEST_PUBLICATION_PATH_LOCAL = '/tmp/test_versioned_dataset_builder/datasets'
TEST_DATASET_PATH_S3 = ('s3://kosta-mlops/test_versioned_dataset_builder/'
                        'raw_dataset')
TEST_PUBLICATION_PATH_S3 = ('s3://kosta-mlops/test_versioned_dataset_builder/'
                            'datasets')
TEST_DATASET_FILENAMES = ['file0.txt', 'file1.txt', 'file2.txt']


def _remove_test_directories_local() -> None:
    """Removes the local test directories."""
    for dirname in (TEST_DATASET_PATH_LOCAL, TEST_PUBLICATION_PATH_LOCAL):
        try:
            shutil.rmtree(dirname)
        except FileNotFoundError:
            pass


def _remove_test_directories_s3() -> None:
    """Removes the S3 test directories."""
    s3 = boto3.resource('s3')
    for dirname in (TEST_DATASET_PATH_S3, TEST_PUBLICATION_PATH_S3):
        parse_result = urlparse(dirname)
        bucket_name = parse_result.netloc
        # Remove leading slash
        prefix = parse_result.path[1:]
        bucket = s3.Bucket(bucket_name)
        bucket.objects.filter(Prefix=prefix).delete()


def _create_test_dataset_local() -> None:
    """Creates a preset raw dataset at the local test dataset path."""
    path = Path(TEST_DATASET_PATH_LOCAL)
    path.mkdir(parents=True)
    for filename in TEST_DATASET_FILENAMES:
        with open(os.path.join(TEST_DATASET_PATH_LOCAL, filename),
                  'w',
                  encoding='utf-8') as outfile:
            outfile.write(filename)


def _create_test_dataset_s3() -> None:
    """Creates a preset raw dataset at the S3 test dataset path."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    s3 = boto3.client('s3')
    parse_result = urlparse(TEST_DATASET_PATH_S3)
    bucket_name = parse_result.netloc
    # Remove leading slash
    prefix = parse_result.path[1:]
    for filename in TEST_DATASET_FILENAMES:
        local_path = os.path.join(TEST_DATASET_PATH_LOCAL, filename)
        s3_path = os.path.join(prefix, filename)
        try:
            _ = s3.upload_file(local_path,
                               bucket_name,
                               s3_path)
        except ClientError as exc:
            raise exc


def test_publish_appends_explicit_version() -> None:
    """Tests that publish appends the version string to the path."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version)
    expected_filename = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version)
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
    expected_features = {'X_train.npy', 'X_val.npy', 'X_test.npy'}
    expected_labels = {'y_train.npy', 'y_val.npy', 'y_test.npy'}
    assert set(os.listdir(publication_dir)) == expected_features.union(
        expected_labels).union(
        {'data_processor.pkl', 'meta.json', 'raw'})


def test_publish_s3_path_creates_expected_files() -> None:
    """Tests that publish on an S3 path creates the expected files/directories
    on S3."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    _remove_test_directories_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_S3, version)
    dirname = os.path.join(TEST_PUBLICATION_PATH_S3, version)
    parse_result = urlparse(dirname)
    bucket_name = parse_result.netloc
    # Remove leading slash
    prefix = parse_result.path[1:]
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    items = list(bucket.objects.filter(Prefix=prefix))
    item_keys = set(item.key for item in items)
    # Check for every key except raw.
    expected_features = {'X_train.npy', 'X_val.npy', 'X_test.npy'}
    expected_labels = {'y_train.npy', 'y_val.npy', 'y_test.npy'}
    expected_keys = expected_features.union(
        expected_labels).union(
        {'data_processor.pkl', 'meta.json'})
    expected_keys = {os.path.join(prefix, key) for key in expected_keys}
    assert expected_keys.intersection(item_keys) == expected_keys
    # Check for raw "directory" (flat filesystem).
    raw_directory_key = os.path.join(prefix, 'raw/')
    assert any(key.startswith(raw_directory_key) for key in item_keys)


def test_publish_from_raw_dataset_in_s3() -> None:
    """Tests that publish correctly reads the dataset path when the dataset is
    in S3."""
    _remove_test_directories_local()
    _remove_test_directories_s3()
    _create_test_dataset_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_S3, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version)
    raw_dataset_dir = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version, 'raw')
    assert set(os.listdir(raw_dataset_dir)) == set(TEST_DATASET_FILENAMES)


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
    _remove_test_directories_local()
    _create_test_dataset_local()
    _remove_test_directories_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_S3, version)
    with pytest.raises(PublicationPathAlreadyExistsError):
        builder.publish(TEST_PUBLICATION_PATH_S3, version)


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
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version)
    meta_path = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version, 'meta.json')
    with open(meta_path, 'r', encoding='utf-8') as infile:
        contents = json.loads(infile.read())
    assert set(contents.keys()) == {'version', 'hash', 'created_at', 'tags'}


def test_publish_timestamps_match() -> None:
    """Tests that all 3 timestamps match if no version string is supplied:
    metadata.json's version and created_at fields, and the final directory
    of the published path."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    builder.publish(TEST_PUBLICATION_PATH_LOCAL)
    assert len(os.listdir(TEST_PUBLICATION_PATH_LOCAL)) == 1
    dirname = os.listdir(TEST_PUBLICATION_PATH_LOCAL)[0]
    meta_path = os.path.join(TEST_PUBLICATION_PATH_LOCAL, dirname, 'meta.json')
    with open(meta_path, 'r', encoding='utf-8') as infile:
        contents = json.loads(infile.read())
    version_time = contents['version']
    created_at_time = contents['created_at']
    assert dirname == version_time == created_at_time


def test_publish_accepts_path_with_trailing_slash() -> None:
    """Tests that publish accepts a path with (potentially many) trailing
    slashes and creates the files as if the trailing slashes were absent."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    # One trailing slash.
    builder.publish(TEST_PUBLICATION_PATH_LOCAL + '/', version)
    expected_filename = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)
    _remove_test_directories_local()
    _create_test_dataset_local()
    # Many trailing slashes.
    builder.publish(TEST_PUBLICATION_PATH_LOCAL + '///', version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)


def test_same_datasets_have_same_hashes() -> None:
    """Tests that the hash values from two datasets that have identical files
    are the same."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, 'v1')
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, 'v2')
    meta_path1 = os.path.join(TEST_PUBLICATION_PATH_LOCAL,
                              'v1',
                              'meta.json')
    meta_path2 = os.path.join(TEST_PUBLICATION_PATH_LOCAL,
                              'v2',
                              'meta.json')
    with open(meta_path1, 'r', encoding='utf-8') as infile:
        contents1 = json.loads(infile.read())
    with open(meta_path2, 'r', encoding='utf-8') as infile:
        contents2 = json.loads(infile.read())
    assert contents1['created_at'] != contents2['created_at']
    assert contents1['hash'] == contents2['hash']


def test_different_datasets_have_different_hashes() -> None:
    """Tests that the hash values from two datasets that have different files
    are different."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, 'v1')
    # The second dataset is missing one of the raw dataset files.
    os.remove(os.path.join(TEST_DATASET_PATH_LOCAL, TEST_DATASET_FILENAMES[0]))
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, 'v2')
    meta_path1 = os.path.join(TEST_PUBLICATION_PATH_LOCAL,
                              'v1',
                              'meta.json')
    meta_path2 = os.path.join(TEST_PUBLICATION_PATH_LOCAL,
                              'v2',
                              'meta.json')
    with open(meta_path1, 'r', encoding='utf-8') as infile:
        contents1 = json.loads(infile.read())
    with open(meta_path2, 'r', encoding='utf-8') as infile:
        contents2 = json.loads(infile.read())
    assert contents1['created_at'] != contents2['created_at']
    assert contents1['hash'] != contents2['hash']


def test_publish_local_and_s3_create_same_dataset() -> None:
    """Tests that publishing locally or remotely on S3 produces the same
    dataset. Verifies identity by comparing dataset hashes."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    _remove_test_directories_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version)
    builder.publish(TEST_PUBLICATION_PATH_S3, version)
    meta_path1 = os.path.join(TEST_PUBLICATION_PATH_LOCAL,
                              version,
                              'meta.json')
    with open(meta_path1, 'r', encoding='utf-8') as infile:
        contents1 = json.loads(infile.read())
    s3 = boto3.client('s3')
    parse_result = urlparse(TEST_PUBLICATION_PATH_S3)
    bucket_name = parse_result.netloc
    # Remove leading slash
    prefix = parse_result.path[1:]
    contents2 = s3.get_object(Bucket=bucket_name,
                              Key=os.path.join(prefix, version, 'meta.json'))
    contents2 = json.loads(contents2.decode('utf-8'))
    assert contents1['created_at'] != contents2['created_at']
    assert contents1['hash'] == contents2['hash']


def test_hash_is_reproducible() -> None:
    """Tests that hashing of files is reproducible."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    files_to_hash_forward = [os.path.join(TEST_DATASET_PATH_LOCAL, filename)
                             for filename in TEST_DATASET_FILENAMES]
    files_to_hash_reverse = [os.path.join(TEST_DATASET_PATH_LOCAL, filename)
                             for filename in TEST_DATASET_FILENAMES[::-1]]
    hash_forward = VersionedDatasetBuilder._get_hash(files_to_hash_forward)
    hash_reverse = VersionedDatasetBuilder._get_hash(files_to_hash_reverse)
    assert hash_forward
    assert hash_reverse
    assert hash_forward == hash_reverse
