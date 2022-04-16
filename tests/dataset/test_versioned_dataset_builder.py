"""Tests versioned_dataset_builder.py."""

import os
import shutil
from pathlib import Path
from datetime import datetime
import json
import pickle
from urllib.parse import urlparse
import dateutil
import numpy as np
import pytest
import boto3
from s3fs import S3FileSystem
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder, \
    STRATEGY_COPY_ZIP, STRATEGY_COPY, STRATEGY_LINK
from mlops.errors import PublicationPathAlreadyExistsError, \
    InvalidDatasetCopyStrategyError
from tests.dataset.preset_data_processor import PresetDataProcessor
from tests.dataset.doubled_preset_data_processor import \
    DoubledPresetDataProcessor

TEST_DATASET_PATH_LOCAL = '/tmp/test_versioned_dataset_builder/raw_dataset'
TEST_PUBLICATION_PATH_LOCAL = '/tmp/test_versioned_dataset_builder/datasets'
TEST_DATASET_PATH_S3 = ('s3://kosta-mlops/test_versioned_dataset_builder/'
                        'raw_dataset')
TEST_PUBLICATION_PATH_S3 = ('s3://kosta-mlops/test_versioned_dataset_builder/'
                            'datasets')
TEST_DATASET_FILENAMES = ['file0.txt',
                          'file1.txt',
                          'file2.txt',
                          'sub1/file3.txt',
                          'sub1/sub2/file4.txt']
TEST_DATASET_HASH = 'cf178bc689619bce0844222e3d8a387b'


def _remove_test_directories_local() -> None:
    """Removes the local test directories."""
    for dirname in (TEST_DATASET_PATH_LOCAL, TEST_PUBLICATION_PATH_LOCAL):
        try:
            shutil.rmtree(dirname)
        except FileNotFoundError:
            pass


def _remove_test_directories_s3() -> None:
    """Removes the S3 test directories."""
    fs = S3FileSystem()
    for dirname in (TEST_DATASET_PATH_S3, TEST_PUBLICATION_PATH_S3):
        try:
            fs.rm(dirname, recursive=True)
        except FileNotFoundError:
            pass


def _create_test_dataset_local() -> None:
    """Creates a preset raw dataset at the local test dataset path."""
    for filename in TEST_DATASET_FILENAMES:
        dirname = os.path.dirname(os.path.join(TEST_DATASET_PATH_LOCAL,
                                               filename))
        dirpath = Path(dirname)
        dirpath.mkdir(parents=True, exist_ok=True)
        with open(os.path.join(TEST_DATASET_PATH_LOCAL, filename),
                  'w',
                  encoding='utf-8') as outfile:
            outfile.write(filename)


def _create_test_dataset_s3() -> None:
    """Creates a preset raw dataset at the S3 test dataset path."""
    try:
        shutil.rmtree(TEST_DATASET_PATH_LOCAL)
    except FileNotFoundError:
        pass
    _create_test_dataset_local()
    fs = S3FileSystem()
    for filename in TEST_DATASET_FILENAMES:
        local_path = os.path.join(TEST_DATASET_PATH_LOCAL, filename)
        s3_path = os.path.join(TEST_DATASET_PATH_S3, filename)
        fs.put(local_path, s3_path)


def test_publish_appends_explicit_version() -> None:
    """Tests that publish appends the version string to the path."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version=version)
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
    publication_time = dateutil.parser.parse(dirname)
    assert start < publication_time < end


def test_publish_local_path_creates_expected_files() -> None:
    """Tests that publish on a local path creates the expected
    files/directories on the local filesystem."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version=version)
    assert len(os.listdir(TEST_PUBLICATION_PATH_LOCAL)) == 1
    assert os.listdir(TEST_PUBLICATION_PATH_LOCAL)[0] == version
    publication_dir = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version)
    expected_features = {'X_train.npy', 'X_val.npy', 'X_test.npy'}
    expected_labels = {'y_train.npy', 'y_val.npy', 'y_test.npy'}
    assert set(os.listdir(publication_dir)) == expected_features.union(
        expected_labels).union(
        {'data_processor.pkl', 'meta.json', 'raw.tar.bz2'})


@pytest.mark.mockedawstest
def test_publish_s3_path_creates_expected_files(mocked_s3: None) -> None:
    """Tests that publish on an S3 path creates the expected files/directories
    on S3.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument,too-many-locals
    _remove_test_directories_local()
    _create_test_dataset_local()
    _remove_test_directories_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_S3, version=version)
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
    # Check for raw zip file.
    raw_directory_key = os.path.join(prefix, 'raw.tar.bz2')
    assert any(key.startswith(raw_directory_key) for key in item_keys)


@pytest.mark.mockedawstest
def test_publish_from_raw_dataset_in_s3_to_local(mocked_s3: None) -> None:
    """Tests that publish correctly reads the dataset path when the dataset is
    in S3 and writes to the local filesystem.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _remove_test_directories_s3()
    _create_test_dataset_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_S3, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL,
                    version=version,
                    dataset_copy_strategy=STRATEGY_COPY)
    raw_dataset_dir = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version, 'raw')
    raw_dataset_paths = set()
    for current_path, _, filenames in os.walk(raw_dataset_dir):
        for filename in filenames:
            raw_dataset_paths.add(os.path.join(current_path, filename).replace(
                raw_dataset_dir + '/', '', 1))
    assert raw_dataset_paths == set(TEST_DATASET_FILENAMES)


@pytest.mark.mockedawstest
def test_publish_from_raw_dataset_in_s3_to_s3(mocked_s3: None) -> None:
    """Tests that publish correctly reads the dataset path when the dataset is
    in S3 and writes to S3.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _remove_test_directories_s3()
    _create_test_dataset_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_S3, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_S3,
                    version=version,
                    dataset_copy_strategy=STRATEGY_COPY)
    raw_dataset_dir = os.path.join(TEST_PUBLICATION_PATH_S3, version, 'raw')
    fs = S3FileSystem()
    s3_filenames = {f's3://{key}' for key in fs.find(raw_dataset_dir)}
    dataset_filenames = {os.path.join(raw_dataset_dir, name)
                         for name in TEST_DATASET_FILENAMES}
    assert s3_filenames == dataset_filenames


def test_publish_local_path_raises_path_already_exists_error() -> None:
    """Tests that publish on a local path that already exists raises a
    PublicationPathAlreadyExistsError."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version=version)
    with pytest.raises(PublicationPathAlreadyExistsError):
        builder.publish(TEST_PUBLICATION_PATH_LOCAL, version=version)


@pytest.mark.mockedawstest
def test_publish_s3_path_raises_path_already_exists_error(
        mocked_s3: None) -> None:
    """Tests that publish on an S3 path that already exists raises a
    PublicationPathAlreadyExistsError.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _create_test_dataset_local()
    _remove_test_directories_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_S3, version=version)
    with pytest.raises(PublicationPathAlreadyExistsError):
        builder.publish(TEST_PUBLICATION_PATH_S3, version=version)


def test_publish_zips_raw_dataset() -> None:
    """Tests that publish copies and zips the raw dataset when the copy
    strategy is STRATEGY_COPY_ZIP."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL,
                    version=version,
                    dataset_copy_strategy=STRATEGY_COPY_ZIP)
    raw_dataset_zip = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version,
                                   'raw.tar.bz2')
    assert os.path.exists(raw_dataset_zip)


@pytest.mark.mockedawstest
def test_publish_zips_s3_to_s3(mocked_s3: None) -> None:
    """Tests that publish correctly reads the dataset path when the dataset is
    in S3 and writes to S3.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _remove_test_directories_s3()
    _create_test_dataset_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_S3, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_S3,
                    version=version,
                    dataset_copy_strategy=STRATEGY_COPY_ZIP)
    raw_dataset_zip = os.path.join(TEST_PUBLICATION_PATH_S3, version,
                                   'raw.tar.bz2')
    fs = S3FileSystem()
    assert fs.exists(raw_dataset_zip)


def test_publish_copies_raw_dataset() -> None:
    """Tests that publish copies the entire raw dataset when the copy strategy
    is STRATEGY_COPY."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL,
                    version=version,
                    dataset_copy_strategy=STRATEGY_COPY)
    raw_dataset_dir = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version, 'raw')
    raw_dataset_paths = set()
    for current_path, _, filenames in os.walk(raw_dataset_dir):
        for filename in filenames:
            raw_dataset_paths.add(os.path.join(current_path, filename).replace(
                raw_dataset_dir + '/', '', 1))
    assert raw_dataset_paths == set(TEST_DATASET_FILENAMES)


def test_publish_includes_raw_dataset_link() -> None:
    """Tests that publish includes a link to the raw dataset when the copy
    strategy is STRATEGY_LINK."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL,
                    version=version,
                    dataset_copy_strategy=STRATEGY_LINK)
    raw_dataset_dir = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version, 'raw')
    link_filename = 'link.txt'
    assert set(os.listdir(raw_dataset_dir)) == {link_filename}
    with open(os.path.join(raw_dataset_dir, link_filename),
              'r',
              encoding='utf-8') as infile:
        assert infile.read() == TEST_DATASET_PATH_LOCAL


@pytest.mark.mockedawstest
def test_publish_includes_raw_dataset_link_s3(mocked_s3: None) -> None:
    """Tests that publish to S3 includes a link to the raw dataset when the
    copy strategy is STRATEGY_LINK.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _create_test_dataset_local()
    _remove_test_directories_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_S3,
                    version=version,
                    dataset_copy_strategy=STRATEGY_LINK)
    raw_dataset_dir = os.path.join(TEST_PUBLICATION_PATH_S3, version, 'raw')
    link_filename = 'link.txt'
    fs = S3FileSystem()
    # Remove 's3://' from latter path.
    assert set(fs.ls(raw_dataset_dir)) == {os.path.join(raw_dataset_dir,
                                                        link_filename)[5:]}
    with fs.open(os.path.join(raw_dataset_dir, link_filename),
                 'r',
                 encoding='utf-8') as infile:
        assert infile.read() == TEST_DATASET_PATH_LOCAL


def test_publish_raises_invalid_dataset_copy_strategy_error() -> None:
    """Tests that publish raises an InvalidDatasetCopyStrategyError when the
    dataset copy strategy is not one of the available options."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    with pytest.raises(InvalidDatasetCopyStrategyError):
        builder.publish(TEST_PUBLICATION_PATH_LOCAL,
                        version=version,
                        dataset_copy_strategy='dne')


def test_publish_includes_expected_metadata() -> None:
    """Tests that publish creates a file meta.json with the expected
    metadata."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version=version)
    meta_path = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version, 'meta.json')
    with open(meta_path, 'r', encoding='utf-8') as infile:
        contents = json.loads(infile.read())
    assert set(contents.keys()) == {
        'name', 'version', 'hash', 'created_at', 'tags'}


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
    builder.publish(TEST_PUBLICATION_PATH_LOCAL + '/', version=version)
    expected_filename = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)
    _remove_test_directories_local()
    _create_test_dataset_local()
    # Many trailing slashes.
    builder.publish(TEST_PUBLICATION_PATH_LOCAL + '///', version=version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)


def test_same_datasets_have_same_hashes() -> None:
    """Tests that the hash values from two datasets that have identical files
    are the same."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version='v1')
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version='v2')
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


def test_rebuilt_datasets_have_same_hashes_local_to_local() -> None:
    """Tests that the hash values from two datasets that have identical files
    are the same, even when the datasets have different metadata (e.g.,
    timestamp)."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version='v1')
    shutil.rmtree(TEST_DATASET_PATH_LOCAL)
    _create_test_dataset_local()
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version='v2')
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
    assert contents1['hash'] == TEST_DATASET_HASH


@pytest.mark.mockedawstest
def test_rebuilt_datasets_have_same_hashes_s3_to_local(
        mocked_s3: None) -> None:
    """Tests that the hash values from two datasets that have identical files
    are the same, even when the datasets have different metadata (e.g.,
    timestamp).

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _remove_test_directories_s3()
    _create_test_dataset_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_S3, processor)
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version='v1')
    fs = S3FileSystem()
    fs.rm(TEST_DATASET_PATH_S3, recursive=True)
    _create_test_dataset_s3()
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version='v2')
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
    assert contents1['hash'] == TEST_DATASET_HASH


@pytest.mark.mockedawstest
def test_rebuilt_datasets_have_same_hashes_local_to_s3(
        mocked_s3: None) -> None:
    """Tests that the hash values from two datasets that have identical files
    are the same, even when the datasets have different metadata (e.g.,
    timestamp).

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _remove_test_directories_s3()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    builder.publish(TEST_PUBLICATION_PATH_S3, version='v1')
    shutil.rmtree(TEST_DATASET_PATH_LOCAL)
    _create_test_dataset_local()
    builder.publish(TEST_PUBLICATION_PATH_S3, version='v2')
    meta_path1 = os.path.join(TEST_PUBLICATION_PATH_S3,
                              'v1',
                              'meta.json')
    meta_path2 = os.path.join(TEST_PUBLICATION_PATH_S3,
                              'v2',
                              'meta.json')
    fs = S3FileSystem()
    with fs.open(meta_path1, 'r', encoding='utf-8') as infile:
        contents1 = json.loads(infile.read())
    with fs.open(meta_path2, 'r', encoding='utf-8') as infile:
        contents2 = json.loads(infile.read())
    assert contents1['created_at'] != contents2['created_at']
    assert contents1['hash'] == contents2['hash']
    assert contents1['hash'] == TEST_DATASET_HASH


@pytest.mark.mockedawstest
def test_rebuilt_datasets_have_same_hashes_s3_to_s3(mocked_s3: None) -> None:
    """Tests that the hash values from two datasets that have identical files
    are the same, even when the datasets have different metadata (e.g.,
    timestamp).

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _remove_test_directories_s3()
    _create_test_dataset_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_S3, processor)
    builder.publish(TEST_PUBLICATION_PATH_S3, version='v1')
    fs = S3FileSystem()
    fs.rm(TEST_DATASET_PATH_S3, recursive=True)
    _create_test_dataset_s3()
    builder.publish(TEST_PUBLICATION_PATH_S3, version='v2')
    meta_path1 = os.path.join(TEST_PUBLICATION_PATH_S3,
                              'v1',
                              'meta.json')
    meta_path2 = os.path.join(TEST_PUBLICATION_PATH_S3,
                              'v2',
                              'meta.json')
    with fs.open(meta_path1, 'r', encoding='utf-8') as infile:
        contents1 = json.loads(infile.read())
    with fs.open(meta_path2, 'r', encoding='utf-8') as infile:
        contents2 = json.loads(infile.read())
    assert contents1['created_at'] != contents2['created_at']
    assert contents1['hash'] == contents2['hash']
    assert contents1['hash'] == TEST_DATASET_HASH


def test_different_datasets_have_different_hashes() -> None:
    """Tests that the hash values from two datasets that have different files
    are different."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version='v1')
    # The second dataset is missing one of the raw dataset files.
    os.remove(os.path.join(TEST_DATASET_PATH_LOCAL, TEST_DATASET_FILENAMES[0]))
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version='v2')
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


@pytest.mark.mockedawstest
def test_publish_local_and_s3_create_same_dataset(mocked_s3: None) -> None:
    """Tests that publishing locally or remotely on S3 produces the same
    dataset. Verifies identity by comparing dataset hashes.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _create_test_dataset_local()
    _remove_test_directories_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version=version)
    builder.publish(TEST_PUBLICATION_PATH_S3, version=version)
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
    contents2 = json.loads(contents2['Body'].read().decode('utf-8'))
    assert contents1['created_at'] != contents2['created_at']
    assert contents1['hash'] == contents2['hash']


def test_publish_local_with_trailing_slash() -> None:
    """Tests that publishing to a local path with a trailing slash works
    properly."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL + '/', processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL + '/', version=version)
    expected_filename = os.path.join(TEST_PUBLICATION_PATH_LOCAL, version)
    assert os.path.exists(expected_filename)
    assert os.path.isdir(expected_filename)


@pytest.mark.mockedawstest
def test_publish_s3_with_trailing_slash(mocked_s3: None) -> None:
    """Tests that publishing to an S3 path with a trailing slash works
    properly.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    # pylint: disable=unused-argument
    _remove_test_directories_local()
    _remove_test_directories_s3()
    _create_test_dataset_s3()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_S3 + '/', processor)
    version = 'v1'
    # One trailing slash.
    builder.publish(TEST_PUBLICATION_PATH_S3 + '/', version=version)
    fs = S3FileSystem()
    expected_filename = os.path.join(TEST_PUBLICATION_PATH_S3, version)
    assert fs.ls(expected_filename)
    assert fs.isdir(expected_filename)
    # Many trailing slashes.
    _remove_test_directories_s3()
    _create_test_dataset_s3()
    builder.publish(TEST_PUBLICATION_PATH_S3 + '///', version=version)
    fs = S3FileSystem()
    expected_filename = os.path.join(TEST_PUBLICATION_PATH_S3, version)
    assert fs.ls(expected_filename)
    assert fs.isdir(expected_filename)


def test_published_data_processor_reproduces_dataset() -> None:
    """Tests that the published data processor object can be loaded from the
    pickled file and used to reproduce the dataset."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    version = 'v1'
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version=version)
    processor_filename = os.path.join(TEST_PUBLICATION_PATH_LOCAL,
                                      version,
                                      'data_processor.pkl')
    with open(processor_filename, 'rb') as infile:
        processor_loaded = pickle.load(infile)
    original_features = processor.get_preprocessed_features(
        TEST_DATASET_PATH_LOCAL)
    loaded_features = processor_loaded.get_preprocessed_features(
        TEST_DATASET_PATH_LOCAL)
    assert set(original_features.keys()) == set(loaded_features.keys())
    for name in original_features.keys():
        original_tensor = original_features[name]
        loaded_tensor = loaded_features[name]
        assert np.array_equal(original_tensor, loaded_tensor)


def test_publish_hashes_tensors() -> None:
    """Tests that publish() hashes tensors."""
    _remove_test_directories_local()
    _create_test_dataset_local()
    processor = PresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version='v1')
    processor = DoubledPresetDataProcessor()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH_LOCAL, processor)
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version='v2')
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
    assert contents1['hash'] != contents2['hash']
