"""Contains the VersionedDatasetBuilder class."""

import os
from pathlib import Path
import shutil
from tempfile import TemporaryFile
from typing import Optional, Collection
import pickle
import hashlib
from functools import partial
from datetime import datetime
import json
import numpy as np
from s3fs import S3FileSystem
from mlops.dataset.data_processor import DataProcessor
from mlops.errors import PublicationPathAlreadyExistsError, \
    InvalidDatasetCopyStrategyError

STRATEGY_COPY = 'copy'
STRATEGY_LINK = 'link'
CHUNK_SIZE = 2 ** 20


class VersionedDatasetBuilder:
    """An object containing all of the components that form a versioned dataset.
    This object is only used to ensure a standard format for datasets stored in
    a dataset archive (such as the local filesystem or S3), and is not meant for
    consumption by downstream models."""

    def __init__(self,
                 dataset_path: str,
                 data_processor: DataProcessor) -> None:
        """Instantiates the object. Features and labels will be extracted from
        the dataset path using the DataProcessor object. When this object is
        published, the feature and label tensors will be saved with '.npy' as
        the suffix. For example, if a tensor is named 'X_train', it will be
        saved as 'X_train.npy'.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :param data_processor: The DataProcessor object with which the features
            and labels were generated. This object is saved so that properly
            formatted features can be generated at prediction time, and so that
            features and labels can be "unpreprocessed" to match their original
            representations.
        """
        self.dataset_path = dataset_path
        self.data_processor = data_processor
        self.features = data_processor.get_preprocessed_features(dataset_path)
        self.labels = data_processor.get_preprocessed_labels(dataset_path)

    def publish(self,
                path: str,
                version: Optional[str] = None,
                dataset_copy_strategy: str = STRATEGY_COPY,
                tags: Optional[list[str]] = None) -> None:
        """Saves the versioned dataset files to the given path. If the path and
        appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        The following files will be created:
            path/version/ (the publication path and version)
                X_train.npy (and other feature tensors by their given names)
                y_train.npy (and other label tensors by their given names)
                data_processor.pkl (DataProcessor object)
                meta.json (metadata)
                raw/ (non-empty directory with the raw dataset files)
                    ...

        The contents of meta.json will be:
            {
                version: (dataset version)
                hash: (MD5 hash of all objects apart from data_processor.pkl and
                    meta.json)
                created_at: (timestamp)
                tags: (optional list of tags)
            }

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the dataset should be saved. The version
            will be appended to this path as a subdirectory. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir". It is
            recommended to use this same path to publish all datasets, since it
            will prevent the user from creating two different datasets with the
            same version.
        :param version: A string indicating the dataset version. The version
            should be unique to this dataset. If None, the publication timestamp
            will be used as the version.
        :param dataset_copy_strategy: The strategy by which to copy the
            original, raw dataset to the published path. The default is
            STRATEGY_COPY, which recursively copies all files and directories
            from the dataset path supplied at instantiation to the published
            path so that the dataset can be properly versioned. STRATEGY_LINK
            will instead create a file 'link.txt' containing the supplied
            dataset path; this is desirable if the raw dataset is already stored
            in a versioned repository, and copying would create an unnecessary
            duplicate.
        :param tags: An optional list of string tags to add to the dataset
            metadata.
        """
        timestamp = datetime.now().isoformat()
        if not version:
            version = timestamp
        if not tags:
            tags = []
        publication_path = os.path.join(path, version)
        copy_path = os.path.join(publication_path, 'raw')
        link_path = os.path.join(copy_path, 'link.txt')
        metadata_path = os.path.join(publication_path, 'meta.json')
        metadata = {
            'version': version,
            'hash': 'TDB',
            'created_at': timestamp,
            'tags': tags}
        if path.startswith('s3://'):
            self._publish_s3(publication_path,
                             copy_path,
                             link_path,
                             metadata_path,
                             dataset_copy_strategy,
                             metadata)
        else:
            self._publish_local(publication_path,
                                copy_path,
                                link_path,
                                metadata_path,
                                dataset_copy_strategy,
                                metadata)

    def _publish_local(self,
                       publication_path: str,
                       copy_path: str,
                       link_path: str,
                       metadata_path: str,
                       dataset_copy_strategy: str,
                       metadata: dict) -> None:
        """Saves the versioned dataset files to the given local path. See
        publish() for more detailed information.

        :param publication_path: The local path to which to publish the dataset.
        :param copy_path: The path to which the raw dataset is copied.
        :param link_path: The path to the file containing the raw dataset link.
        :param metadata_path: The path to which the metadata should be saved.
        :param dataset_copy_strategy: The strategy by which to copy the
            original, raw dataset to the published path.
        :param metadata: Dataset metadata.
        """
        files_to_hash = set()
        # Create publication path.
        VersionedDatasetBuilder._make_publication_path_local(publication_path)
        # Save tensors.
        self._write_tensors_local(publication_path)
        # Save the raw dataset.
        if dataset_copy_strategy == STRATEGY_COPY:
            file_paths = self._copy_raw_dataset_local(copy_path)
            files_to_hash = files_to_hash.union(file_paths)
        elif dataset_copy_strategy == STRATEGY_LINK:
            self._make_raw_dataset_link_local(copy_path, link_path)
            files_to_hash.add(link_path)
        else:
            raise InvalidDatasetCopyStrategyError
        # Save metadata.
        hash_digest = VersionedDatasetBuilder._get_hash_local(files_to_hash)
        metadata['hash'] = hash_digest
        VersionedDatasetBuilder._write_metadata_local(metadata, metadata_path)
        # Save data processor object.
        self._write_data_processor_local(publication_path)

    def _publish_s3(self,
                    publication_path: str,
                    copy_path: str,
                    link_path: str,
                    metadata_path: str,
                    dataset_copy_strategy: str,
                    metadata: dict) -> None:
        """Saves the versioned dataset files to the given S3 path. See publish()
        for more detailed information.

        :param publication_path: The local path to which to publish the dataset.
        :param copy_path: The path to which the raw dataset is copied.
        :param link_path: The path to the file containing the raw dataset link.
        :param metadata_path: The path to which the metadata should be saved.
        :param dataset_copy_strategy: The strategy by which to copy the
            original, raw dataset to the published path.
        :param metadata: Dataset metadata.
        """
        fs = S3FileSystem()
        files_to_hash = set()
        # Create publication path.
        VersionedDatasetBuilder._make_publication_path_s3(publication_path, fs)
        # Save tensors.
        self._write_tensors_s3(publication_path, fs)
        # Save the raw dataset.
        if dataset_copy_strategy == STRATEGY_COPY:
            file_paths = self._copy_raw_dataset_s3(copy_path, fs)
            files_to_hash = files_to_hash.union(file_paths)
        elif dataset_copy_strategy == STRATEGY_LINK:
            self._make_raw_dataset_link_s3(link_path, fs)
            files_to_hash.add(link_path)
        else:
            raise InvalidDatasetCopyStrategyError
        # Save metadata.
        hash_digest = VersionedDatasetBuilder._get_hash_s3(files_to_hash, fs)
        metadata['hash'] = hash_digest
        VersionedDatasetBuilder._write_metadata_s3(metadata, metadata_path, fs)
        # Save data processor object.
        self._write_data_processor_s3(publication_path, fs)

    @staticmethod
    def _make_publication_path_local(publication_path: str) -> None:
        """TODO"""
        path_obj = Path(publication_path)
        try:
            path_obj.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise PublicationPathAlreadyExistsError

    @staticmethod
    def _make_publication_path_s3(publication_path: str, fs: S3FileSystem) -> None:
        """TODO"""
        # fs.mkdirs with exist_ok=False does not raise an error, so use ls.
        if fs.ls(publication_path):
            raise PublicationPathAlreadyExistsError
        fs.mkdirs(publication_path)

    def _write_tensors_local(self, publication_path: str) -> set[str]:
        """TODO"""
        file_paths = set()
        for name, tensor in {**self.features, **self.labels}.items():
            file_path = os.path.join(publication_path, f'{name}.npy')
            file_paths.add(file_path)
            VersionedDatasetBuilder._write_tensor_local(tensor, file_path)
        return file_paths

    def _write_tensors_s3(self, publication_path: str, fs: S3FileSystem) -> set[str]:
        """TODO"""
        file_paths = set()
        for name, tensor in {**self.features, **self.labels}.items():
            file_path = os.path.join(publication_path, f'{name}.npy')
            file_paths.add(file_path)
            VersionedDatasetBuilder._write_tensor_s3(tensor, file_path, fs)
        return file_paths

    @staticmethod
    def _write_tensor_local(tensor: np.ndarray, path: str) -> None:
        """TODO"""
        np.save(path, tensor)

    @staticmethod
    def _write_tensor_s3(tensor: np.ndarray, path: str, fs: S3FileSystem) -> None:
        """TODO"""
        with TemporaryFile() as tmp_file:
            np.save(tmp_file, tensor)
            tmp_file.seek(0)
            with fs.open(path, 'wb') as outfile:
                outfile.write(tmp_file.read())

    def _copy_raw_dataset_local(self, copy_path: str) -> set[str]:
        """TODO"""
        file_paths = set()
        if self.dataset_path.startswith('s3://'):
            # Copy raw dataset from S3 to local filesystem.
            fs = S3FileSystem()
            fs.get(self.dataset_path, copy_path, recursive=True)
        else:
            # Copy raw dataset from local filesystem to local filesystem.
            shutil.copytree(self.dataset_path, copy_path)
            for current_path, _, filenames in os.walk(copy_path):
                for filename in filenames:
                    file_paths.add(os.path.join(current_path, filename))
        return file_paths

    def _copy_raw_dataset_s3(self, copy_path: str, fs: S3FileSystem) -> set[str]:
        """TODO"""
        s3_file_paths = set()
        if self.dataset_path.startswith('s3://'):
            # Copy raw dataset from S3 to S3.
            dataset_path_no_prefix = self.dataset_path.replace('s3://', '', 1)
            copy_path_no_prefix = copy_path.replace('s3://', '', 1)
            for current_path, subdirs, filenames in fs.walk(self.dataset_path):
                subdirs = current_path.replace(dataset_path_no_prefix,
                                               copy_path_no_prefix, 1)
                for filename in filenames:
                    infile_path = os.path.join(current_path,
                                               filename)
                    outfile_path = os.path.join(subdirs,
                                                filename)
                    fs.copy(infile_path, outfile_path)
        else:
            # Copy raw dataset from local filesystem to S3.
            fs.put(self.dataset_path, copy_path, recursive=True)
        return s3_file_paths

    def _make_raw_dataset_link_local(self, copy_path: str, link_path: str) -> None:
        """TODO"""
        os.mkdir(copy_path)
        with open(link_path, 'w', encoding='utf-8') as outfile:
            outfile.write(self.dataset_path)

    def _make_raw_dataset_link_s3(self, link_path: str, fs: S3FileSystem) -> None:
        """TODO"""
        with fs.open(link_path, 'w', encoding='utf-8') as outfile:
            outfile.write(self.dataset_path)

    @staticmethod
    def _write_metadata_local(metadata: dict, metadata_path: str) -> None:
        """TODO"""
        with open(metadata_path, 'w', encoding='utf-8') as outfile:
            outfile.write(json.dumps(metadata))

    @staticmethod
    def _write_metadata_s3(metadata: dict, metadata_path: str, fs: S3FileSystem) -> None:
        """TODO"""
        with fs.open(metadata_path, 'w', encoding='utf-8') as outfile:
            outfile.write(json.dumps(metadata))

    def _write_data_processor_local(self, publication_path: str) -> None:
        """TODO"""
        with open(os.path.join(publication_path, 'data_processor.pkl'),
                  'wb') as outfile:
            outfile.write(pickle.dumps(self.data_processor))

    def _write_data_processor_s3(self, publication_path: str, fs: S3FileSystem) -> None:
        """TODO"""
        with fs.open(os.path.join(publication_path, 'data_processor.pkl'),
                     'wb') as outfile:
            outfile.write(pickle.dumps(self.data_processor))

    @staticmethod
    def _get_hash_local(files_to_hash: Collection[str]) -> str:
        """Returns the MD5 hex digest string from hashing the content of all the
        given files on the local filesystem. The files are sorted before hashing
        so that the process is reproducible.

        :param files_to_hash: A collection of paths to files whose contents
            should be hashed.
        :return: The MD5 hex digest string from hashing the content of all the
            given files.
        """
        hash_md5 = hashlib.md5()
        for filename in sorted(files_to_hash):
            with open(filename, 'rb') as infile:
                for chunk in iter(partial(infile.read, CHUNK_SIZE), b''):
                    hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @staticmethod
    def _get_hash_s3(files_to_hash: Collection[str],
                     fs: S3FileSystem) -> str:
        """Returns the MD5 hex digest string from hashing the content of all the
        given files in S3. The files are sorted before hashing so that the
        process is reproducible.

        :param files_to_hash: A collection of paths to files whose contents
            should be hashed.
        :param fs: TODO
        :return: The MD5 hex digest string from hashing the content of all the
            given files.
        """
        hash_md5 = hashlib.md5()
        for filename in sorted(files_to_hash):
            with fs.open(filename, 'rb') as infile:
                for chunk in iter(partial(infile.read, CHUNK_SIZE), b''):
                    hash_md5.update(chunk)
        return hash_md5.hexdigest()
