"""Contains the VersionedDatasetBuilder class."""

import os
from pathlib import Path
import shutil
from typing import Optional, Collection
import pickle
import hashlib
from functools import partial
from datetime import datetime
import json
import numpy as np
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
        # TODO refactor
        # TODO add S3 logic after local implementation complete
        timestamp = datetime.now().isoformat()
        if not version:
            version = timestamp
        if not tags:
            tags = []
        publication_path = os.path.join(path, version)
        path_obj = Path(publication_path)
        try:
            path_obj.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise PublicationPathAlreadyExistsError
        files_to_hash = set()
        # Save tensors.
        for name, tensor in {**self.features, **self.labels}.items():
            file_path = os.path.join(publication_path, f'{name}.npy')
            files_to_hash.add(file_path)
            np.save(file_path, tensor)
        # Save the raw dataset.
        # TODO dataset_copy_strategy; this could be a function call.
        raw_dataset_path = os.path.join(publication_path, 'raw')
        if dataset_copy_strategy == STRATEGY_COPY:
            shutil.copytree(self.dataset_path, raw_dataset_path)
            for current_path, _, filenames in os.walk(raw_dataset_path):
                for filename in filenames:
                    files_to_hash.add(os.path.join(current_path, filename))
        elif dataset_copy_strategy == STRATEGY_LINK:
            try:
                os.mkdir(raw_dataset_path)
            except FileExistsError:
                pass
            link_path = os.path.join(raw_dataset_path, 'link.txt')
            with open(link_path, 'w', encoding='utf-8') as outfile:
                outfile.write(self.dataset_path)
            files_to_hash.add(link_path)
        else:
            raise InvalidDatasetCopyStrategyError
        # Save metadata.
        hash_digest = VersionedDatasetBuilder._get_hash(files_to_hash)
        metadata = {
            'version': version,
            'hash': hash_digest,
            'created_at': timestamp,
            'tags': tags}
        metadata_path = os.path.join(publication_path, 'meta.json')
        with open(metadata_path, 'w', encoding='utf-8') as outfile:
            outfile.write(json.dumps(metadata))
        # Save data processor object.
        with open(os.path.join(publication_path, 'data_processor.pkl'),
                  'wb') as outfile:
            outfile.write(pickle.dumps(self.data_processor))

    def _publish_local(self,
                       path: str,
                       version: str,
                       dataset_copy_strategy: str,
                       tags: list[str]) -> None:
        """Saves the versioned dataset files to the given local path. See
        publish() for more detailed information.

        :param path: The local path.
        :param version: A string indicating the dataset version.
        :param dataset_copy_strategy: The strategy by which to copy the
            original, raw dataset to the published path.
        :param tags: A list of string tags to add to the dataset metadata.
        """
        # TODO

    def _publish_s3(self,
                    path: str,
                    version: str,
                    dataset_copy_strategy: str,
                    tags: list[str]) -> None:
        """Saves the versioned dataset files to the given S3 path. See publish()
        for more detailed information.

        :param path: The S3 path.
        :param version: A string indicating the dataset version.
        :param dataset_copy_strategy: The strategy by which to copy the
            original, raw dataset to the published path.
        :param tags: A list of string tags to add to the dataset metadata.
        """
        # TODO

    @staticmethod
    def _get_hash(files_to_hash: Collection[str]) -> str:
        """Returns the MD5 hex digest string from hashing the content of all the
        given files. The files are sorted before hashing so that the process is
        reproducible.

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
