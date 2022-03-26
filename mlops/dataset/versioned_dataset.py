"""Contains the VersionedDataset class."""

import os
import shutil
import json
import dill as pickle
import numpy as np
from s3fs import S3FileSystem
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder


class VersionedDataset:
    """Represents a versioned dataset."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, from which the dataset should be loaded. An S3
            path should be a URL of the form "s3://bucket-name/path/to/dir".
        """
        self.path = path
        if path.startswith('s3://'):
            fs = S3FileSystem()
            # Get tensors.
            tensor_paths = {tensor_path
                            for tensor_path in fs.ls(path)
                            if tensor_path.endswith('.npy')}
            for tensor_path in tensor_paths:
                attr_name = tensor_path.split('.npy')[0].split('/')[-1]
                with fs.open(tensor_path, 'rb') as infile:
                    tensor = np.load(infile)
                setattr(self, attr_name, tensor)
            # Get metadata.
            with fs.open(os.path.join(path, 'meta.json'),
                         'r',
                         encoding='utf-8') as infile:
                metadata = json.loads(infile.read())
            self.name = metadata['name']
            self.version = metadata['version']
            self.md5 = metadata['hash']
            # Get data processor.
            with fs.open(os.path.join(path, 'data_processor.pkl'),
                         'rb') as infile:
                processor = pickle.loads(infile.read(), ignore=True)
            self.data_processor = processor
        else:
            # Get tensors.
            tensor_filenames = {tensor_filename
                                for tensor_filename in os.listdir(path)
                                if tensor_filename.endswith('.npy')}
            for tensor_filename in tensor_filenames:
                tensor_path = os.path.join(path, tensor_filename)
                attr_name = tensor_filename.split('.npy')[0]
                tensor = np.load(tensor_path)
                setattr(self, attr_name, tensor)
            # Get metadata.
            with open(os.path.join(path, 'meta.json'),
                      'r',
                      encoding='utf-8') as infile:
                metadata = json.loads(infile.read())
            self.name = metadata['name']
            self.version = metadata['version']
            self.md5 = metadata['hash']
            # Get data processor.
            with open(os.path.join(path, 'data_processor.pkl'), 'rb') as infile:
                processor = pickle.loads(infile.read(), ignore=True)
            self.data_processor = processor

    def _republish_to_local(self, path: str) -> str:
        """Saves the versioned dataset files to the given path. If the path and
        appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        :param path: The local path to which to publish the dataset.
        :return: The versioned dataset's publication path.
        """
        # pylint: disable=protected-access
        publication_path = os.path.join(path, self.version)
        VersionedDatasetBuilder._make_publication_path_local(publication_path)
        if self.path.startswith('s3://'):
            fs = S3FileSystem()
            fs.get(self.path, publication_path, recursive=True)
        else:
            shutil.copytree(self.path, publication_path, dirs_exist_ok=True)
        return publication_path

    def _republish_to_s3(self, path: str) -> str:
        """Saves the versioned dataset files to the given path. If the path and
        appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        :param path: The S3 path to which to publish the dataset.
        :return: The versioned dataset's publication path.
        """
        # pylint: disable=protected-access
        publication_path = os.path.join(path, self.version)
        fs = S3FileSystem()
        VersionedDatasetBuilder._make_publication_path_s3(publication_path, fs)
        if self.path.startswith('s3://'):
            dataset_path_no_prefix = self.path.replace('s3://', '', 1)
            copy_path_no_prefix = publication_path.replace('s3://', '', 1)
            for current_path, _, filenames in fs.walk(self.path):
                outfile_prefix = current_path.replace(dataset_path_no_prefix,
                                                      copy_path_no_prefix, 1)
                for filename in filenames:
                    infile_path = os.path.join(current_path,
                                               filename)
                    outfile_path = os.path.join(outfile_prefix,
                                                filename)
                    fs.copy(infile_path, outfile_path)
        else:
            fs.put(self.path, publication_path, recursive=True)
        return publication_path

    def republish(self, path: str) -> str:
        """Saves the versioned dataset files to the given path. If the path and
        appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the dataset should be saved. The version
            will be appended to this path as a subdirectory. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir". It is
            recommended to use this same path to publish all datasets, since it
            will prevent the user from creating two different datasets with the
            same version.
        :return: The versioned dataset's publication path.
        """
        if path.startswith('s3://'):
            return self._republish_to_s3(path)
        return self._republish_to_local(path)

    def __eq__(self, other: 'VersionedDataset') -> bool:
        """Returns True if the two objects have the same loaded MD5 hash code,
        False otherwise.

        :param other: The dataset with which to compare this object.
        :return: True if the object MD5 hashes match.
        """
        return self.md5 == other.md5

    def __hash__(self) -> int:
        """Returns this object's hashcode based on the loaded MD5 hashcode.

        :return: The object's hashcode based on the loaded MD5 hashcode.
        """
        return hash(self.md5)
