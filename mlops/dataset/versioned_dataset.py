"""Contains the VersionedDataset class."""

import os
import json
import dill as pickle
import numpy as np
from s3fs import S3FileSystem
from mlops.republication import republication


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
        return republication.republish(self.path, path, self.version)

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
