"""Contains the VersionedDataset class."""

import os
import json
import dill as pickle
import numpy as np
from s3fs import S3FileSystem
from mlops.artifact.versioned_artifact import VersionedArtifact


class VersionedDataset(VersionedArtifact):
    """Represents a versioned dataset."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, from which the dataset should be loaded. An S3
            path should be a URL of the form "s3://bucket-name/path/to/dir".
        """
        self._path = path
        self._metadata_path = os.path.join(path, 'meta.json')
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
            with fs.open(self.metadata_path,
                         'r',
                         encoding='utf-8') as infile:
                metadata = json.loads(infile.read())
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
            with open(self.metadata_path,
                      'r',
                      encoding='utf-8') as infile:
                metadata = json.loads(infile.read())
            # Get data processor.
            with open(
                    os.path.join(path, 'data_processor.pkl'),
                    'rb') as infile:
                processor = pickle.loads(infile.read(), ignore=True)
            self.data_processor = processor
        self._name = metadata['name']
        self._version = metadata['version']
        self._md5 = metadata['hash']

    @property
    def name(self) -> str:
        """Returns the artifact's name.

        :return: The artifact's name.
        """
        return self._name

    @property
    def path(self) -> str:
        """Returns the local or remote path to the artifact.

        :return: The local or remote path to the artifact.
        """
        return self._path

    @property
    def metadata_path(self) -> str:
        """Returns the local or remote path to the artifact's metadata.

        :return: The local or remote path to the artifact's metadata.
        """
        return self._metadata_path

    @property
    def version(self) -> str:
        """Returns the artifact's version.

        :return: The artifact's version.
        """
        return self._version

    @property
    def md5(self) -> str:
        """Returns the artifact's MD5 hash.

        :return: The artifact's MD5 hash.
        """
        return self._md5
