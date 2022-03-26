"""Contains the VersionedModel class."""
# pylint: disable=no-name-in-module

import os
import json
from tempfile import NamedTemporaryFile
from s3fs import S3FileSystem
from tensorflow.keras.models import load_model
from mlops.republication import republication


class VersionedModel:
    """Represents a versioned model."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, from which the model should be loaded. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir".
        """
        self.path = path
        if path.startswith('s3://'):
            fs = S3FileSystem()
            # Get model.
            with NamedTemporaryFile() as tmp_file:
                with fs.open(os.path.join(path, 'model.h5'), 'rb') as infile:
                    tmp_file.write(infile.read())
                tmp_file.seek(0)
                self.model = load_model(tmp_file.name)
            # Get metadata.
            with fs.open(os.path.join(path, 'meta.json'),
                         'r',
                         encoding='utf-8') as infile:
                metadata = json.loads(infile.read())
            self.name = metadata['name']
            self.version = metadata['version']
            self.md5 = metadata['hash']
        else:
            # Get model.
            self.model = load_model(os.path.join(path, 'model.h5'))
            # Get metadata.
            with open(os.path.join(path, 'meta.json'),
                      'r',
                      encoding='utf-8') as infile:
                metadata = json.loads(infile.read())
            self.name = metadata['name']
            self.version = metadata['version']
            self.md5 = metadata['hash']

    def republish(self, path: str) -> str:
        """Saves the versioned model files to the given path. If the path and
        appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the model should be saved. The version
            will be appended to this path as a subdirectory. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir". It is
            recommended to use this same path to publish all models, since it
            will prevent the user from creating two different models with the
            same version.
        :return: The versioned model's publication path.
        """
        return republication.republish(self.path, path, self.version)

    def __eq__(self, other: 'VersionedModel') -> bool:
        """Returns True if the two objects have the same loaded MD5 hash code,
        False otherwise.

        :param other: The model with which to compare this object.
        :return: True if the object MD5 hashes match.
        """
        return self.md5 == other.md5

    def __hash__(self) -> int:
        """Returns this object's hashcode based on the loaded MD5 hashcode.

        :return: The object's hashcode based on the loaded MD5 hashcode.
        """
        return hash(self.md5)
