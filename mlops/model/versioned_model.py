"""Contains the VersionedModel class."""
# pylint: disable=no-name-in-module

import os
import json
from tempfile import NamedTemporaryFile
from s3fs import S3FileSystem
from tensorflow.keras.models import load_model


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
            # Get hash.
            with fs.open(os.path.join(path, 'meta.json'),
                         'r',
                         encoding='utf-8') as infile:
                metadata = json.loads(infile.read())
            self.md5 = metadata['hash']
        else:
            # Get model.
            self.model = load_model(os.path.join(path, 'model.h5'))
            # Get hash.
            with open(os.path.join(path, 'meta.json'),
                      'r',
                      encoding='utf-8') as infile:
                metadata = json.loads(infile.read())
            self.md5 = metadata['hash']

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
