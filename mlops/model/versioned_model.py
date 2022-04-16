"""Contains the VersionedModel class."""
# pylint: disable=no-name-in-module

import os
import json
from tempfile import NamedTemporaryFile
from s3fs import S3FileSystem
from tensorflow.keras.models import load_model
from mlops.artifact.versioned_artifact import VersionedArtifact


class VersionedModel(VersionedArtifact):
    """Represents a versioned model."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, from which the model should be loaded. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir".
        """
        self._path = path
        self._metadata_path = os.path.join(path, "meta.json")
        if path.startswith("s3://"):
            fs = S3FileSystem()
            # Get model.
            with NamedTemporaryFile() as tmp_file:
                with fs.open(os.path.join(path, "model.h5"), "rb") as infile:
                    tmp_file.write(infile.read())
                tmp_file.seek(0)
                self.model = load_model(tmp_file.name)
            # Get metadata.
            with fs.open(self.metadata_path, "r", encoding="utf-8") as infile:
                metadata = json.loads(infile.read())
        else:
            # Get model.
            self.model = load_model(os.path.join(path, "model.h5"))
            # Get metadata.
            with open(self.metadata_path, "r", encoding="utf-8") as infile:
                metadata = json.loads(infile.read())
        self._name = metadata["name"]
        self._version = metadata["version"]
        self._md5 = metadata["hash"]
        self.dataset_path = metadata["dataset"]

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
