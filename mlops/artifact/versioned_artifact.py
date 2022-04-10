"""Contains the VersionedArtifact class."""

from abc import ABC, abstractmethod
from typing import Any
import os
import shutil
import json
from s3fs import S3FileSystem
from mlops.artifact.versioned_artifact_builder import VersionedArtifactBuilder


class VersionedArtifact(ABC):
    """Represents a versioned artifact (e.g., a dataset or model)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the artifact's name.

        :return: The artifact's name.
        """

    @property
    @abstractmethod
    def path(self) -> str:
        """Returns the local or remote path to the artifact.

        :return: The local or remote path to the artifact.
        """

    @property
    @abstractmethod
    def metadata_path(self) -> str:
        """Returns the local or remote path to the artifact's metadata.

        :return: The local or remote path to the artifact's metadata.
        """

    @property
    @abstractmethod
    def version(self) -> str:
        """Returns the artifact's version.

        :return: The artifact's version.
        """

    @property
    @abstractmethod
    def md5(self) -> str:
        """Returns the artifact's MD5 hash.

        :return: The artifact's MD5 hash.
        """

    def __eq__(self, other: 'VersionedArtifact') -> bool:
        """Returns True if the two objects have the same loaded MD5 hash code,
        False otherwise.

        :param other: The artifact with which to compare this object.
        :return: True if the object MD5 hashes match.
        """
        return self.md5 == other.md5

    def __hash__(self) -> int:
        """Returns this object's hashcode based on the loaded MD5 hashcode.

        :return: The object's hashcode based on the loaded MD5 hashcode.
        """
        return hash(self.md5)

    def update_metadata(self, updates: dict[str, Any]) -> None:
        """Updates the artifact's metadata with the new values.

        The current object will not reflect changes made to the metadata and
        will need to be reloaded from the path. For example, updating the
        object's name in the metadata will not change the object's name field,
        but will change the name of any future copies of the object loaded from
        the same versioned files.

        :param updates: The dictionary of keys and values to add or update. If
            a key does not exist in the metadata, it is added; if it does
            exist, its value is overwritten.
        """
        if self.metadata_path.startswith('s3://'):
            fs = S3FileSystem()
            with fs.open(self.metadata_path,
                         'r',
                         encoding='utf-8') as infile:
                metadata = json.loads(infile.read())
        else:
            with open(self.metadata_path, 'r', encoding='utf-8') as infile:
                metadata = json.loads(infile.read())
        updated_metadata = {**metadata, **updates}
        if self.metadata_path.startswith('s3://'):
            with fs.open(self.metadata_path,
                         'w',
                         encoding='utf-8') as outfile:
                outfile.write(json.dumps(updated_metadata))
        else:
            with open(self.metadata_path, 'w', encoding='utf-8') as outfile:
                outfile.write(json.dumps(updated_metadata))

    def republish(self, republication_path: str) -> str:
        """Saves the versioned artifact files to the given path. If the path
        and appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        :param republication_path: The path, either on the local filesystem or
            in a cloud store such as S3, to which the artifact should be saved.
            The version will be appended to this path as a subdirectory. An S3
            path should be a URL of the form "s3://bucket-name/path/to/dir". It
            is recommended to use this same path to publish all artifacts,
            since it will prevent the user from creating two different
            artifacts with the same version.
        :return: The versioned artifact's publication path.
        """
        if republication_path.startswith('s3://'):
            return self._republish_to_s3(republication_path)
        return self._republish_to_local(republication_path)

    def _republish_to_local(self, republication_path: str) -> str:
        """Saves the versioned dataset files to the given path. If the path and
        appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        :param republication_path: The local path to which to publish the
            versioned object.
        :return: The versioned object's republication path.
        """
        # pylint: disable=protected-access
        publication_path = os.path.join(republication_path, self.version)
        VersionedArtifactBuilder._make_publication_path_local(publication_path)
        if self.path.startswith('s3://'):
            fs = S3FileSystem()
            fs.get(self.path, publication_path, recursive=True)
        else:
            shutil.copytree(self.path, publication_path, dirs_exist_ok=True)
        return publication_path

    def _republish_to_s3(self, republication_path: str) -> str:
        """Saves the versioned dataset files to the given path. If the path and
        appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        :param republication_path: The S3 path to which to publish the
            versioned object.
        :return: The versioned object's republication path.
        """
        # pylint: disable=protected-access
        publication_path = os.path.join(republication_path, self.version)
        fs = S3FileSystem()
        VersionedArtifactBuilder._make_publication_path_s3(
            publication_path, fs)
        if self.path.startswith('s3://'):
            artifact_path_no_prefix = self.path.replace('s3://', '', 1)
            copy_path_no_prefix = publication_path.replace('s3://', '', 1)
            for current_path, _, filenames in fs.walk(self.path):
                outfile_prefix = current_path.replace(artifact_path_no_prefix,
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
