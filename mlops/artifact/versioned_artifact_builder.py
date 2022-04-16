"""Contains the VersionedArtifactBuilder class."""

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from s3fs import S3FileSystem
from mlops.errors import PublicationPathAlreadyExistsError


class VersionedArtifactBuilder(ABC):
    """Represents a versioned artifact builder."""

    # pylint: disable=too-few-public-methods

    @abstractmethod
    def publish(self, path: str, *args: Any, **kwargs: Any) -> str:
        """Saves the versioned artifact files to the given path. If the path
        and appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.


        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the artifact should be saved. The
            version will be appended to this path as a subdirectory. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir". It is
            recommended to use this same path to publish all artifacts of a
            given type (e.g., datasets, models), since it will prevent the user
            from creating two different artifacts with the same version.
        :param args: Additional positional args.
        :param kwargs: Keyword args.
        :return: The versioned artifact's publication path.
        """

    @staticmethod
    def _make_publication_path_local(publication_path: str) -> None:
        """Creates the directories that compose the publication path.

        :param publication_path: The path to which to publish the artifact.
        """
        path_obj = Path(publication_path)
        try:
            path_obj.mkdir(parents=True, exist_ok=False)
        except FileExistsError as err:
            raise PublicationPathAlreadyExistsError from err

    @staticmethod
    def _make_publication_path_s3(
        publication_path: str, fs: S3FileSystem
    ) -> None:
        """Creates the directories that compose the publication path.

        :param publication_path: The path to which to publish the artifact.
        :param fs: The S3 filesystem object to interface with S3.
        """
        # fs.mkdirs with exist_ok=False does not raise an error, so use ls.
        if fs.ls(publication_path):
            raise PublicationPathAlreadyExistsError
        fs.mkdirs(publication_path)
