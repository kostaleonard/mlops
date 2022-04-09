"""Contains the VersionedArtifact class."""

from abc import ABC, abstractmethod
from typing import Any
import json
from s3fs import S3FileSystem


class VersionedArtifact(ABC):
    """Represents a versioned artifact (e.g., a dataset or model)."""

    @property
    @abstractmethod
    def metadata_path(self) -> str:
        """Returns the local or remote path to the artifact's metadata.

        :return: The local or remote path to the artifact's metadata.
        """

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
