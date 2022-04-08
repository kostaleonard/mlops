"""Contains the VersionedArtifact class."""

from abc import ABC, abstractmethod
from typing import Any


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

        :param updates: The dictionary of keys and values to add or update. If
            a key does not exist in the metadata, it is added; if it does
            exist, its value is overwritten.
        """
        # TODO
