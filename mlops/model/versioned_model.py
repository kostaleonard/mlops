"""Contains the VersionedModel class."""
# pylint: disable=no-name-in-module

from typing import Callable, Any


class VersionedModel:
    """Represents a versioned model."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, from which the model should be loaded. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir".
        """
        # TODO get model, dataset path

    # TODO mirror VersionedDataset interface, inc. __eq__ and __hash__
