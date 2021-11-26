"""Contains the VersionedModel class."""
# pylint: disable=no-name-in-module

from typing import Callable, Any
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from mlops.dataset.versioned_dataset import VersionedDataset


class VersionedModel:
    """Represents a versioned model."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, from which the model should be loaded. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir".
        """
        # TODO Model, History, and VersionedDataset

    @staticmethod
    def get_best(paths: list[str],
                 metric: str = 'val_loss',
                 comparator: Callable[[Any, Any], Any] = min) -> \
            'VersionedModel':
        """Returns an instance of the best VersionedModel from the given paths.

        :param paths: A list of the VersionedModel paths to check. Each path may
            be on the local filesystem or remote, independent of the other
            paths. An S3 path should be a URL of the form
            "s3://bucket-name/path/to/dir".
        :param metric: The name of the metric in the model's training history
            to use as the basis of comparison for determining the best model.
            This metric is often the validation loss.
        :param comparator: The function to use to compare model metric values.
            It takes two model metric values and returns the more desirable of
            the two. For example, min will return the model with the lowest
            metric value, say, validation loss.
        """
        # TODO

    # TODO search (by version, by tag, by timestamp)

    # TODO mirror VersionedDataset interface
