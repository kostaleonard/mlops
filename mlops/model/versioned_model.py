"""Contains the VersionedModel class."""
# pylint: disable=no-name-in-module

from typing import Callable, Any
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from mlops import ENDPOINT_LOCAL, ENDPOINT_S3
from mlops.dataset.versioned_dataset import VersionedDataset


class VersionedModel:
    """Represents a versioned model."""
    versioned_dataset: VersionedDataset
    model: Model
    history: History

    def __init__(self,
                 path: str,
                 endpoint: str = ENDPOINT_LOCAL) -> None:
        """Instantiates the object.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, from which the model should be loaded.
        :param endpoint: The system from which the model files should be
            loaded. 'local' loads the files from the local filesystem; 's3'
            loads the files from S3, in which case path should be an S3 URL.
        """
        # TODO

    @staticmethod
    def get_best(paths: list[str],
                 endpoint: str = ENDPOINT_LOCAL,
                 metric: str = 'val_loss',
                 comparator: Callable[[Any, Any], Any] = min) -> \
            'VersionedModel':
        """Returns an instance of the best VersionedModel from the given paths.

        :param paths: A list of the VersionedModel paths to check.
        :param endpoint: The system from which the model files should be
            loaded. 'local' loads the files from the local filesystem; 's3'
            loads the files from S3, in which case path should be an S3 URL.
        :param metric: The name of the metric in the model's training history
            to use as the basis of comparison for determining the best model.
            This metric is often the validation loss.
        :param comparator: The function to use to compare model metric values.
            It takes two model metric values and returns the more desirable of
            the two. For example, min will return the model with the lowest
            metric value, say, validation loss.
        """
        # TODO
