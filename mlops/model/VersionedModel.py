"""Contains the VersionedModel class."""

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from mlops import ENDPOINT_LOCAL, ENDPOINT_S3
from mlops.dataset.VersionedDataset import VersionedDataset


class VersionedModel:
    """TODO"""
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
                 metric: str = 'val_loss',
                 function: str = 'min') -> 'VersionedModel':
        """TODO"""
