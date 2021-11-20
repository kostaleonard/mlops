"""Contains the VersionedModelBuilder class."""
# pylint: disable=no-name-in-module

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from mlops import ENDPOINT_LOCAL, ENDPOINT_S3
from mlops.dataset.versioned_dataset import VersionedDataset


class VersionedModelBuilder:
    """An object containing all of the components that form a versioned
    model."""

    def __init__(self,
                 versioned_dataset: VersionedDataset,
                 model: Model,
                 history: History) -> None:
        """Instantiates the object.

        TODO
        """

    def publish(self,
                path: str,
                endpoint: str = ENDPOINT_LOCAL) -> None:
        """Saves the versioned model files to the given path.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the model should be saved. This path
            should indicate the version.
        :param endpoint: The system to which the model files should be saved.
            'local' saves the files to the local filesystem; 's3' saves the
            files to S3, in which case path should be an S3 URL.
        """
        # TODO
