"""Contains the VersionedModelBuilder class."""
# pylint: disable=no-name-in-module

from typing import Optional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from mlops.dataset.versioned_dataset import VersionedDataset


class VersionedModelBuilder:
    """An object containing all of the components that form a versioned
    model."""

    def __init__(self,
                 versioned_dataset: VersionedDataset,
                 model: Model,
                 history: History) -> None:
        """Instantiates the object.

        :param versioned_dataset: The versioned dataset object with which the
            model was trained/validated/tested. Used to preprocess new data at
            prediction time.
        :param model: The trained Keras model.
        :param history: The model's training history.
        """

    def publish(self,
                path: str,
                version: str,
                tags: Optional[list[str]] = None) -> None:
        """Saves the versioned model files to the given path.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the model should be saved. This path
            should indicate the version for easier user reference. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir". If the
            path already exists, this operation will raise a
            PublicationPathAlreadyExistsError.
        :param version: A string indicating the model version. The version
            string can be in any format, but should be unique, descriptive, and
            consistent with project standards.
        :param tags: An optional list of string tags to add to the model
            metadata.
        """
        # TODO

    # TODO mirror VersionedDatasetBuilder interface
