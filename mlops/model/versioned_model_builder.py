"""Contains the VersionedModelBuilder class."""
# pylint: disable=no-name-in-module

from typing import Optional
from tensorflow.keras.models import Model
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.training_config import TrainingConfig


class VersionedModelBuilder:
    """An object containing all of the components that form a versioned
    model."""

    def __init__(self,
                 versioned_dataset: VersionedDataset,
                 model: Model,
                 training_config: TrainingConfig) -> None:
        """Instantiates the object.

        :param versioned_dataset: The versioned dataset object with which the
            model was trained/validated/tested. Used to preprocess new data at
            prediction time.
        :param model: The trained Keras model. This is the exact version of the
            model that will be saved; if you intend to keep the best weights and
            not the ones with which the model finished training, ensure that you
            set the model's weights to those desired (this can be done easily
            using the ModelCheckpoint callback or a custom callback that stores
            the best weights in memory).
        :param training_config: The model's training configuration.
        """
        # TODO

    def publish(self,
                path: str,
                version: Optional[str] = None,
                tags: Optional[list[str]] = None) -> None:
        """Saves the versioned model files to the given path. If the path and
        appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        The following files will be created:
            path/version/ (the publication path and version)
                model.h5 (the saved model)
                meta.json (metadata)

        The contents of meta.json will be:
            {
                version: (model version)
                hash: (MD5 hash of all objects apart from meta.json)
                dataset: (the link to the dataset used during training)
                history: (the training history dictionary)
                train_args: (the training arguments dictionary)
                created_at: (timestamp)
                tags: (optional list of tags)
            }

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the model should be saved. The version
            will be appended to this path as a subdirectory. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir". It is
            recommended to use this same path to publish all models, since it
            will prevent the user from creating two different models with the
            same version.
        :param version: A string indicating the model version. The version
            should be unique to this model. If None, the publication timestamp
            will be used as the version.
        :param tags: An optional list of string tags to add to the model
            metadata.
        """
        # TODO

    # TODO mirror VersionedDatasetBuilder interface
