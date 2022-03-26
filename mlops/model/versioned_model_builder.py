"""Contains the VersionedModelBuilder class."""
# pylint: disable=no-name-in-module

import os
import json
from tempfile import TemporaryDirectory
from typing import Optional, List
from datetime import datetime
from s3fs import S3FileSystem
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.training_config import TrainingConfig
from mlops.hashing.hashing import get_hash_local, get_hash_s3


class VersionedModelBuilder:
    """An object containing all of the components that form a versioned
    model."""
    # pylint: disable=too-few-public-methods

    def __init__(self,
                 versioned_dataset: VersionedDataset,
                 model: Model,
                 training_config: Optional[TrainingConfig] = None) -> None:
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
        :param training_config: (Optional) The model's training configuration.
        """
        self.versioned_dataset = versioned_dataset
        self.model = model
        if training_config:
            self.training_config = training_config
        else:
            empty_history = History()
            empty_train_args = {}
            self.training_config = TrainingConfig(empty_history,
                                                  empty_train_args)

    def publish(self,
                path: str,
                name: str = 'model',
                version: Optional[str] = None,
                tags: Optional[List[str]] = None) -> str:
        """Saves the versioned model files to the given path. If the path and
        appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        The following files will be created:
            path/version/ (the publication path and version)
                model.h5 (the saved model)
                meta.json (metadata)

        The contents of meta.json will be:
            {
                name: (model name)
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
        :param name: The name of the model, e.g., "vgg16".
        :param version: A string indicating the model version. The version
            should be unique to this model. If None, the publication timestamp
            will be used as the version.
        :param tags: An optional list of string tags to add to the model
            metadata.
        :return: The versioned model's publication path.
        """
        timestamp = datetime.now().isoformat()
        if not version:
            version = timestamp
        if not tags:
            tags = []
        publication_path = os.path.join(path.rstrip('/'), version)
        model_path = os.path.join(publication_path, 'model.h5')
        metadata_path = os.path.join(publication_path, 'meta.json')
        metadata = {
            'name': name,
            'version': version,
            'hash': 'TDB',
            'dataset': self.versioned_dataset.path,
            'history': self.training_config.history.history,
            'train_args': self.training_config.train_args,
            'created_at': timestamp,
            'tags': tags}
        if path.startswith('s3://'):
            return self._publish_s3(publication_path,
                                    model_path,
                                    metadata_path,
                                    metadata)
        return self._publish_local(publication_path,
                                   model_path,
                                   metadata_path,
                                   metadata)

    def _publish_local(self,
                       publication_path: str,
                       model_path: str,
                       metadata_path: str,
                       metadata: dict) -> str:
        """Saves the versioned model files to the given local path. See
        publish() for more detailed information.

        :param publication_path: The local path to which to publish the model.
        :param model_path: The path to which the model is saved.
        :param metadata_path: The path to which the metadata is saved.
        :param metadata: Model metadata.
        :return: The versioned model's publication path.
        """
        files_to_hash = set()
        # Create publication path.
        # pylint: disable=protected-access
        VersionedDatasetBuilder._make_publication_path_local(publication_path)
        # Save model.
        self.model.save(model_path)
        files_to_hash.add(model_path)
        # Save metadata.
        hash_digest = get_hash_local(files_to_hash)
        metadata['hash'] = hash_digest
        with open(metadata_path, 'w', encoding='utf-8') as outfile:
            outfile.write(json.dumps(metadata))
        return publication_path

    def _publish_s3(self,
                    publication_path: str,
                    model_path: str,
                    metadata_path: str,
                    metadata: dict) -> str:
        """Saves the versioned model files to the given S3 path. See publish()
        for more detailed information.

        :param publication_path: The S3 path to which to publish the model.
        :param model_path: The path to which the model is saved.
        :param metadata_path: The path to which the metadata is saved.
        :param metadata: Model metadata.
        :return: The versioned model's publication path.
        """
        fs = S3FileSystem()
        files_to_hash = set()
        # Create publication path.
        # pylint: disable=protected-access
        VersionedDatasetBuilder._make_publication_path_s3(publication_path, fs)
        # Save model.
        with TemporaryDirectory() as tmp_dir:
            self.model.save(f'{tmp_dir}/model.h5')
            fs.put(f'{tmp_dir}/model.h5', model_path)
        files_to_hash.add(model_path)
        # Save metadata.
        hash_digest = get_hash_s3(files_to_hash)
        metadata['hash'] = hash_digest
        with fs.open(metadata_path, 'w', encoding='utf-8') as outfile:
            outfile.write(json.dumps(metadata))
        return publication_path
