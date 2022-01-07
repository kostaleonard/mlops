"""Trains a new model on the Pokemon classification task."""
# pylint: disable=no-name-in-module

import os
from typing import Optional, Any, List
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, \
    Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from mlops.errors import PublicationPathAlreadyExistsError
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.model.training_config import TrainingConfig
from mlops.examples.image.classification.publish_dataset import \
    publish_dataset, DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION

MODEL_PUBLICATION_PATH_LOCAL = os.path.join('models', 'pokemon', 'versioned')
MODEL_PUBLICATION_PATH_S3 = 's3://kosta-mlops/models/pokemon'
MODEL_CHECKPOINT_FILENAME = os.path.join('models', 'pokemon', 'checkpoints',
                                         'model_best.h5')
TAGS = ['baseline']


def get_baseline_model(dataset: VersionedDataset) -> Model:
    """Returns a new Keras Model for use on the dataset. This model is only a
    baseline; developers should also experiment with custom models in notebook
    environments.

    :param dataset: The input dataset. Used to determine model input and output
        shapes.
    :return: A new Keras Model for use on the dataset.
    """
    model = Sequential()
    # Shape: (None, 120, 120, 3).
    model.add(Conv2D(16, (3, 3), activation='relu',
                     input_shape=dataset.X_train.shape[1:]))
    # Shape: (None, 118, 118, 16).
    model.add(MaxPooling2D((2, 2)))
    # Shape: (None, 59, 59, 16).
    model.add(Conv2D(32, (3, 3), activation='relu'))
    # Shape: (None, 57, 57, 32).
    model.add(MaxPooling2D((2, 2)))
    # Shape: (None, 28, 28, 32).
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # Shape: (None, 26, 26, 64).
    model.add(MaxPooling2D((2, 2)))
    # Shape: (None, 13, 13, 64).
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # Shape: (None, 11, 11, 128).
    model.add(MaxPooling2D((2, 2)))
    # Shape: (None, 5, 5, 128).
    model.add(Conv2D(256, (3, 3), activation='relu'))
    # Shape: (None, 3, 3, 256).
    model.add(Flatten())
    # Shape: (None, 2304).
    model.add(Dense(128, activation='relu'))
    # Shape: (None, 128).
    model.add(Dropout(0.4))
    # Shape: (None, 128).
    model.add(Dense(dataset.y_train.shape[1], activation='sigmoid'))
    # Shape: (None, 18).
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


def train_model(model: Model,
                dataset: VersionedDataset,
                model_checkpoint_filename: Optional[str] = None,
                **fit_kwargs: Any) -> TrainingConfig:
    """Trains the model on the dataset and returns the training configuration
    object.

    :param model: The Keras Model to be trained.
    :param dataset: The input dataset.
    :param model_checkpoint_filename: If supplied, saves model checkpoints to
        the specified path.
    :param fit_kwargs: Keyword arguments to be passed to model.fit().
    :return: The training configuration.
    """
    callbacks = []
    if model_checkpoint_filename:
        checkpoint_callback = ModelCheckpoint(
            model_checkpoint_filename,
            save_best_only=True)
        callbacks.append(checkpoint_callback)
        model_dir = os.path.dirname(model_checkpoint_filename)
        os.makedirs(model_dir, exist_ok=True)
    history = model.fit(x=dataset.X_train,
                        y=dataset.y_train,
                        validation_data=(dataset.X_val, dataset.y_val),
                        callbacks=callbacks,
                        **fit_kwargs)
    return TrainingConfig(history, fit_kwargs)


def publish_model(model: Model,
                  dataset: VersionedDataset,
                  training_config: TrainingConfig,
                  publication_path: str,
                  tags: Optional[List[str]] = None) -> str:
    """Publishes the model to the path on the local or remote filesystem.

    :param model: The model to be published, with the exact weights desired for
        publication (the user needs to set the weights to the best found during
        training if that is what they desire).
    :param dataset: The input dataset.
    :param training_config: The training configuration.
    :param publication_path: The path to which the model will be published.
    :param tags: Optional tags for the published model.
    :return: The versioned model's publication path.
    """
    builder = VersionedModelBuilder(dataset, model, training_config)
    return builder.publish(publication_path, tags=tags)


def main() -> None:
    """Runs the program."""
    try:
        dataset_path = publish_dataset(DATASET_PUBLICATION_PATH_LOCAL)
    except PublicationPathAlreadyExistsError:
        dataset_path = os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                    DATASET_VERSION)
    dataset = VersionedDataset(dataset_path)
    model = get_baseline_model(dataset)
    training_config = train_model(
        model,
        dataset,
        model_checkpoint_filename=MODEL_CHECKPOINT_FILENAME,
        epochs=5,
        batch_size=4)
    publish_model(
        model,
        dataset,
        training_config,
        MODEL_PUBLICATION_PATH_LOCAL,
        tags=TAGS)


if __name__ == '__main__':
    main()
