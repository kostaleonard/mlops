"""Trains a new model on the Pokemon classification task."""

import os
from typing import Optional, Any
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.model.training_config import TrainingConfig
from mlops.examples.image.classification.publish_dataset import \
    DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION

MODEL_PUBLICATION_PATH_LOCAL = os.path.join('models', 'pokemon')
MODEL_PUBLICATION_PATH_S3 = 's3://kosta-mlops/models/pokemon'

TAGS = ['baseline']
# TODO test script


def get_baseline_model(dataset: VersionedDataset) -> Model:
    """Returns a new Keras Model for use on the dataset. This model is only a
    baseline; developers should also experiment with custom models in notebook
    environments.

    :param dataset: The input dataset. Used to determine model input and output
        shapes.
    :return: A new Keras Model for use on the dataset.
    """
    # TODO


def train_model(model: Model,
                dataset: VersionedDataset,
                use_wandb: bool = False,
                **fit_kwargs) -> TrainingConfig:
    """Trains the model on the dataset and returns the training configuration
    object.

    :param model: The Keras Model to be trained.
    :param dataset: The input dataset.
    :param use_wandb: If True, sync the run with wandb.
    :param fit_kwargs: Keyword arguments to be passed to model.fit().
    :return: The training configuration.
    """
    # TODO


def publish_model(model: Model,
                  dataset: VersionedDataset,
                  training_config: TrainingConfig,
                  publication_path: str,
                  tags: Optional[list[str]] = None) -> None:
    """Publishes the model to the path on the local or remote filesystem."""
    # TODO docstring
    builder = VersionedModelBuilder(dataset, model, training_config)
    builder.publish(publication_path, tags=tags)


def main() -> None:
    """Runs the program."""
    dataset = VersionedDataset(os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                            DATASET_VERSION))
    model = get_baseline_model(dataset)
    training_config = train_model(model, dataset, epochs=5, batch_size=4)
    publish_model(model,
                  dataset,
                  training_config,
                  MODEL_PUBLICATION_PATH_LOCAL,
                  tags=TAGS)


if __name__ == '__main__':
    main()
