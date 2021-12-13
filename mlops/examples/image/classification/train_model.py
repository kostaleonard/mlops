"""Trains a new model on the Pokemon classification task."""

import os
from typing import Optional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model_builder import VersionedModelBuilder
from mlops.examples.image.classification.publish_dataset import \
    DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION
from mlops.examples.image.classification.pokemon_classifcation_data_processor \
    import DEFAULT_DATASET_PRED_PATH

MODEL_PUBLICATION_PATH_LOCAL = os.path.join('models', 'pokemon')
MODEL_PUBLICATION_PATH_S3 = 's3://kosta-mlops/models/pokemon'

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
                **fit_kwargs) -> History:
    """Trains the model on the dataset and returns the training history object.

    :param model: The Keras Model to be trained.
    :param dataset: The input dataset.
    :param use_wandb: If True, sync the run with wandb.
    :param fit_kwargs: Keyword arguments to be passed to model.fit().
    """
    # TODO how does this capture training config? Maybe it should be JSON of all hyperparams, and the History should be just a part of the JSON


def publish_model(model: Model,
                  dataset: VersionedDataset,
                  history: History,
                  publication_path: str,
                  tags: Optional[list[str]] = None) -> None:
    """Publishes the model to the path on the local or remote filesystem."""
    builder = VersionedModelBuilder(dataset, model, history)
    builder.publish(publication_path, tags=tags)


def main() -> None:
    """Runs the program."""
    dataset = VersionedDataset(os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                            DATASET_VERSION))
    print(len(dataset.X_train))
    print(dataset.y_train)
    print(dataset.data_processor.unpreprocess_labels(dataset.y_train))
    print(dataset.data_processor.get_raw_features(
        DEFAULT_DATASET_PRED_PATH)['X_pred'].shape)


if __name__ == '__main__':
    main()
