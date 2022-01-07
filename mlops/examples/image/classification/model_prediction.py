"""Loads a VersionedModel and uses it to run prediction on unseen data."""

import os
from typing import Collection
import json
import numpy as np
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model import VersionedModel
from mlops.examples.image.classification.pokemon_classification_data_processor \
    import PokemonClassificationDataProcessor, DEFAULT_DATASET_PRED_PATH
from mlops.examples.image.classification.publish_dataset import \
    DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION
from mlops.examples.image.classification.train_model import \
    MODEL_PUBLICATION_PATH_LOCAL
from mlops.examples.image.classification.errors import NoModelPathsSuppliedError


def model_evaluate(dataset: VersionedDataset,
                   model: VersionedModel) -> float:
    """Returns the model's loss on the test dataset.

    :param dataset: The dataset.
    :param model: The model.
    :return: The model's loss on the test dataset.
    """
    return model.model.evaluate(x=dataset.X_test,
                                y=dataset.y_test)


def model_predict(features: np.ndarray,
                  dataset: VersionedDataset,
                  model: VersionedModel) -> np.ndarray:
    """Returns the model's unpreprocessed predictions on the data located at the
    given path.

    :param features: The preprocessed features on which to run prediction.
    :param dataset: The dataset.
    :param model: The model.
    :return: The model's unpreprocessed predictions on the data located at the
        given path.
    """
    raw_predictions = model.model.predict(features)
    valid_predictions = \
        PokemonClassificationDataProcessor.get_valid_prediction(raw_predictions)
    return dataset.data_processor.unpreprocess_labels(valid_predictions)


def get_best_model(model_paths: Collection) -> VersionedModel:
    """Returns the versioned model with the best performance on the validation
    dataset.

    :param model_paths: The paths to the versioned models to load.
    :return: The versioned model with the best performance on the validation
        dataset.
    """
    if not model_paths:
        raise NoModelPathsSuppliedError
    best_model_path = None
    best_model_val_loss = 0
    for model_path in model_paths:
        meta_path = os.path.join(model_path, 'meta.json')
        with open(meta_path, 'r', encoding='utf-8') as infile:
            contents = json.loads(infile.read())
        model_val_loss = contents['history']['val_loss'][-1]
        if not best_model_path or model_val_loss < best_model_val_loss:
            best_model_path = model_path
            best_model_val_loss = model_val_loss
    return VersionedModel(best_model_path)


def main() -> None:
    """Runs the program."""
    dataset = VersionedDataset(os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                            DATASET_VERSION))
    model_filenames = os.listdir(MODEL_PUBLICATION_PATH_LOCAL)
    model_paths = [os.path.join(MODEL_PUBLICATION_PATH_LOCAL, filename)
                   for filename in model_filenames]
    model = get_best_model(model_paths)
    test_err = model_evaluate(dataset, model)
    print(f'Best model\'s test error: {test_err:.3f}')
    features = dataset.data_processor.get_preprocessed_features(
        DEFAULT_DATASET_PRED_PATH)['X_pred']
    predictions = model_predict(features, dataset, model)
    print(f'Predictions:\n{predictions}')


if __name__ == '__main__':
    main()
