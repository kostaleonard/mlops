"""Loads a VersionedModel and uses it to run prediction on unseen data."""

import os
import numpy as np
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.versioned_model import VersionedModel
from mlops.examples.image.classification.pokemon_classification_data_processor \
    import PokemonClassificationDataProcessor, DEFAULT_DATASET_PRED_PATH
from mlops.examples.image.classification.publish_dataset import \
    DATASET_PUBLICATION_PATH_LOCAL, DATASET_VERSION
from mlops.examples.image.classification.train_model import \
    MODEL_PUBLICATION_PATH_LOCAL


def model_evaluate(dataset: VersionedDataset,
                   model: VersionedModel) -> float:
    """Returns the model's loss on the test dataset.

    :param dataset: The dataset.
    :param model: The model.
    :return: The model's loss on the test dataset.
    """
    return model.model.evaluate(x=dataset.X_test,
                                y=dataset.y_test)


def model_predict(path: str,
                  dataset: VersionedDataset,
                  model: VersionedModel) -> np.ndarray:
    """Returns the model's unpreprocessed predictions on the data located at the
    given path.

    :param path: The path to the data on which to run prediction. The format of
        the data files must be consistent with what the dataset's data processor
        expects.
    :param dataset: The dataset.
    :param model: The model.
    :return: The model's unpreprocessed predictions on the data located at the
        given path.
    """
    features = dataset.data_processor.get_preprocessed_features(path)
    raw_predictions = model.model.predict(features['X_pred'])
    valid_predictions = \
        PokemonClassificationDataProcessor.get_valid_prediction(raw_predictions)
    return dataset.data_processor.unpreprocess_labels(valid_predictions)


def get_best_model() -> VersionedModel:
    """Returns the versioned model with the best performance on the validation
    dataset.

    :return: The versioned model with the best performance on the validation
        dataset.
    """
    # TODO raise error if no models found
    # TODO get actual best model
    model_filenames = os.listdir(MODEL_PUBLICATION_PATH_LOCAL)
    first_model_path = os.path.join(MODEL_PUBLICATION_PATH_LOCAL,
                                    model_filenames[0])
    # TODO remove print
    print(f'Using model: {first_model_path}')
    return VersionedModel(first_model_path)


def main() -> None:
    """Runs the program."""
    dataset = VersionedDataset(os.path.join(DATASET_PUBLICATION_PATH_LOCAL,
                                            DATASET_VERSION))
    model = get_best_model()
    test_err = model_evaluate(dataset, model)
    print(f'Test error: {test_err:.3f}')
    predictions = model_predict(DEFAULT_DATASET_PRED_PATH, dataset, model)
    print(f'Predictions:\n{predictions}')


if __name__ == '__main__':
    main()
