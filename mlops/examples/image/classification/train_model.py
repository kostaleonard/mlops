"""Trains a new model on the Pokemon classification task."""

import os
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.examples.image.classification.publish_dataset import \
    PUBLICATION_PATH_LOCAL, VERSION
from mlops.examples.image.classification.pokemon_classifcation_data_processor \
    import DEFAULT_DATASET_PRED_PATH

# TODO test script


def main() -> None:
    """Runs the program."""
    dataset = VersionedDataset(os.path.join(PUBLICATION_PATH_LOCAL, VERSION))
    print(len(dataset.X_train))
    print(dataset.y_train)
    print(dataset.data_processor.unpreprocess_labels(dataset.y_train))
    print(dataset.data_processor.get_raw_features(
        DEFAULT_DATASET_PRED_PATH)['X_pred'].shape)


if __name__ == '__main__':
    main()
