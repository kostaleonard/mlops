"""Trains a new model on the Pokemon classification task."""

import os
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.errors import PublicationPathAlreadyExistsError
from mlops.examples.image.classification.pokemon_classifcation_data_processor \
    import PokemonClassificationDataProcessor, \
    DEFAULT_DATASET_TRAINVALTEST_PATH, DEFAULT_DATASET_PRED_PATH

VERSION = 'v1'
PUBLICATION_PATH_LOCAL = os.path.join('datasets', 'pokemon')
PUBLICATION_PATH_S3 = 's3://kosta-mlops/datasets/pokemon'
TAGS = ['image', 'classification']

# TODO test script


def publish_dataset(publication_path: str) -> None:
    """Builds and publishes the dataset."""
    processor = PokemonClassificationDataProcessor()
    builder = VersionedDatasetBuilder(DEFAULT_DATASET_TRAINVALTEST_PATH,
                                      processor)
    try:
        builder.publish(publication_path, VERSION, tags=TAGS)
    except PublicationPathAlreadyExistsError:
        pass


def main() -> None:
    """Runs the program."""
    publish_dataset(PUBLICATION_PATH_LOCAL)
    dataset = VersionedDataset(os.path.join(PUBLICATION_PATH_LOCAL, VERSION))
    print(len(dataset.X_train))
    print(dataset.y_train)
    print(dataset.data_processor.unpreprocess_labels(dataset.y_train))


if __name__ == '__main__':
    main()
