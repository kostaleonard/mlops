"""Publishes a new dataset to the local or remote filesystem. This script should
be run any time the data processor changes."""

import os
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder
from mlops.examples.image.classification.pokemon_classification_data_processor \
    import PokemonClassificationDataProcessor, \
    DEFAULT_DATASET_TRAINVALTEST_PATH

DATASET_VERSION = 'v1'
DATASET_PUBLICATION_PATH_LOCAL = os.path.join('datasets', 'pokemon')
DATASET_PUBLICATION_PATH_S3 = 's3://kosta-mlops/datasets/pokemon'
TAGS = ['image', 'classification']


def publish_dataset(publication_path: str) -> str:
    """Builds and publishes the dataset.

    :param publication_path: The path on the local or remote filesystem to which
        to publish the dataset.
    :return: The versioned dataset's publication path.
    """
    processor = PokemonClassificationDataProcessor()
    builder = VersionedDatasetBuilder(DEFAULT_DATASET_TRAINVALTEST_PATH,
                                      processor)
    return builder.publish(publication_path,
                           version=DATASET_VERSION,
                           tags=TAGS)


def main() -> None:
    """Runs the program."""
    publish_dataset(DATASET_PUBLICATION_PATH_LOCAL)


if __name__ == '__main__':
    main()
