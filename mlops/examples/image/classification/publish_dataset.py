"""Publishes a new dataset to the local or remote filesystem. This script should
be run any time the data processor changes."""

import os
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder
from mlops.examples.image.classification.pokemon_classifcation_data_processor \
    import PokemonClassificationDataProcessor, \
    DEFAULT_DATASET_TRAINVALTEST_PATH

VERSION = 'v1'
PUBLICATION_PATH_LOCAL = os.path.join('datasets', 'pokemon')
PUBLICATION_PATH_S3 = 's3://kosta-mlops/datasets/pokemon'
TAGS = ['image', 'classification']

# TODO test script


def publish_dataset(publication_path: str) -> None:
    """Builds and publishes the dataset."""
    # TODO update docstring
    processor = PokemonClassificationDataProcessor()
    builder = VersionedDatasetBuilder(DEFAULT_DATASET_TRAINVALTEST_PATH,
                                      processor)
    builder.publish(publication_path, VERSION, tags=TAGS)


def main() -> None:
    """Runs the program."""
    publish_dataset(PUBLICATION_PATH_LOCAL)


if __name__ == '__main__':
    main()

