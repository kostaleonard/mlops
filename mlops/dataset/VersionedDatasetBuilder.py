"""Contains the VersionedDatasetBuilder class."""

import numpy as np
from mlops.dataset import ENDPOINT_LOCAL, ENDPOINT_S3
from mlops.dataset.DataProcessor import DataProcessor

STRATEGY_COPY = 'copy'
STRATEGY_LINK = 'link'


class VersionedDatasetBuilder:
    """An object containing all of the components that form a versioned dataset.
    This object is only used to ensure a standard format for datasets stored in
    a dataset archive (such as the local filesystem or S3), and is not meant for
    consumption by downstream models."""

    def __init__(self,
                 dataset_path: str,
                 data_processor: DataProcessor,
                 features: dict[str, np.ndarray],
                 labels: dict[str, np.ndarray]) -> None:
        """TODO"""
        # TODO

    @staticmethod
    def from_path(dataset_path: str,
                  data_processor: DataProcessor) -> \
            'VersionedDatasetBuilder':
        """TODO"""
        features = data_processor.get_preprocessed_features(dataset_path)
        labels = data_processor.get_preprocessed_labels(dataset_path)
        return VersionedDatasetBuilder(
            dataset_path,
            data_processor,
            features,
            labels)

    def publish(self,
                path: str,
                endpoint: str = ENDPOINT_LOCAL,
                dataset_copy_strategy: str = STRATEGY_COPY) -> None:
        """Saves the versioned dataset files to the given path.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the dataset should be saved. This path
            should indicate the version.
        :param endpoint: The system to which the dataset files should be saved.
            'local' saves the files to the local filesystem; 's3' saves the
            files to S3, in which case path should be an S3 URL.
        :param dataset_copy_strategy: The strategy by which to copy the
            original, raw dataset to the published path. The default is 'copy',
            which copies the entire raw dataset to the published path so that it
            can be properly versioned. 'link' will instead create a file
            containing the supplied dataset path; this is desirable if the raw
            dataset is already stored in a versioned repository, and copying
            would create an unnecessary duplicate.
        """
        # TODO
