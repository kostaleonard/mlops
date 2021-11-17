"""Contains the VersionedDatasetPackage class."""

from typing import Any
import numpy as np
from mlops.dataset import ENDPOINT_LOCAL, ENDPOINT_S3
from mlops.dataset.DataProcessor import DataProcessor


class VersionedDatasetPackage:
    """An object containing all of the components that form a versioned dataset.
    This object is only used to ensure a standard format for datasets stored in
    a dataset archive (such as the local filesystem or S3), and is not meant for
    consumption by downstream models."""

    def __init__(self,
                 version: str,
                 data_processor: DataProcessor,
                 raw_dataset: Any,
                 features_and_labels: dict[str, np.ndarray]) -> None:
        """TODO"""
        # TODO

    @staticmethod
    def from_path(version: str,
                  dataset_path: str,
                  data_processor: DataProcessor,
                  endpoint: str = ENDPOINT_LOCAL) -> 'VersionedDatasetPackage':
        """TODO"""
        # TODO

    def publish(self,
                path: str,
                endpoint: str = ENDPOINT_LOCAL) -> None:
        """Saves the versioned dataset files to the given path.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the dataset should be saved.
        :param endpoint: The system to which the dataset files should be saved.
            'local' saves the files to the local filesystem; 's3' saves the
            files to S3, in which case path should be an S3 URL.
        """
        # TODO
