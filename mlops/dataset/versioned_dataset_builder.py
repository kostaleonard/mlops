"""Contains the VersionedDatasetBuilder class."""

import numpy as np
from mlops import ENDPOINT_LOCAL, ENDPOINT_S3
from mlops.dataset.data_processor import DataProcessor

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
        """Instantiates the object.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :param data_processor: The DataProcessor object with which the features
            and labels were generated. This object is saved so that properly
            formatted features can be generated at prediction time, and so that
            features and labels can be "unpreprocessed" to match their original
            representations.
        :param features: A dictionary containing the preprocessed features with
            which the model will be trained, validated, tested, etc. The keys of
            this dictionary are the tensor names, and the values are the tensors
            themselves. When this object is published, these tensors will be
            saved with '.h5' as the suffix. For example, if a tensor is named
            'X_train', it will be saved as 'X_train.h5'.
        :param labels: A dictionary containing the preprocessed labels with
            which the model will be trained, validated, tested, etc. The keys of
            this dictionary are the tensor names, and the values are the tensors
            themselves. When this object is published, these tensors will be
            saved with '.h5' as the suffix. For example, if a tensor is named
            'y_train', it will be saved as 'y_train.h5'.
        """
        # TODO

    @staticmethod
    def from_path(dataset_path: str,
                  data_processor: DataProcessor) -> 'VersionedDatasetBuilder':
        """Returns a new instantiation of this class using the given path and
        DataProcessor object. Features and labels will be extracted from the
        dataset path using the DataProcessor object.

        :param dataset_path: The path to the file or directory on the local
            filesystem containing the dataset.
        :param data_processor: The DataProcessor object with which the features
            and labels were generated. This object is saved so that properly
            formatted features can be generated at prediction time, and so that
            features and labels can be "unpreprocessed" to match their original
            representations.
        :return: A new instantiation of this class using the given path and
            DataProcessor object.
        """
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
        """Saves the versioned dataset files to the given path. If the path
        already exists, this operation will raise an IOError.

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
