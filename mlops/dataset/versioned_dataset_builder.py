"""Contains the VersionedDatasetBuilder class."""

from typing import Optional, Callable
from mlops.dataset.data_processor import DataProcessor
from mlops.versioning.path_version_extractor import get_version_from_last_entry

STRATEGY_COPY = 'copy'
STRATEGY_LINK = 'link'


class VersionedDatasetBuilder:
    """An object containing all of the components that form a versioned dataset.
    This object is only used to ensure a standard format for datasets stored in
    a dataset archive (such as the local filesystem or S3), and is not meant for
    consumption by downstream models."""

    def __init__(self,
                 dataset_path: str,
                 data_processor: DataProcessor) -> None:
        """Instantiates the object. Features and labels will be extracted from
        the dataset path using the DataProcessor object. When this object is
        published, the feature and label tensors will be saved with '.h5' as the
        suffix. For example, if a tensor is named 'X_train', it will be saved as
        'X_train.h5'.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :param data_processor: The DataProcessor object with which the features
            and labels were generated. This object is saved so that properly
            formatted features can be generated at prediction time, and so that
            features and labels can be "unpreprocessed" to match their original
            representations.
        """
        self.dataset_path = dataset_path
        self.data_processor = data_processor
        self.features = data_processor.get_preprocessed_features(dataset_path)
        self.labels = data_processor.get_preprocessed_labels(dataset_path)

    def publish(self,
                path: str,
                version: str,
                dataset_copy_strategy: str = STRATEGY_COPY,
                tags: Optional[list[str]] = None) -> None:
        """Saves the versioned dataset files to the given path. If the path and
            appended version already exists, this operation will raise a
            PublicationPathAlreadyExistsError.

        The following files will be created:
            path/version/ (the publication path and version)
                X_train.h5 (and other feature tensors by their given names)
                y_train.h5 (and other label tensors by their given names)
                data_processor.pkl (DataProcessor object)
                meta.json (metadata)
                raw/ (non-empty directory with the raw dataset files)
                    ...

        The contents of meta.json will be:
            {
                version: (dataset version)
                hash: (MD5 hash of all objects apart from meta.json)
                created_at: (timestamp)
                tags: (optional list of tags)
            }

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the dataset should be saved. The version
            will be appended to this path as a subdirectory. An S3 path
            should be a URL of the form "s3://bucket-name/path/to/dir". It is
            recommended to use this same path to publish all datasets, since it
            will prevent the user from creating two different datasets with the
            same version.
        :param version: A string indicating the dataset version. The version
            should be unique to this dataset.
        :param dataset_copy_strategy: The strategy by which to copy the
            original, raw dataset to the published path. The default is
            STRATEGY_COPY, which recursively copies all files and directories
            from the dataset path supplied at instantiation to the published
            path so that the dataset can be properly versioned. STRATEGY_LINK
            will instead create a file containing the supplied dataset path;
            this is desirable if the raw dataset is already stored in a
            versioned repository, and copying would create an unnecessary
            duplicate.
        :param tags: An optional list of string tags to add to the dataset
            metadata.
        """
        # TODO update docstring
        # TODO VersionedModelBuilder should match this interface
