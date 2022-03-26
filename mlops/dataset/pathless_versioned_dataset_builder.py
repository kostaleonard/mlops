"""Contains the PathlessVersionedDatasetBuilder class."""

from typing import List, Optional, Dict
import numpy as np
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder, \
    STRATEGY_LINK
from mlops.dataset.pathless_data_processor import PathlessDataProcessor
from mlops.errors import InvalidDatasetCopyStrategyError

DATASET_PATH_DNE = 'pathless'


class PathlessVersionedDatasetBuilder(VersionedDatasetBuilder):
    """Builds a versioned dataset directly from feature and label tensors."""
    # pylint: disable=too-few-public-methods

    def __init__(self,
                 features: Dict[str, np.ndarray],
                 labels: Dict[str, np.ndarray]) -> None:
        """Instantiates the object.

        :param features: The training features.
        :param labels: The training labels.
        """
        processor = PathlessDataProcessor(features, labels)
        super().__init__(DATASET_PATH_DNE, processor)

    def publish(self,
                path: str,
                name: str = 'dataset',
                version: Optional[str] = None,
                dataset_copy_strategy: str = STRATEGY_LINK,
                tags: Optional[List[str]] = None) -> str:
        """Saves the versioned dataset files to the given path. If the path and
        appended version already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        The following files will be created:
            path/version/ (the publication path and version)
                X_train.npy (and other feature tensors by their given names)
                y_train.npy (and other label tensors by their given names)
                data_processor.pkl (DataProcessor object)
                meta.json (metadata)
                raw.tar.bz2 (bz2-zipped directory with the raw dataset files)

        The contents of meta.json will be:
            {
                name: (dataset name)
                version: (dataset version)
                hash: (MD5 hash of all objects apart from data_processor.pkl and
                    meta.json)
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
        :param name: The name of the dataset, e.g., "mnist".
        :param version: A string indicating the dataset version. The version
            should be unique to this dataset. If None, the publication timestamp
            will be used as the version.
        :param dataset_copy_strategy: The strategy by which to copy the
            original, raw dataset to the published path. STRATEGY_COPY
            recursively copies all files and directories from the dataset path
            supplied at instantiation to the published path so that the dataset
            can be properly versioned. STRATEGY_COPY_ZIP is identical in
            behavior, but zips the directory upon completion. STRATEGY_LINK
            will instead create a file 'link.txt' containing the supplied
            dataset path; this is desirable if the raw dataset is already stored
            in a versioned repository, and copying would create an unnecessary
            duplicate.
        :param tags: An optional list of string tags to add to the dataset
            metadata.
        :return: The versioned dataset's publication path.
        """
        # pylint: disable=too-many-arguments
        if dataset_copy_strategy != STRATEGY_LINK:
            raise InvalidDatasetCopyStrategyError(
                'PathlessVersionedDatasetBuilder must use the link strategy.')
        return super().publish(
            path,
            name=name,
            version=version,
            dataset_copy_strategy=STRATEGY_LINK,
            tags=tags)
