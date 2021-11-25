"""Contains the VersionedDatasetBuilder class."""

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
                dataset_copy_strategy: str = STRATEGY_COPY) -> None:
        """Saves the versioned dataset files to the given path. If the path
        already exists, this operation will raise a
        PublicationPathAlreadyExistsError.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the dataset should be saved. This path
            should indicate the version. An S3 path should be a URL of the form
            "s3://bucket-name/path/to/file.txt".
        :param dataset_copy_strategy: The strategy by which to copy the
            original, raw dataset to the published path. The default is
            STRATEGY_COPY, which recursively copies all files and directories
            from the dataset path supplied at instantiation to the published
            path so that the dataset can be properly versioned. STRATEGY_LINK
            will instead create a file containing the supplied dataset path;
            this is desirable if the raw dataset is already stored in a
            versioned repository, and copying would create an unnecessary
            duplicate.
        """
        # TODO
