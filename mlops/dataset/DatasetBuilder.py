"""Contains the DatasetBuilder class."""

from mlops.dataset.DataProcessor import DataProcessor

ENDPOINT_LOCAL = 'local'
ENDPOINT_S3 = 's3'


class DatasetBuilder:
    """Builds the dataset from a file or directory and publishes it to S3."""

    def __init__(self,
                 dataset_path: str,
                 data_processor: DataProcessor) -> None:
        """Instantiates the object.

        :param dataset_path: The path to the file or directory on the local
            filesystem containing the dataset. This path will be passed to
            data_processor to create the feature and label tensors.
        :param data_processor: The DataProcessor object that transforms files or
            directories into feature and label tensors.
        """
        # TODO

    def publish(self,
                path: str,
                endpoint: str = ENDPOINT_LOCAL) -> None:
        """Saves the dataset files to the given path.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, to which the dataset should be saved.
        :param endpoint: The system to which the dataset files should be saved.
            'local' saves the files to the local filesystem; 's3' saves the
            files to S3, in which case path should be an S3 URL.
        """
        # TODO
