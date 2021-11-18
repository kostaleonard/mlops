"""Contains the VersionedDataset class."""

from mlops.dataset import ENDPOINT_LOCAL, ENDPOINT_S3


class VersionedDataset:
    """TODO"""

    def __init__(self,
                 path: str,
                 endpoint: str = ENDPOINT_LOCAL) -> None:
        """Instantiates the object.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, from which the dataset should be loaded.
        :param endpoint: The system from which the dataset files should be
            loaded. 'local' loads the files from the local filesystem; 's3'
            loads the files from S3, in which case path should be an S3 URL.
        """
