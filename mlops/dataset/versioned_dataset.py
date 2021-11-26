"""Contains the VersionedDataset class."""


class VersionedDataset:
    """Represents a versioned dataset."""

    def __init__(self, path: str) -> None:
        """Instantiates the object.

        :param path: The path, either on the local filesystem or in a cloud
            store such as S3, from which the dataset should be loaded. An S3
            path should be a URL of the form
            "s3://bucket-name/path/to/dir".
        """
        # TODO

    # TODO public accessor methods?

    # TODO search (by version, by tag, by timestamp)
