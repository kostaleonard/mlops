"""Extracts a version string from a path."""


def get_version_from_last_entry(path: str) -> str:
    """Returns the last file/directory name as the version.

    :param path: The path, either on the local filesystem or in a cloud
        store such as S3, to a VersionedDataset or VersionedModel. An S3 path
        should be a URL of the form "s3://bucket-name/path/to/dir". This path
        should indicate the version in the last entry, e.g.,
        "path/to/dataset/v2" (version will be "v2") or
        "s3://bucket-name/path/to/model/1-5/" (version will be "1-5"). Trailing
        slashes will be ignored.
    :return: The version extracted from the last file/directory in the path.
    """
    # TODO test paths with no slashes, trailing slashes (possibly multiple), local/s3, from root dir, empty string
