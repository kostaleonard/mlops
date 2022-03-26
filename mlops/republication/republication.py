"""Contains functions for republishing versioned objects."""

import os
import shutil
from s3fs import S3FileSystem
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder


def republish(versioned_object_path: str,
              republication_path: str,
              version: str) -> str:
    """Saves the versioned dataset files to the given path. If the path and
    appended version already exists, this operation will raise a
    PublicationPathAlreadyExistsError.

    :param versioned_object_path: The path to the published versioned object.
    :param republication_path: The path, either on the local filesystem or in a
        cloud store such as S3, to which the object should be saved. The
        version will be appended to this path as a subdirectory. An S3 path
        should be a URL of the form "s3://bucket-name/path/to/dir". It is
        recommended to use this same path to publish all datasets, since it
        will prevent the user from creating two different datasets with the
        same version.
    :param version: The version of the versioned object.
    :return: The versioned object's republication path.
    """
    if republication_path.startswith('s3://'):
        return _republish_to_s3(
            versioned_object_path, republication_path, version)
    return _republish_to_local(
        versioned_object_path, republication_path, version)


def _republish_to_local(versioned_object_path: str,
                        republication_path: str,
                        version: str) -> str:
    """Saves the versioned dataset files to the given path. If the path and
    appended version already exists, this operation will raise a
    PublicationPathAlreadyExistsError.

    :param versioned_object_path: The path to the published versioned object.
    :param republication_path: The local path to which to publish the versioned
        object.
    :param version: The version of the versioned object.
    :return: The versioned object's republication path.
    """
    # pylint: disable=protected-access
    publication_path = os.path.join(republication_path, version)
    VersionedDatasetBuilder._make_publication_path_local(publication_path)
    if versioned_object_path.startswith('s3://'):
        fs = S3FileSystem()
        fs.get(versioned_object_path, publication_path, recursive=True)
    else:
        shutil.copytree(
            versioned_object_path, publication_path, dirs_exist_ok=True)
    return publication_path


def _republish_to_s3(versioned_object_path: str,
                     republication_path: str,
                     version: str) -> str:
    """Saves the versioned dataset files to the given path. If the path and
    appended version already exists, this operation will raise a
    PublicationPathAlreadyExistsError.

    :param versioned_object_path: The path to the published versioned object.
    :param republication_path: The S3 path to which to publish the versioned
        object.
    :param version: The version of the versioned object.
    :return: The versioned object's republication path.
    """
    # pylint: disable=protected-access
    publication_path = os.path.join(republication_path, version)
    fs = S3FileSystem()
    VersionedDatasetBuilder._make_publication_path_s3(publication_path, fs)
    if versioned_object_path.startswith('s3://'):
        dataset_path_no_prefix = versioned_object_path.replace('s3://', '', 1)
        copy_path_no_prefix = publication_path.replace('s3://', '', 1)
        for current_path, _, filenames in fs.walk(versioned_object_path):
            outfile_prefix = current_path.replace(dataset_path_no_prefix,
                                                  copy_path_no_prefix, 1)
            for filename in filenames:
                infile_path = os.path.join(current_path,
                                           filename)
                outfile_path = os.path.join(outfile_prefix,
                                            filename)
                fs.copy(infile_path, outfile_path)
    else:
        fs.put(versioned_object_path, publication_path, recursive=True)
    return publication_path
