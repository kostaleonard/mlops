"""Contains hashing functions."""

from typing import Collection
import hashlib
from functools import partial
from s3fs import S3FileSystem

CHUNK_SIZE = 2 ** 20


def get_hash_local(files_to_hash: Collection) -> str:
    """Returns the MD5 hex digest string from hashing the content of all the
    given files on the local filesystem. The files are sorted before hashing
    so that the process is reproducible.

    :param files_to_hash: A collection of paths to files whose contents
        should be hashed.
    :return: The MD5 hex digest string from hashing the content of all the
        given files.
    """
    hash_md5 = hashlib.md5()
    for filename in sorted(files_to_hash):
        with open(filename, 'rb') as infile:
            for chunk in iter(partial(infile.read, CHUNK_SIZE), b''):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_hash_s3(files_to_hash: Collection) -> str:
    """Returns the MD5 hex digest string from hashing the content of all the
    given files in S3. The files are sorted before hashing so that the
    process is reproducible.

    :param files_to_hash: A collection of paths to files whose contents
        should be hashed.
    :return: The MD5 hex digest string from hashing the content of all the
        given files.
    """
    hash_md5 = hashlib.md5()
    fs = S3FileSystem()
    for filename in sorted(files_to_hash):
        with fs.open(filename, 'rb') as infile:
            for chunk in iter(partial(infile.read, CHUNK_SIZE), b''):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()
