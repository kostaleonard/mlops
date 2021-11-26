"""Tests __init__.py."""

from mlops.errors import PublicationPathAlreadyExistsError


def test_publication_path_error_extends_file_exists_error() -> None:
    """Tests that PublicationPathAlreadyExistsError extends FileExistsError."""
    err = PublicationPathAlreadyExistsError()
    assert isinstance(err, FileExistsError)
