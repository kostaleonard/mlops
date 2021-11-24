"""Tests __init__.py."""

from mlops.errors import PublicationPathAlreadyExistsError


def test_publication_path_alread_exists_error_extends_io_error() -> None:
    """Tests that PublicationPathAlreadyExistsError extends IOError."""
    err = PublicationPathAlreadyExistsError()
    assert isinstance(err, IOError)
