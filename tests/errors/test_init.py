"""Tests __init__.py."""

from mlops.errors import PublicationPathAlreadyExistsError, \
    InvalidDatasetCopyStrategyError


def test_publication_path_error_extends_file_exists_error() -> None:
    """Tests that PublicationPathAlreadyExistsError extends FileExistsError."""
    err = PublicationPathAlreadyExistsError()
    assert isinstance(err, FileExistsError)


def test_invalid_dataset_copy_strategy_error_extends_value_error() -> None:
    """Tests that InvalidDatasetCopyStrategyError extends ValueError."""
    err = InvalidDatasetCopyStrategyError()
    assert isinstance(err, ValueError)
