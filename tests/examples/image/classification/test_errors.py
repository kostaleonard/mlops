"""Tests errors.py."""

from mlops.examples.image.classification.errors import (
    LabelsNotFoundError,
    NoModelPathsSuppliedError,
)


def test_labels_not_found_error_extends_file_not_found_error() -> None:
    """Tests that LabelsNotFoundError extends FileNotFoundError."""
    err = LabelsNotFoundError()
    assert isinstance(err, FileNotFoundError)


def test_no_model_paths_supplied_error_extends_value_error() -> None:
    """Tests that NoModelPathsSuppliedError extends ValueError."""
    err = NoModelPathsSuppliedError()
    assert isinstance(err, ValueError)
