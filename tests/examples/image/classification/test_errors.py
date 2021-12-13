"""Tests errors.py."""

from mlops.examples.image.classification.errors import LabelsNotFoundError


def test_labels_not_found_error_extends_file_not_found_error() -> None:
    """Tests that LabelsNotFoundError extends FileNotFoundError."""
    err = LabelsNotFoundError()
    assert isinstance(err, FileNotFoundError)
