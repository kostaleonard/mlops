"""Tests versioned_model_builder.py."""

import pytest


def test_publish_appends_explicit_version() -> None:
    """Tests that publish appends the version string to the path."""
    # TODO
    assert False


def test_publish_appends_version_timestamp() -> None:
    """Tests that publish appends the timestamp to the path when no version is
    given."""
    # TODO
    assert False


def test_publish_local_path_creates_expected_files() -> None:
    """Tests that publish on a local path creates the expected
    files/directories on the local filesystem."""
    # TODO
    assert False


@pytest.mark.awstest
def test_publish_s3_path_creates_expected_files() -> None:
    """Tests that publish on an S3 path creates the expected files/directories
    on S3."""
    # TODO
    assert False


def test_publish_local_path_raises_path_already_exists_error() -> None:
    """Tests that publish on a local path that already exists raises a
    PublicationPathAlreadyExistsError."""
    # TODO
    assert False


@pytest.mark.awstest
def test_publish_s3_path_raises_path_already_exists_error() -> None:
    """Tests that publish on an S3 path that already exists raises a
    PublicationPathAlreadyExistsError."""
    # TODO
    assert False


def test_publish_includes_expected_metadata() -> None:
    """Tests that publish creates a file meta.json with the expected
    metadata."""
    # TODO
    assert False


def test_publish_timestamps_match() -> None:
    """Tests that all 3 timestamps match if no version string is supplied:
    metadata.json's version and created_at fields, and the final directory
    of the published path."""
    # TODO
    assert False


def test_publish_accepts_path_with_trailing_slash() -> None:
    """Tests that publish accepts a path with (potentially many) trailing
    slashes and creates the files as if the trailing slashes were absent."""
    # TODO
    assert False


def test_publish_local_with_trailing_slash() -> None:
    """Tests that publishing to a local path with a trailing slash works
    properly."""
    # TODO
    assert False


@pytest.mark.awstest
def test_publish_s3_with_trailing_slash() -> None:
    """Tests that publishing to an S3 path with a trailing slash works
    properly."""
    # TODO
    assert False


def test_same_models_have_same_hashes() -> None:
    """Tests that the hash values from two models that have identical files are
    the same."""
    # TODO
    assert False


def test_different_models_have_different_hashes() -> None:
    """Tests that the hash values from two models that have different files are
    different."""
    # TODO
    assert False


@pytest.mark.awstest
def test_publish_local_and_s3_create_same_model() -> None:
    """Tests that publishing locally or remotely on S3 produces the same model.
    Verifies identity by comparing model hashes."""
    # TODO
    assert False


def test_metadata_includes_training_history() -> None:
    """Tests that the metadata includes the training history dictionary, and
    that it is consistent with the training results."""
    # TODO
    assert False


def test_metadata_includes_training_args() -> None:
    """Tests that the metadata includes any kwargs supplied to the training
    function."""
    # TODO
    assert False


def test_metadata_includes_dataset_link() -> None:
    """Tests that the metadata includes a link to the dataset."""
    # TODO
    assert False


def test_metadata_includes_expected_tags() -> None:
    """Tests that the metadata includes the expected tags."""
    # TODO
    assert False


def test_published_model_performance_matches_trained_model() -> None:
    """Tests that the published model has the same performance as the model
    supplied during object instantiation."""
    # TODO
    assert False
