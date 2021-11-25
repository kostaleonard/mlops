"""Tests versioned_dataset_builder.py."""


def test_publish_local_path_creates_expected_files() -> None:
    """Tests that publish on a local path creates the expected
    files/directories on the local filesystem.
    """
    # TODO
    assert False


def test_publish_s3_path_creates_expected_files() -> None:
    """Tests that publish on an S3 path creates the expected files/directories
    on the local filesystem."""
    # TODO
    assert False


def test_publish_local_path_raises_path_already_exists_error() -> None:
    """Tests that publish on a local path that already exists raises a
    PublicationPathAlreadyExistsError."""
    # TODO
    assert False


def test_publish_s3_path_raises_path_already_exists_error() -> None:
    """Tests that publish on an S3 path that already exists raises a
    PublicationPathAlreadyExistsError."""
    # TODO
    assert False


def test_publish_copies_raw_dataset() -> None:
    """Tests that publish copies the entire raw dataset when the copy strategy
    is STRATEGY_COPY."""
    # TODO
    assert False


def test_publish_includes_raw_dataset_link() -> None:
    """Tests that publish includes a link to the raw dataset when the copy
    strategy is STRATEGY_LINK."""
    # TODO
    assert False


def test_publish_includes_expected_metadata() -> None:
    """Tests that publish creates a file meta.json with the expected metadata.
    """
    # TODO
    assert False
