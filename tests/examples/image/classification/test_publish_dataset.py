"""Tests publish_dataset.py."""

import os
import shutil
from mlops.examples.image.classification.publish_dataset import publish_dataset

TEST_PUBLICATION_PATH_LOCAL = "/tmp/test_publish_dataset/datasets"


def test_publish_dataset_creates_files() -> None:
    """Tests that publish_dataset creates the dataset files."""
    try:
        shutil.rmtree(TEST_PUBLICATION_PATH_LOCAL)
    except FileNotFoundError:
        pass
    publish_dataset(TEST_PUBLICATION_PATH_LOCAL)
    assert os.path.exists(TEST_PUBLICATION_PATH_LOCAL)
