"""Tests pathless_versioned_dataset_builder.py."""

import os
import shutil
import pytest
import numpy as np
from mlops.dataset.versioned_dataset_builder import (
    STRATEGY_COPY_ZIP,
    STRATEGY_COPY,
)
from mlops.dataset.pathless_versioned_dataset_builder import (
    PathlessVersionedDatasetBuilder,
)
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.errors import InvalidDatasetCopyStrategyError

PRESET_RAW_FEATURES = np.array(
    [
        [10, 20, 30, 40],
        [0, 20, 40, 50],
        [10, 20, 20, 60],
        [20, 20, 50, 70],
        [10, 20, 10, 80],
        [10, 20, 60, 90],
        [10, 20, 0, 100],
        [30, 20, 70, 110],
        [10, 20, -10, 120],
        [-10, 20, 30, 130],
    ]
)
PRESET_RAW_LABELS = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
TEST_PUBLICATION_PATH_LOCAL = (
    "/tmp/test_pathless_versioned_dataset_builder/datasets"
)


def _remove_test_directories_local() -> None:
    """Removes the local test directories."""
    try:
        shutil.rmtree(TEST_PUBLICATION_PATH_LOCAL)
    except FileNotFoundError:
        pass


def test_publish_creates_valid_versioned_dataset() -> None:
    """Tests that a VersionedDataset can be loaded from the published files."""
    # pylint: disable=no-member
    _remove_test_directories_local()
    builder = PathlessVersionedDatasetBuilder(
        {"X_train": PRESET_RAW_FEATURES}, {"y_train": PRESET_RAW_LABELS}
    )
    builder.publish(TEST_PUBLICATION_PATH_LOCAL, version="v1")
    dataset_path = os.path.join(TEST_PUBLICATION_PATH_LOCAL, "v1")
    dataset = VersionedDataset(dataset_path)
    assert hasattr(dataset, "X_train")
    assert hasattr(dataset, "y_train")
    assert np.array_equal(dataset.X_train, PRESET_RAW_FEATURES)
    assert np.array_equal(dataset.y_train, PRESET_RAW_LABELS)


def test_publish_invalid_strategy_raises_error() -> None:
    """Tests that publish raises an error when dataset_copy_strategy is
    anything other than link."""
    builder = PathlessVersionedDatasetBuilder(
        {"X_train": PRESET_RAW_FEATURES}, {"y_train": PRESET_RAW_LABELS}
    )
    for strategy in STRATEGY_COPY_ZIP, STRATEGY_COPY:
        with pytest.raises(InvalidDatasetCopyStrategyError):
            builder.publish(
                TEST_PUBLICATION_PATH_LOCAL,
                version="v1",
                dataset_copy_strategy=strategy,
            )
