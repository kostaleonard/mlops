"""Tests pathless_data_processor.py."""

import numpy as np
from mlops.dataset.pathless_data_processor import PathlessDataProcessor

PRESET_RAW_FEATURES = np.array([
    [10, 20, 30, 40],
    [0, 20, 40, 50],
    [10, 20, 20, 60],
    [20, 20, 50, 70],
    [10, 20, 10, 80],
    [10, 20, 60, 90],
    [10, 20, 0, 100],
    [30, 20, 70, 110],
    [10, 20, -10, 120],
    [-10, 20, 30, 130]
])
PRESET_RAW_LABELS = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])


def test_get_raw_features_and_labels_returns_presets() -> None:
    """Tests that get_raw_features_and_labels returns the preset values."""
    processor = PathlessDataProcessor(PRESET_RAW_FEATURES, PRESET_RAW_LABELS)
    features, labels = processor.get_raw_features_and_labels('dne')
    assert set(features.keys()) == {'X_train'}
    assert set(labels.keys()) == {'y_train'}
    assert np.array_equal(features['X_train'], PRESET_RAW_FEATURES)
    assert np.array_equal(labels['y_train'], PRESET_RAW_LABELS)


def test_get_raw_features_returns_presets() -> None:
    """Tests that get_raw_features returns the preset values."""
    processor = PathlessDataProcessor(PRESET_RAW_FEATURES, PRESET_RAW_LABELS)
    features = processor.get_raw_features('dne')
    assert set(features.keys()) == {'X_train'}
    assert np.array_equal(features['X_train'], PRESET_RAW_FEATURES)


def test_preprocess_features_is_identity_function() -> None:
    """Tests that preprocess_features is the identity function."""
    processor = PathlessDataProcessor(PRESET_RAW_FEATURES, PRESET_RAW_LABELS)
    preprocessed = processor.preprocess_features(PRESET_RAW_FEATURES)
    assert np.array_equal(preprocessed, PRESET_RAW_FEATURES)


def test_preprocess_labels_is_identity_function() -> None:
    """Tests that preprocess_labels is the identity function."""
    processor = PathlessDataProcessor(PRESET_RAW_FEATURES, PRESET_RAW_LABELS)
    preprocessed = processor.preprocess_labels(PRESET_RAW_LABELS)
    assert np.array_equal(preprocessed, PRESET_RAW_LABELS)
