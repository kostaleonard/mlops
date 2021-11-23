"""Tests invertible_data_processor.py."""

import pytest
import numpy as np
from mlops.dataset import invertible_data_processor as idp
from tests.dataset.preset_data_processor import PresetDataProcessor


def test_invertible_data_processor_is_abstract() -> None:
    """Tests that the InvertibleDataProcessor object is abstract."""
    with pytest.raises(TypeError):
        _ = idp.InvertibleDataProcessor()


def test_unpreprocess_features_returns_raw_features() -> None:
    """Tests that unpreprocess_features inverts feature preprocessing."""
    processor = PresetDataProcessor()
    features = processor.get_preprocessed_features('path/to/my/features')
    raw_features = processor.get_raw_features('path/to/my/features')
    for name in features:
        inverted_features = processor.unpreprocess_features(features[name])
        assert np.array_equal(inverted_features, raw_features[name])


def test_unpreprocess_labels_returns_raw_labels() -> None:
    """Tests that unpreprocess_labels inverts label preprocessing."""
    processor = PresetDataProcessor()
    labels = processor.get_preprocessed_labels('path/to/my/labels')
    raw_labels = processor.get_raw_labels('path/to/my/labels')
    for name in labels:
        inverted_labels = processor.unpreprocess_labels(labels[name])
        assert np.array_equal(inverted_labels, raw_labels[name])
