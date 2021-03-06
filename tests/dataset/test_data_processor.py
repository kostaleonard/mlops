"""Tests data_processor.py."""

import numpy as np
import pytest
from mlops.dataset.data_processor import DataProcessor
from tests.dataset.preset_data_processor import PresetDataProcessor


@pytest.mark.xfail
def test_data_processor_is_abstract() -> None:
    """Tests that the DataProcessor object is abstract."""
    # This test currently fails because we needed to make DataProcessor
    # concrete as a serialization workaround. See:
    # https://github.com/kostaleonard/mlops/issues/12
    with pytest.raises(TypeError):
        _ = DataProcessor()


def test_data_processor_accepts_local_path() -> None:
    """Tests that the DataProcessor object accepts paths on the local
    filesystem."""
    processor = PresetDataProcessor()
    features, labels = processor.get_preprocessed_features_and_labels(
        "path/to/dataset"
    )
    assert set(features.keys()) == {"X_train", "X_val", "X_test"}
    assert set(labels.keys()) == {"y_train", "y_val", "y_test"}


def test_data_processor_accepts_remote_path() -> None:
    """Tests that the DataProcessor object accepts paths on remote filesystens,
    e.g., S3."""
    processor = PresetDataProcessor()
    features, labels = processor.get_preprocessed_features_and_labels(
        "s3://path/to/my/dataset"
    )
    assert set(features.keys()) == {"X_train", "X_val", "X_test"}
    assert set(labels.keys()) == {"y_train", "y_val", "y_test"}


def test_raw_feature_keys_match_preprocessed_feature_keys() -> None:
    """Tests that the dictionaries returned by get_preprocessed_features and
    get_raw_features have the same keys."""
    processor = PresetDataProcessor()
    features = processor.get_preprocessed_features("path/to/my/features")
    raw_features = processor.get_raw_features("path/to/my/features")
    assert set(features.keys()) == set(raw_features.keys())


def test_raw_label_keys_match_preprocessed_label_keys() -> None:
    """Tests that the dictionaries returned by get_preprocessed_labels and
    get_raw_labels have the same keys."""
    processor = PresetDataProcessor()
    _, labels = processor.get_preprocessed_features_and_labels(
        "path/to/my/dataset"
    )
    _, raw_labels = processor.get_raw_features_and_labels("path/to/my/dataset")
    assert set(labels.keys()) == set(raw_labels.keys())


def test_get_raw_features_returns_raw_features() -> None:
    """Tests that get_raw_features returns the raw features, before
    preprocessing is applied."""
    processor = PresetDataProcessor()
    features = processor.get_preprocessed_features("path/to/my/features")
    raw_features = processor.get_raw_features("path/to/my/features")
    for name in features.keys():
        assert not np.array_equal(raw_features[name], features[name])


def test_get_raw_labels_returns_raw_labels() -> None:
    """Tests that get_raw_labels returns the raw labels, before
    preprocessing is applied."""
    processor = PresetDataProcessor()
    _, labels = processor.get_preprocessed_features_and_labels(
        "path/to/my/dataset"
    )
    _, raw_labels = processor.get_raw_features_and_labels("path/to/my/dataset")
    for name in labels.keys():
        assert not np.array_equal(raw_labels[name], labels[name])


def test_preprocessed_features_match() -> None:
    """Tests that the result of applying preprocess_features on the raw
    features is the same as the output of get_preprocessed_features."""
    processor = PresetDataProcessor()
    features = processor.get_preprocessed_features("path/to/my/features")
    raw_features = processor.get_raw_features("path/to/my/features")
    for name, raw in raw_features.items():
        manually_preprocessed_features = processor.preprocess_features(raw)
        assert np.array_equal(manually_preprocessed_features, features[name])


def test_preprocessed_labels_match() -> None:
    """Tests that the result of applying preprocess_labels on the raw labels
    is the same as the output of get_preprocessed_labels."""
    processor = PresetDataProcessor()
    _, labels = processor.get_preprocessed_features_and_labels(
        "path/to/my/dataset"
    )
    _, raw_labels = processor.get_raw_features_and_labels("path/to/my/dataset")
    for name, raw in raw_labels.items():
        manually_preprocessed_labels = processor.preprocess_labels(raw)
        assert np.array_equal(manually_preprocessed_labels, labels[name])


def test_get_raw_features_and_labels_features_match() -> None:
    """Tests that get_raw_features_and_labels and get_raw_features have
    matching features."""
    processor = PresetDataProcessor()
    features, _ = processor.get_raw_features_and_labels("path/to/dataset")
    features_only = processor.get_raw_features("path/to/my/features")
    assert set(features.keys()) == set(features_only.keys())
    for name, arr in features.items():
        assert np.array_equal(arr, features_only[name])


def test_get_preprocessed_features_and_labels_features_match() -> None:
    """Tests that get_preprocessed_features_and_labels and
    get_preprocessed_features have matching features."""
    processor = PresetDataProcessor()
    features, _ = processor.get_preprocessed_features_and_labels(
        "path/to/dataset"
    )
    features_only = processor.get_preprocessed_features("path/to/my/features")
    assert set(features.keys()) == set(features_only.keys())
    for name in features:
        assert np.array_equal(features[name], features_only[name])
