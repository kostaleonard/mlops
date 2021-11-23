"""Tests data_processor.py."""
import numpy as np
import pytest
from mlops.dataset import data_processor as dp
from tests.dataset.preset_data_processor import PresetDataProcessor


def test_data_processor_is_abstract() -> None:
    """Tests that the DataProcessor object is abstract."""
    with pytest.raises(TypeError):
        _ = dp.DataProcessor()


def test_data_processor_accepts_local_path() -> None:
    """Tests that the DataProcessor object accepts paths on the local
    filesystem."""
    processor = PresetDataProcessor()
    features = processor.get_preprocessed_features('path/to/my/features')
    assert set(features.keys()) == {'X_train', 'X_val', 'X_test'}
    labels = processor.get_preprocessed_labels('path/to/my/labels')
    assert set(labels.keys()) == {'y_train', 'y_val', 'y_test'}


def test_data_processor_accepts_remote_path() -> None:
    """Tests that the DataProcessor object accepts paths on remote filesystens,
    e.g., S3."""
    processor = PresetDataProcessor()
    features = processor.get_preprocessed_features('s3://path/to/my/features')
    assert set(features.keys()) == {'X_train', 'X_val', 'X_test'}
    labels = processor.get_preprocessed_labels('s3://path/to/my/labels')
    assert set(labels.keys()) == {'y_train', 'y_val', 'y_test'}


def test_raw_feature_keys_match_preprocessed_feature_keys() -> None:
    """Tests that the dictionaries returned by get_preprocessed_features and
    get_raw_feature_tensors have the same keys."""
    processor = PresetDataProcessor()
    features = processor.get_preprocessed_features('path/to/my/features')
    raw_features = processor.get_raw_feature_tensors('path/to/my/features')
    assert set(features.keys()) == set(raw_features.keys())


def test_raw_label_keys_match_preprocessed_label_keys() -> None:
    """Tests that the dictionaries returned by get_preprocessed_labels and
    get_raw_label_tensors have the same keys."""
    processor = PresetDataProcessor()
    labels = processor.get_preprocessed_labels('path/to/my/labels')
    raw_labels = processor.get_raw_label_tensors('path/to/my/labels')
    assert set(labels.keys()) == set(raw_labels.keys())


def test_get_raw_feature_tensors_returns_raw_features() -> None:
    """Tests that get_raw_feature_tensors returns the raw features, before
    preprocessing is applied."""
    processor = PresetDataProcessor()
    features = processor.get_preprocessed_features('path/to/my/features')
    raw_features = processor.get_raw_feature_tensors('path/to/my/features')
    for name in features.keys():
        assert not np.array_equal(raw_features[name], features[name])


def test_get_raw_label_tensors_returns_raw_labels() -> None:
    """Tests that get_raw_feature_labels returns the raw labels, before
    preprocessing is applied."""
    processor = PresetDataProcessor()
    labels = processor.get_preprocessed_labels('path/to/my/labels')
    raw_labels = processor.get_raw_label_tensors('path/to/my/labels')
    for name in labels.keys():
        assert not np.array_equal(raw_labels[name], labels[name])


def test_preprocessed_features_match() -> None:
    """Tests that the result of applying preprocess_features on the raw features
    is the same as the output of get_preprocessed_features."""
    processor = PresetDataProcessor()
    features = processor.get_preprocessed_features('path/to/my/features')
    raw_features = processor.get_raw_feature_tensors('path/to/my/features')
    for name in raw_features:
        manually_preprocessed_features = processor.preprocess_features(
            raw_features[name])
        assert np.array_equal(manually_preprocessed_features, features[name])


def test_preprocessed_labels_match() -> None:
    """Tests that the result of applying preprocess_labels on the raw labels
    is the same as the output of get_preprocessed_labels."""
    processor = PresetDataProcessor()
    labels = processor.get_preprocessed_labels('path/to/my/labels')
    raw_labels = processor.get_raw_label_tensors('path/to/my/labels')
    for name in raw_labels:
        manually_preprocessed_labels = processor.preprocess_labels(
            raw_labels[name])
        assert np.array_equal(manually_preprocessed_labels, labels[name])