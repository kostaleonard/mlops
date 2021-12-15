"""Tests pokemon_classification_data_processor.py."""

import pytest
import numpy as np
from mlops.examples.image.classification.pokemon_classification_data_processor \
    import PokemonClassificationDataProcessor, \
    DEFAULT_DATASET_TRAINVALTEST_PATH, DEFAULT_DATASET_PRED_PATH, \
    HEIGHT, WIDTH, CHANNELS, CLASSES
from mlops.examples.image.classification.errors import LabelsNotFoundError

EXPECTED_NUM_TRAINVALTEST = 10
EXPECTED_NUM_TRAIN = 7
EXPECTED_NUM_VAL = 2
EXPECTED_NUM_PRED = 3
PIXEL_MIN = 0
PIXEL_MAX = 255
BULBASAUR_IMG_MEAN = 0.06437409
BULBASAUR_LABEL = {'Grass', 'Poison'}
CHARIZARD_IMG_MEAN = 0.17114125
CHARIZARD_LABEL = {'Fire', 'Flying'}


def test_get_raw_features_and_labels_returns_expected_keys() -> None:
    """Tests that get_raw_features_and_labels returns the expected keys for the
    train/val/test dataset."""
    processor = PokemonClassificationDataProcessor()
    features, labels = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    assert set(features.keys()) == {'X_train', 'X_val', 'X_test'}
    assert set(labels.keys()) == {'y_train', 'y_val', 'y_test'}


def test_get_raw_features_and_labels_pred_raises_error() -> None:
    """Tests that get_raw_features_and_labels raises LabelsNotFoundError when
    called on the prediction directory."""
    processor = PokemonClassificationDataProcessor()
    with pytest.raises(LabelsNotFoundError):
        _ = processor.get_raw_features_and_labels(DEFAULT_DATASET_PRED_PATH)


def test_get_raw_features_and_labels_trainvaltest_correct_split() -> None:
    """Tests that the train/val/test datasets are split into the expected
    sizes."""
    processor = PokemonClassificationDataProcessor()
    features, labels = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    num_examples = sum(map(len, features.values()))
    assert num_examples == EXPECTED_NUM_TRAINVALTEST
    assert len(features['X_train']) == EXPECTED_NUM_TRAIN
    assert len(features['X_val']) == EXPECTED_NUM_VAL
    assert len(features['X_test']) == EXPECTED_NUM_TRAINVALTEST - \
           EXPECTED_NUM_TRAIN - EXPECTED_NUM_VAL
    assert len(features['X_train']) == len(labels['y_train'])
    assert len(features['X_val']) == len(labels['y_val'])
    assert len(features['X_test']) == len(labels['y_test'])


def test_get_raw_features_trainvaltest_returns_expected_keys() -> None:
    """Tests that get_raw_features returns the expected keys {'X_train',
    'X_val', 'X_test} when called on the train/val/test directory.
    """
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    assert set(raw.keys()) == {'X_train', 'X_val', 'X_test'}


def test_get_raw_features_match() -> None:
    """Tests that the features produced by get_raw_features_and_labels and
    get_raw_features are the same features."""
    processor = PokemonClassificationDataProcessor()
    features, _ = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    X_all = np.concatenate((features['X_train'],
                            features['X_val'],
                            features['X_test']))
    features_only = processor.get_raw_features(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    X_all_only = np.concatenate((features_only['X_train'],
                                 features_only['X_val'],
                                 features_only['X_test']))
    X_all.sort(axis=0)
    X_all_only.sort(axis=0)
    assert np.array_equal(X_all, X_all_only)


def test_get_raw_features_pred_returns_expected_keys() -> None:
    """Tests that get_raw_features returns the expected keys {'X_pred'} when
    called on the prediction directory.
    """
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_PRED_PATH)
    assert set(raw.keys()) == {'X_pred'}


def test_get_raw_features_correct_shape() -> None:
    """Tests that get_raw_features returns tensors with the expected shapes."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_PRED_PATH)
    for tensor in raw.values():
        assert tensor.shape[1:] == (HEIGHT, WIDTH, CHANNELS)


def test_get_raw_features_correct_dtype() -> None:
    """Tests that get_raw_features returns tensors with dtype float32."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_PRED_PATH)
    for tensor in raw.values():
        assert tensor.dtype == np.float32


def test_get_raw_features_correct_value_range() -> None:
    """Tests that get_raw_features returns tensors in the range [0, 255]."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert tensor.min() >= 0
        assert tensor.max() <= 1


def test_get_raw_features_no_na() -> None:
    """Tests that get_raw_features returns tensors with no missing values."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert not np.isnan(tensor).any()


def test_get_raw_features_have_multiple_pixel_values() -> None:
    """Tests that the images were loaded correctly by ensuring that more than
    one pixel value exists in the tensors."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert len(np.unique(tensor)) > 1


def test_get_raw_labels_trainvaltest_lengths_match_features() -> None:
    """Tests that all entries in the raw label dictionary have the same number
    of examples as their counterpart features."""
    processor = PokemonClassificationDataProcessor()
    raw_features, raw_labels = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    assert len(raw_features['X_train']) == len(raw_labels['y_train'])
    assert len(raw_features['X_val']) == len(raw_labels['y_val'])
    assert len(raw_features['X_test']) == len(raw_labels['y_test'])


def test_get_raw_labels_correct_tensor_shapes() -> None:
    """Tests that labels are of the correct shape."""
    processor = PokemonClassificationDataProcessor()
    _, raw = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert tensor.shape[1:] == (2,)


def test_get_raw_labels_correct_dtype() -> None:
    """Tests that labels are of type object (string)."""
    processor = PokemonClassificationDataProcessor()
    _, raw = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert tensor.dtype == object


def test_get_raw_labels_valid_classes() -> None:
    """Tests that all raw label classes are valid Pokemon types."""
    processor = PokemonClassificationDataProcessor()
    _, raw = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        for row in tensor:
            assert row[0] in CLASSES
            assert row[1] is None or row[1] in CLASSES


def test_preprocessed_features_same_shape_as_raw() -> None:
    """Tests that the preprocessed features have the same shape as the raw
    features."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_features(tensor)
        assert tensor.shape == preprocessed.shape


def test_preprocess_features_correct_dtype() -> None:
    """Tests that preprocessed features are of dtype float32."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_features(tensor)
        assert preprocessed.dtype == np.float32


def test_preprocess_features_no_na() -> None:
    """Tests that preprocessed features have no missing values."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_features(tensor)
        assert not np.isnan(preprocessed).any()


def test_preprocessed_features_scaled() -> None:
    """Tests that preprocessing scales the features to the range [0, 1]."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_features(tensor)
        assert preprocessed.min() >= 0
        assert preprocessed.max() <= 1


def test_preprocess_labels_correct_shape() -> None:
    """Tests that the preprocessed labels have the correct shape."""
    processor = PokemonClassificationDataProcessor()
    _, raw = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        assert preprocessed.shape == (len(tensor), len(CLASSES))


def test_preprocess_labels_correct_dtype() -> None:
    """Tests that the preprocessed labels are of dtype float32."""
    processor = PokemonClassificationDataProcessor()
    _, raw = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        assert preprocessed.dtype == np.float32


def test_preprocess_labels_no_na() -> None:
    """Tests that the preprocessed labels have no missing values."""
    processor = PokemonClassificationDataProcessor()
    _, raw = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        assert not np.isnan(preprocessed).any()


def test_preprocess_labels_binary() -> None:
    """Tests that the preprocessed labels have values in the set {0, 1}."""
    processor = PokemonClassificationDataProcessor()
    _, raw = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        assert set(np.unique(preprocessed)) == {0, 1}


def test_preprocess_labels_min_one_max_two_classes() -> None:
    """Tests that each preprocessed label has at least one and at most two
    ones indicating the class(es)."""
    processor = PokemonClassificationDataProcessor()
    _, raw = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        row_sums = preprocessed.sum(axis=1)
        assert set(np.unique(row_sums)).union({1, 2}) == {1, 2}


def test_unpreprocess_features_inverts_transformation() -> None:
    """Tests that unpreprocessing the preprocessed features results in the raw
    features."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_features(tensor)
        unpreprocessed = processor.unpreprocess_features(preprocessed)
        assert (unpreprocessed == tensor).all()


def test_unpreprocess_labels_inverts_transformation() -> None:
    """Tests that unpreprocessing the preprocessed labels results in the raw
    labels."""
    processor = PokemonClassificationDataProcessor()
    _, raw = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        unpreprocessed = processor.unpreprocess_labels(preprocessed)
        assert (unpreprocessed == tensor).all()


def test_get_raw_features_and_labels_examples_in_same_order() -> None:
    """Tests that the raw features and raw labels have examples in the same
    order. For example, say X_train[0] is the raw Bulbasaur image; then
    y_train[0] must be the labels for Bulbasaur."""
    processor = PokemonClassificationDataProcessor()
    features, labels = processor.get_raw_features_and_labels(
        DEFAULT_DATASET_TRAINVALTEST_PATH)
    X_all = np.concatenate((features['X_train'],
                            features['X_val'],
                            features['X_test']))
    y_all = np.concatenate((labels['y_train'],
                            labels['y_val'],
                            labels['y_test']))
    bulbasaur_idx = None
    for idx in range(len(X_all)):
        if np.isclose(X_all[idx].mean(), BULBASAUR_IMG_MEAN):
            bulbasaur_idx = idx
    assert bulbasaur_idx is not None
    assert set(y_all[bulbasaur_idx]) == BULBASAUR_LABEL
    charizard_idx = None
    for idx in range(len(X_all)):
        if np.isclose(X_all[idx].mean(), CHARIZARD_IMG_MEAN):
            charizard_idx = idx
    assert charizard_idx is not None
    assert set(y_all[charizard_idx]) == CHARIZARD_LABEL


def test_get_valid_prediction_correct_shape() -> None:
    """Tests that the output of get_valid_prediction is of the same shape as
    the input."""
    pred_arr = np.array([[0.8, 0.4, 0.2, 0.6],
                         [0.3, 0.4, 0.1, 0.1]])
    valid = PokemonClassificationDataProcessor.get_valid_prediction(pred_arr)
    assert pred_arr.shape == valid.shape


def test_get_valid_prediction_output_is_binary() -> None:
    """Tests that the output of get_valid_prediction on arbitrary input is
    binary."""
    pred_arr = np.array([[0.8, 0.4, 0.2, 0.6],
                         [0.3, 0.4, 0.1, 0.1],
                         [-1, 5, 2, 0.5]])
    valid = PokemonClassificationDataProcessor.get_valid_prediction(pred_arr)
    assert set(np.unique(valid)) == {0, 1}


def test_get_valid_prediction_chooses_highest() -> None:
    """Tests that get_valid_prediction chooses the highest scores as output."""
    pred_arr = np.array([[0.8, 0.4, 0.2, 0.6],
                         [0.3, 0.4, 0.1, 0.1],
                         [0.9, 0.9, 0.8, 0.8]])
    valid = PokemonClassificationDataProcessor.get_valid_prediction(pred_arr)
    assert valid.tolist() == [[1, 0, 0, 1],
                              [0, 1, 0, 0],
                              [1, 1, 0, 0]]


def test_get_valid_prediction_one_or_two_classes() -> None:
    """Tests that get_valid_prediction returns predictions with one or two
    classes."""
    pred_arr = np.array([[0.8, 0.4, 0.2, 0.6],
                         [0.3, 0.4, 0.1, 0.1],
                         [0.9, 0.9, 0.9, 0.9],
                         [0.1, 0.1, 0.1, 0.1],
                         [2.0, 2.0, 2.0, 2.0]])
    valid = PokemonClassificationDataProcessor.get_valid_prediction(pred_arr)
    row_sums = valid.sum(axis=1)
    assert set(row_sums) == {1, 2}


def test_get_valid_prediction_threshold_only_affects_second_highest() -> None:
    """Tests that the decision threshold only affects the second highest
    prediction value."""
    pred_arr = np.array([[0.8, 0.4, 0.2, 0.6],
                         [0.3, 0.4, 0.1, 0.1],
                         [0.9, 0.8, 0.7, 0.7]])
    valid = PokemonClassificationDataProcessor.get_valid_prediction(
        pred_arr, threshold=0.6)
    assert valid.tolist() == [[1, 0, 0, 1],
                              [0, 1, 0, 0],
                              [1, 1, 0, 0]]
    valid = PokemonClassificationDataProcessor.get_valid_prediction(
        pred_arr, threshold=0.99)
    assert valid.tolist() == [[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [1, 0, 0, 0]]
