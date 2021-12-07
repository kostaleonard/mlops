"""Tests pokemon_classification_data_processor.py."""

import numpy as np
from mlops.examples.image.classification.pokemon_classifcation_data_processor \
    import PokemonClassificationDataProcessor, \
    DEFAULT_DATASET_TRAINVALTEST_PATH, DEFAULT_DATASET_PRED_PATH, \
    HEIGHT, WIDTH, CHANNELS, CLASSES

EXPECTED_NUM_TRAINVALTEST = 10
EXPECTED_NUM_TRAIN = 7
EXPECTED_NUM_VAL = 2
EXPECTED_NUM_PRED = 3
PIXEL_MIN = 0
PIXEL_MAX = 255


def test_get_raw_features_trainvaltest_returns_expected_keys() -> None:
    """Tests that get_raw_features returns the expected keys {'X_train',
    'X_val', 'X_test'} when called on the train/val/test directory."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    assert set(raw.keys()) == {'X_train', 'X_val', 'X_test'}


def test_get_raw_features_pred_returns_expected_keys() -> None:
    """Tests that get_raw_features returns the expected keys {'X_pred'} when
    called on the prediction directory.
    """
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_PRED_PATH)
    assert set(raw.keys()) == {'X_pred'}


def test_get_raw_features_trainvaltest_correct_split() -> None:
    """Tests that the train/val/test datasets are split into the expected
    sizes."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    num_examples = sum(map(len, raw.values()))
    assert num_examples == EXPECTED_NUM_TRAINVALTEST
    assert raw['X_train'] == EXPECTED_NUM_TRAIN
    assert raw['X_val'] == EXPECTED_NUM_VAL
    assert raw['X_test'] == EXPECTED_NUM_TRAINVALTEST - EXPECTED_NUM_TRAIN - \
           EXPECTED_NUM_VAL


def test_get_raw_features_correct_tensor_shapes() -> None:
    """Tests that get_raw_features returns tensors of the expected shapes."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert tensor.shape[1:] == (HEIGHT, WIDTH, CHANNELS)


def test_get_raw_features_correct_dtype() -> None:
    """Tests that get_raw_features returns tensors with dtype uint8."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert tensor.dtype == np.uint8


def test_get_raw_features_correct_value_range() -> None:
    """Tests that get_raw_features returns tensors in the range [0, 255]."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert tensor.min() >= PIXEL_MIN
        assert tensor.max() <= PIXEL_MAX


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


def test_get_raw_labels_trainvaltest_returns_expected_keys() -> None:
    """Tests that get_raw_labels returns the expected keys {'y_train', 'y_val',
    'y_test'} when called on the train/val/test directory."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    assert set(raw.keys()) == {'y_train', 'y_val', 'y_test'}


def test_get_raw_labels_pred_returns_empty_dict() -> None:
    """Tests that get_raw_labels returns the empty dict when called on the
    prediction directory (no labels exist for prediction)."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_PRED_PATH)
    assert raw == {}


def test_get_raw_labels_trainvaltest_lengths_match_features() -> None:
    """Tests that all entries in the raw label dictionary have the same number
    of examples as their counterpart features."""
    processor = PokemonClassificationDataProcessor()
    raw_features = processor.get_raw_features(DEFAULT_DATASET_TRAINVALTEST_PATH)
    raw_labels = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    assert len(raw_features['X_train']) == len(raw_labels['y_train'])
    assert len(raw_features['X_val']) == len(raw_labels['y_val'])
    assert len(raw_features['X_test']) == len(raw_labels['y_test'])


def test_get_raw_labels_correct_tensor_shapes() -> None:
    """Tests that get_raw_labels returns tensors of the correct shape."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert len(tensor.shape) == 1


def test_get_raw_labels_correct_dtype() -> None:
    """Tests that get_raw_labels returns tensors of type object (string)."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert tensor.dtype == np.unicode
    # TODO
    assert False


def test_get_raw_labels_min_one_max_two_classes() -> None:
    """Tests that all raw labels have at a minimum one and a maximum two
    classes."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        for row in tensor:
            assert row.count(',') in {0, 1}


def test_get_raw_labels_valid_classes() -> None:
    """Tests that all raw label classes are valid Pokemon types."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        for row in tensor:
            labels = row.split(',')
            for label in labels:
                assert label in CLASSES


def test_get_raw_labels_no_na() -> None:
    """Tests that there are no missing values in the raw labels."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        assert not np.isnan(tensor).any()


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
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        assert preprocessed.shape == (len(tensor), len(CLASSES))


def test_preprocess_labels_correct_dtype() -> None:
    """Tests that the preprocessed labels are of dtype float32."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        assert preprocessed.dtype == np.float32


def test_preprocess_labels_no_na() -> None:
    """Tests that the preprocessed labels have no missing values."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        assert not np.isnan(preprocessed).any()


def test_preprocess_labels_binary() -> None:
    """Tests that the preprocessed labels have values in the set {0, 1}."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        assert set(np.unique(preprocessed)) == {0, 1}


def test_preprocess_labels_min_one_max_two_classes() -> None:
    """Tests that each preprocessed label has at least one and at most two
    ones indicating the class(es)."""
    processor = PokemonClassificationDataProcessor()
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        row_sums = preprocessed.sum(axis=1)
        assert set(np.unique(row_sums)) == {1, 2}


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
    raw = processor.get_raw_labels(DEFAULT_DATASET_TRAINVALTEST_PATH)
    for tensor in raw.values():
        preprocessed = processor.preprocess_labels(tensor)
        unpreprocessed = processor.unpreprocess_labels(preprocessed)
        assert (unpreprocessed == tensor).all()
