"""Tests pokemon_classification_data_processor.py."""


def test_get_raw_features_trainvaltest_returns_expected_keys() -> None:
    """Tests that get_raw_features returns the expected keys {'X_train',
    'X_val', 'X_test'} when called on the train/val/test directory."""
    # TODO
    assert False


def test_get_raw_features_pred_returns_expected_keys() -> None:
    """Tests that get_raw_features returns the expected keys {'X_pred'} when
    called on the prediction directory.
    """
    # TODO
    assert False


def test_get_raw_features_trainvaltest_correct_split() -> None:
    """Tests that the train/val/test datasets are split into the expected
    sizes."""
    # TODO
    assert False


def test_get_raw_features_correct_tensor_shapes() -> None:
    """Tests that get_raw_features returns tensors of the expected shapes."""
    # TODO
    assert False


def test_get_raw_features_correct_dtype() -> None:
    """Tests that get_raw_features returns tensors with dtype uint8."""
    # TODO
    assert False


def test_get_raw_features_correct_value_range() -> None:
    """Tests that get_raw_features returns tensors in the range [0, 255]."""
    # TODO
    assert False


def test_get_raw_features_no_na() -> None:
    """Tests that get_raw_features returns tensors with no missing values."""
    # TODO
    assert False


def test_get_raw_features_have_multiple_pixel_values() -> None:
    """Tests that the images were loaded correctly by ensuring that more than
    one pixel value exists in the tensors."""
    # TODO
    assert False


def test_get_raw_labels_trainvaltest_returns_expected_keys() -> None:
    """Tests that get_raw_labels returns the expected keys {'y_train', 'y_val',
    'y_test'} when called on the train/val/test directory."""
    # TODO
    assert False


def test_get_raw_labels_pred_returns_empty_dict() -> None:
    """Tests that get_raw_labels returns the empty dict when called on the
    prediction directory (no labels exist for prediction)."""
    # TODO
    assert False


def test_get_raw_labels_correct_tensor_shapes() -> None:
    """Tests that get_raw_labels returns tensors of the correct shape."""
    # TODO
    assert False


def test_get_raw_labels_correct_dtype() -> None:
    """Tests that get_raw_labels returns tensors of type object (string)."""
    # TODO
    assert False


def test_get_raw_labels_min_one_max_two_classes() -> None:
    """Tests that all raw labels have at a minimum one and a maximum two
    classes."""
    # TODO
    assert False


def test_get_raw_labels_valid_classes() -> None:
    """Tests that all raw label classes are valid Pokemon types."""
    # TODO
    assert False
