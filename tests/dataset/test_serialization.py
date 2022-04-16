"""Tests DataProcessor object serialization during dataset publication."""

import os
import shutil
from typing import Dict
import numpy as np
from mlops.dataset.data_processor import DataProcessor
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder
from mlops.dataset.versioned_dataset import VersionedDataset

TEST_DATASET_PATH = "sample_data/pokemon/trainvaltest"
TEST_PUBLICATION_PATH = "/tmp/test_serialization/datasets"


class DataProcessorThatWillChange(DataProcessor):
    """A DataProcessor subclass that will change, simulating a user redefining
    how data should enter the model pipeline."""

    def get_raw_features_and_labels(
        self, dataset_path: str
    ) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
        """Returns dummy features and labels

        :param dataset_path: Unused.
        :return: A 2-tuple of the features dictionary and labels dictionary,
            with matching keys and ordered tensors.
        """
        return ({"X": np.array([1, 2, 3])}, {"y": np.array([1, 2, 3])})

    def get_raw_features(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """Returns dummy features.

        :param dataset_path: Unused.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. For example, the training features (value) may be
            called 'X_train' (key).
        """
        return {"X": np.array([1, 2, 3])}

    def preprocess_features(
        self, raw_feature_tensor: np.ndarray
    ) -> np.ndarray:
        """Returns features multiplied by 2.

        :param raw_feature_tensor: The raw features to be preprocessed.
        :return: The preprocessed feature tensor. This tensor is ready for
            downstream model consumption.
        """
        return 2 * raw_feature_tensor

    def preprocess_labels(self, raw_label_tensor: np.ndarray) -> np.ndarray:
        """Returns labels multiplied by 2.

        :param raw_label_tensor: The raw labels to be preprocessed.
        :return: The preprocessed label tensor. This tensor is ready for
            downstream model consumption.
        """
        return 2 * raw_label_tensor


def _redefine_class() -> None:
    """Redefines DataProcessorThatWillChange."""
    # pylint: disable=redefined-outer-name
    # pylint: disable=global-statement
    # pylint: disable=global-variable-undefined
    # pylint: disable=global-variable-not-assigned
    # pylint: disable=invalid-name
    # pylint: disable=unused-variable
    global DataProcessorThatWillChange

    class DataProcessorThatWillChange(DataProcessor):
        """A DataProcessor subclass that will change, simulating a user
        redefining how data should enter the model pipeline."""

        def get_raw_features_and_labels(
            self, dataset_path: str
        ) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
            """Returns dummy features and labels

            :param dataset_path: Unused.
            :return: A 2-tuple of the features dictionary and labels
                dictionary, with matching keys and ordered tensors.
            """
            raise ValueError("The new implementation is different.")

        def get_raw_features(self, dataset_path: str) -> Dict[str, np.ndarray]:
            """Returns dummy features.

            :param dataset_path: Unused.
            :return: A dictionary whose values are feature tensors and whose
                corresponding keys are the names by which those tensors should
                be referenced. For example, the training features (value) may
                be called 'X_train' (key).
            """
            raise ValueError("The new implementation is different.")

        def preprocess_features(
            self, raw_feature_tensor: np.ndarray
        ) -> np.ndarray:
            """Returns features multiplied by 2.

            :param raw_feature_tensor: The raw features to be preprocessed.
            :return: The preprocessed feature tensor. This tensor is ready for
                downstream model consumption.
            """
            raise ValueError("The new implementation is different.")

        def preprocess_labels(
            self, raw_label_tensor: np.ndarray
        ) -> np.ndarray:
            """Returns labels multiplied by 2.

            :param raw_label_tensor: The raw labels to be preprocessed.
            :return: The preprocessed label tensor. This tensor is ready for
                downstream model consumption.
            """
            raise ValueError("The new implementation is different.")


def test_serialized_data_processor_uses_original_methods() -> None:
    """Tests that the serialized data processor object obtained during dataset
    publication uses the original methods, not those of the class definition at
    the time of deserialization, which may be different."""
    try:
        shutil.rmtree(TEST_PUBLICATION_PATH)
    except FileNotFoundError:
        pass
    data_processor = DataProcessorThatWillChange()
    builder = VersionedDatasetBuilder(TEST_DATASET_PATH, data_processor)
    builder.publish(TEST_PUBLICATION_PATH, version="v1")
    dataset = VersionedDataset(os.path.join(TEST_PUBLICATION_PATH, "v1"))
    features, labels = dataset.data_processor.get_raw_features_and_labels(
        "dne"
    )
    assert np.array_equal(features["X"], [1, 2, 3])
    assert np.array_equal(labels["y"], [1, 2, 3])
    _redefine_class()
    dataset = VersionedDataset(os.path.join(TEST_PUBLICATION_PATH, "v1"))
    features, labels = dataset.data_processor.get_raw_features_and_labels(
        "dne"
    )
    assert np.array_equal(features["X"], [1, 2, 3])
    assert np.array_equal(labels["y"], [1, 2, 3])
