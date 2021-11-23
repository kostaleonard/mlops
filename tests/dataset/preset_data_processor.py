"""Contains the PresetDataProcessor class."""

import numpy as np
from tensorflow.keras.utils import to_categorical
from mlops.dataset.invertible_data_processor import InvertibleDataProcessor

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
TRAIN_END = int(len(PRESET_RAW_FEATURES) * 0.7)
VAL_EMD = TRAIN_END + int(len(PRESET_RAW_FEATURES) * 0.2)
SCALING_FACTOR = 10


class PresetDataProcessor(InvertibleDataProcessor):
    """Processes a preset dataset, with no file I/O."""

    def get_raw_features(self,
                         dataset_path: str) -> dict[str, np.ndarray]:
        """Returns the preset raw features.

        :param dataset_path: Unused.
        :return: A dictionary containing the following entries.
            'X_train': The training features. 70% of the dataset.
            'X_val': The validation features. 20% of the dataset.
            'X_test': The test features. 10% of the dataset.
        """
        return {'X_train': PRESET_RAW_FEATURES[:TRAIN_END],
                'X_val': PRESET_RAW_FEATURES[TRAIN_END:VAL_EMD],
                'X_test': PRESET_RAW_FEATURES[VAL_EMD:]}

    def get_raw_labels(self, dataset_path: str) -> dict[str, np.ndarray]:
        """Returns the preset raw label tensors.

        :param dataset_path: Unused.
        :return: A dictionary containing the following entries.
            'y_train': The training labels. 70% of the dataset.
            'y_val': The validation labels. 20% of the dataset.
            'y_test': The test labels. 10% of the dataset.
        """
        return {'y_train': PRESET_RAW_LABELS[:TRAIN_END],
                'y_val': PRESET_RAW_LABELS[TRAIN_END:VAL_EMD],
                'y_test': PRESET_RAW_LABELS[VAL_EMD:]}

    def preprocess_features(self, raw_feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed feature tensor from the raw tensor.

        :param raw_feature_tensor: The raw features to be preprocessed.
        :return: The preprocessed feature tensor. This tensor is ready for
            downstream model consumption.
        """
        return raw_feature_tensor // SCALING_FACTOR

    def preprocess_labels(self, raw_label_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed label tensor from the raw tensor. Returns
        one-hot encoded labels.

        :param raw_label_tensor: The raw labels to be preprocessed.
        :return: The preprocessed label tensor. This tensor is ready for
            downstream model consumption.
        """
        return to_categorical(raw_label_tensor)

    def unpreprocess_features(self, feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw feature tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model inputs into real-world values.

        :param feature_tensor: The preprocessed features to be inverted.
        :return: The raw feature tensor.
        """
        return feature_tensor * SCALING_FACTOR

    def unpreprocess_labels(self, label_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw label tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model outputs into real-world values.

        :param label_tensor: The preprocessed labels to be inverted.
        :return: The raw label tensor.
        """
        return np.argmax(label_tensor)
