"""Contains the PathlessDataProcessor class."""

from typing import Dict, Tuple
import numpy as np
from mlops.dataset.data_processor import DataProcessor


class PathlessDataProcessor(DataProcessor):
    """Loads preset features and labels."""

    def __init__(self,
                 features: Dict[str, np.ndarray],
                 labels: Dict[str, np.ndarray]) -> None:
        """Instantiates the object.

        :param features: The training features.
        :param labels: The training labels.
        """
        self.features = features
        self.labels = labels

    def get_raw_features_and_labels(self, dataset_path: str) -> \
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Returns the training features and labels.

        :param dataset_path: Unused
        :return: A 2-tuple of the features dictionary and labels dictionary,
            with matching keys and ordered tensors.
        """
        return self.features, self.labels

    def get_raw_features(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """Returns the training features.

        :param dataset_path: Unused.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. For example, the training features (value) may be called
            'X_train' (key).
        """
        return self.features

    def preprocess_features(self, raw_feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the identity function on the input features.

        :param raw_feature_tensor: The raw features to be preprocessed.
        :return: The preprocessed feature tensor. This tensor is ready for
            downstream model consumption.
        """
        return raw_feature_tensor.copy()

    def preprocess_labels(self, raw_label_tensor: np.ndarray) -> np.ndarray:
        """Returns the the identity function on the input labels.

        :param raw_label_tensor: The raw labels to be preprocessed.
        :return: The preprocessed label tensor. This tensor is ready for
            downstream model consumption.
        """
        return raw_label_tensor.copy()
