"""Contains the InvertibleDataProcessor class."""

from abc import abstractmethod
import numpy as np
from mlops.dataset.data_processor import DataProcessor


class InvertibleDataProcessor(DataProcessor):
    """A DataProcessor that can invert any preprocessing to transform
    preprocessed data back into raw, real-world values for analysis and
    interpretability."""

    @abstractmethod
    def unpreprocess_features(self, feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw feature tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model inputs into real-world values.

        :param feature_tensor: The preprocessed features to be inverted.
        :return: The raw feature tensor.
        """

    @abstractmethod
    def unpreprocess_labels(self, label_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw label tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model outputs into real-world values.

        :param label_tensor: The preprocessed labels to be inverted.
        :return: The raw label tensor.
        """
