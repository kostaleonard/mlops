"""Contains the DoubledPresetDataProcessor class."""

from typing import Dict, Tuple
import numpy as np
from tests.dataset.preset_data_processor import PresetDataProcessor


class DoubledPresetDataProcessor(PresetDataProcessor):
    """Processes a preset dataset, with no file I/O; doubles tensor values."""

    def get_raw_features_and_labels(self, dataset_path: str) -> \
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Returns doubled preset raw feature and label tensors.

        :param dataset_path: Unused
        :return: A 2-tuple of the features dictionary and labels dictionary,
            with matching keys and ordered tensors.
        """
        # See #49 for why we can't use super().
        features, labels = PresetDataProcessor.get_raw_features_and_labels(
            self, dataset_path)
        for name, tensor in features.items():
            features[name] = 2 * tensor
        return features, labels

    def get_raw_features(self,
                         dataset_path: str) -> Dict[str, np.ndarray]:
        """Returns the preset raw features.

        :param dataset_path: Unused.
        :return: A dictionary containing the following entries.
            'X_train': The training features. 70% of the dataset.
            'X_val': The validation features. 20% of the dataset.
            'X_test': The test features. 10% of the dataset.
        """
        # See #49 for why we can't use super().
        features = PresetDataProcessor.get_raw_features(self, dataset_path)
        for name, tensor in features.items():
            features[name] = 2 * tensor
        return features
