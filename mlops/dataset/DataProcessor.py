"""Contains the DataProcessor class."""

from typing import Optional
from abc import ABC, abstractmethod
import numpy as np


class DataProcessor(ABC):
    """Transforms a RawDataset into features for downstream
    model training, prediction, etc. Also inverts any preprocessing to transform
    preprocessed data back into raw, real-world values for analysis and
    interpretability."""

    def __init__(self,
                 dataset_path: Optional[str]) -> None:
        """Instantiates the object.

        :param dataset_path: The path to the file or directory on the local
            filesystem containing the dataset. If provided, sets any necessary
            local variables with which data should be normalized or otherwise
            transformed. For example, this value can be provided to extract the
            mean and standard deviation from a column of feature data in the
            training dataset so that, at prediction time, new data can be
            normalized according to the training values. This value may not be
            necessary if normalization can be accomplished without knowledge of
            training data beforehand, e.g., image data simply needs to be
            divided by 255 to ensure that pixel values fall in the interval
            [0, 1].
        """
        if dataset_path:
            self._calibrate(dataset_path)

    def _calibrate(self,
                   dataset_path: str) -> None:
        """Sets any necessary local variables with which data should be
        normalized or otherwise transformed. For example, the mean and standard
        deviation of certain features can be extracted from the training data so
        that, at prediction time, new data can be normalized according to the
        training values.

        :param dataset_path: The path to the file or directory on the local
            filesystem containing the dataset.
        """
        raise NotImplementedError(
            'Subclasses must override this function if calibration is '
            'required.')

    def get_preprocessed_features(self, dataset_path: str) -> \
            dict[str, np.ndarray]:
        """Transforms the raw data at the given file or directory into features
        that can be used by downstream models. The data in the directory may be
        the training/validation/test data, or it may be a batch of user data
        that is intended for prediction, or data in some other format.
        Downstream models can expect the features returned by this function to
        be preprocessed in any way required for model consumption.

        :param dataset_path: The path to the file or directory on the local
            filesystem containing the dataset.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. For example, the training features (value) may be called
            'X_train' (key).
        """
        raw_feature_tensors = self.get_raw_feature_tensors(dataset_path)
        return {name: self.preprocess_features(raw_feature_tensor)
                for name, raw_feature_tensor in raw_feature_tensors.items()}

    def get_preprocessed_labels(self, dataset_path: str) -> \
            dict[str, np.ndarray]:
        """Transforms the raw data at the given file or directory into labels
        that can be used by downstream models. The data in the directory may be
        the training/validation/test data, or it may be a batch of user data
        that is intended for prediction, or data in some other format.
        Downstream models can expect the labels returned by this function to
        be preprocessed in any way required for model consumption.

        :param dataset_path: The path to the file or directory on the local
            filesystem containing the dataset.
        :return: A dictionary whose values are label tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. For example, the training labels (value) may be called
            'y_train' (key).
        """
        raw_label_tensors = self.get_raw_label_tensors(dataset_path)
        return {name: self.preprocess_labels(raw_label_tensor)
                for name, raw_label_tensor in raw_label_tensors.items()}

    @abstractmethod
    def get_raw_feature_tensors(self,
                                dataset_path: str) -> dict[str, np.ndarray]:
        """TODO"""

    @abstractmethod
    def get_raw_label_tensors(self, dataset_path: str) -> dict[str, np.ndarray]:
        """TODO"""

    @abstractmethod
    def preprocess_features(self, raw_feature_tensor: np.ndarray) -> np.ndarray:
        """TODO"""

    @abstractmethod
    def preprocess_labels(self, raw_label_tensor: np.ndarray) -> np.ndarray:
        """TODO"""

    @abstractmethod
    def unpreprocess_features(self, feature_tensor: np.ndarray) -> np.ndarray:
        """TODO"""

    @abstractmethod
    def unpreprocess_labels(self, label_tensor: np.ndarray) -> np.ndarray:
        """TODO"""
