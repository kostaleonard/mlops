"""Contains the DataProcessor class."""

from typing import Any
import numpy as np
from mlops.dataset.RawDataset import RawDataset


# TODO abstract class
# TODO this object should allow dataset_path to also point to an S3 bucket.
class DataProcessor:
    """Transforms a RawDataset into features for downstream
    model training, prediction, etc. Also inverts any preprocessing to transform
    preprocessed data back into raw, real-world values for analysis and
    interpretability."""

    def __init__(self,
                 raw_dataset: RawDataset | None) -> None:
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
        TODO update
        """
        if raw_dataset:
            self._calibrate(raw_dataset)

    def _calibrate(self,
                   raw_dataset: RawDataset) -> None:
        """Sets any necessary local variables with which data should be
        normalized or otherwise transformed. For example, the mean and standard
        deviation of certain features can be extracted from the training data so
        that, at prediction time, new data can be normalized according to the
        training values.

        :param dataset_path: The path to the file or directory on the local
            filesystem containing the dataset.
        TODO update
        """
        raise NotImplementedError(
            'Subclasses must override this function if calibration is '
            'required.')

    def get_preprocessed_features(self, raw_dataset: RawDataset) -> \
            dict[str, np.ndarray]:
        """Transforms the raw data at the given file or directory into features
        and labels that can be used by downstream models. The data in the
        directory may be the training/validation/test data, or it may be a batch
        of user data that is intended for prediction, or data in some other
        format. Downstream models can expect the features and labels of this
        function to be preprocessed in any way required for model consumption.

        TODO param
        :return: A dictionary whose values are feature and label tensors and
            whose corresponding keys are the names by which those tensors should
            be referenced. For example, the training features (value) may be
            called "X_train" (key), and the training labels (value) may be
            called "y_train" (value).
        """
        # TODO update docstring
        raw_feature_tensors = self.get_raw_feature_tensors(raw_dataset)
        return {name: self.preprocess_features(raw_feature_tensor)
                for name, raw_feature_tensor in raw_feature_tensors.items()}

    def get_preprocessed_labels(self, raw_dataset: RawDataset) -> \
            dict[str, np.ndarray]:
        """TODO"""
        raw_label_tensors = self.get_raw_label_tensors(raw_dataset)
        return {name: self.preprocess_labels(raw_label_tensor)
                for name, raw_label_tensor in raw_label_tensors.items()}

    # TODO abstract method
    def get_raw_feature_tensors(self, raw_dataset: RawDataset) -> \
            dict[str, np.ndarray]:
        """TODO"""

    # TODO abstract method
    def get_raw_label_tensors(self, raw_dataset: RawDataset) -> \
            dict[str, np.ndarray]:
        """TODO"""

    # TODO abstract method
    def preprocess_features(self, raw_feature_tensor: np.ndarray) -> np.ndarray:
        # TODO
        pass

    # TODO abstract method
    def preprocess_labels(self, raw_label_tensor: np.ndarray) -> np.ndarray:
        # TODO
        pass

    # TODO abstract method
    def unpreprocess_features(self, feature_tensor: np.ndarray) -> np.ndarray:
        # TODO
        pass

    # TODO abstract method
    def unpreprocess_labels(self, label_tensor: np.ndarray) -> np.ndarray:
        # TODO
        pass
