"""Contains the DataProcessor class."""

from abc import abstractmethod
from typing import Dict, Tuple
import numpy as np


class DataProcessor:
    """Transforms a raw dataset into features and labels for downstream model
    training, prediction, etc."""

    def get_preprocessed_features_and_labels(self, dataset_path: str) -> \
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Returns the preprocessed feature and label tensors from the dataset
        path. This method is specifically used for the train/val/test sets and
        not input data for prediction, because in some cases the features and
        labels need to be read simultaneously to ensure proper ordering of
        features and labels.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset, specifically
            train/val/test and not prediction data.
        :return: A 2-tuple of the features dictionary and labels dictionary,
            with matching keys and ordered tensors.
        """
        raw_feature_tensors, raw_label_tensors = \
            self.get_raw_features_and_labels(dataset_path)
        features = {name: self.preprocess_features(raw_feature_tensor)
                    for name, raw_feature_tensor in raw_feature_tensors.items()}
        labels = {name: self.preprocess_labels(raw_label_tensor)
                  for name, raw_label_tensor in raw_label_tensors.items()}
        return features, labels

    def get_preprocessed_features(self, dataset_path: str) -> \
            Dict[str, np.ndarray]:
        """Transforms the raw data at the given file or directory into features
        that can be used by downstream models. The data in the directory may be
        the training/validation/test data, or it may be a batch of user data
        that is intended for prediction, or data in some other format.
        Downstream models can expect the features returned by this function to
        be preprocessed in any way required for model consumption.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. For example, the training features (value) may be called
            'X_train' (key).
        """
        raw_feature_tensors = self.get_raw_features(dataset_path)
        return {name: self.preprocess_features(raw_feature_tensor)
                for name, raw_feature_tensor in raw_feature_tensors.items()}

    @abstractmethod
    def get_raw_features_and_labels(self, dataset_path: str) -> \
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Returns the raw feature and label tensors from the dataset path. This
        method is specifically used for the train/val/test sets and not input
        data for prediction, because in some cases the features and labels need
        to be read simultaneously to ensure proper ordering of features and
        labels.

        For example, when handling image data, the raw features would likely be
        tensors of shape m x h x w x c, where m is the number of images, h is
        the image height, w is the image width, and c is the number of channels
        (3 for RGB), with all values in the interval [0, 255]. The raw labels
        may be tensors of shape m, where m is the number of examples, with all
        values in the set {0, ..., k - 1} indicating the class.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset, specifically
            train/val/test and not prediction data.
        :return: A 2-tuple of the features dictionary and labels dictionary,
            with matching keys and ordered tensors.
        """

    @abstractmethod
    def get_raw_features(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """Returns the raw feature tensors from the dataset path. The raw
        features are how training/validation/test as well as prediction data
        enter the data pipeline. For example, when handling image data, the raw
        features would likely be tensors of shape m x h x w x c, where m is the
        number of images, h is the image height, w is the image width, and c is
        the number of channels (3 for RGB), with all values in the interval
        [0, 255].

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. For example, the training features (value) may be called
            'X_train' (key).
        """

    @abstractmethod
    def preprocess_features(self, raw_feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed feature tensor from the raw tensor. The
        preprocessed features are how training/validation/test as well as
        prediction data are fed into downstream models. For example, when
        handling image data, the preprocessed features would likely be tensors
        of shape m x h x w x c, where m is the number of images, h is the image
        height, w is the image width, and c is the number of channels (3 for
        RGB), with all values in the interval [0, 1].

        :param raw_feature_tensor: The raw features to be preprocessed.
        :return: The preprocessed feature tensor. This tensor is ready for
            downstream model consumption.
        """

    @abstractmethod
    def preprocess_labels(self, raw_label_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed label tensor from the raw tensor. The
        preprocessed labels are how training/validation/test as well as
        prediction data are fed into downstream models. For example, in a
        classification task, the preprocessed labels may be tensors of shape
        m x k, where m is the number of examples, and k is the number of
        classes, where each of the k-length vectors are one-hot encoded.

        :param raw_label_tensor: The raw labels to be preprocessed.
        :return: The preprocessed label tensor. This tensor is ready for
            downstream model consumption.
        """
