"""Contains the PokemonClassificationDataProcessor class."""

import os
import numpy as np
from mlops.dataset.invertible_data_processor import InvertibleDataProcessor

# TODO should these be defined in the model training script?
DEFAULT_DATASET_TRAINVALTEST_PATH = os.path.join('sample_data', 'pokemon',
                                                 'trainvaltest')
DEFAULT_DATASET_PRED_PATH = os.path.join('sample_data', 'pokemon', 'pred')
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
CLASSES = ['Normal', 'Fire', 'Water', 'Grass', 'Electric', 'Ice', 'Fighting',
           'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost',
           'Dark', 'Dragon', 'Steel', 'Fairy']
HEIGHT = 120
WIDTH = 120
CHANNELS = 3


class PokemonClassificationDataProcessor(InvertibleDataProcessor):
    """Transforms the pokemon dataset at sample_data/pokemon into features and
    labels for classification."""

    def get_raw_features(self, dataset_path: str) -> dict[str, np.ndarray]:
        """Returns the raw feature tensors from the dataset path. The raw
        features are how training/validation/test as well as prediction data
        enter the data pipeline. Raw features are tensors of shape
        m x h x w x c, where m is the number of images, h is the image height,
        w is the image width, and c is the number of channels (3 for RGB), with
        all values in the interval [0, 255].

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. The returned keys will be {'X_train', 'X_val', 'X_test'}
            if the directory indicated by dataset_path ends with 'trainvaltest',
            and {'X_pred'} if dataset_path ends with 'pred'.
        """
        # TODO add support for loading dataset from S3
        # TODO

    def get_raw_labels(self, dataset_path: str) -> dict[str, np.ndarray]:
        """Returns the raw label tensors from the dataset path. The raw labels
        are how training/validation/test as well as prediction data enter the
        data pipeline. Raw labels are tensors of shape m, where m is the number
        of examples. All entries are strings from CLASSES indicating 1 or 2
        (if multi-typed) types belonging to the sample. If the sample has 2
        types, then the string will be comma separated as follows:
        'type1,type2'.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :return: A dictionary whose values are label tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. The returned keys will be {'y_train', 'y_val', 'y_test'}
            if the directory indicated by dataset_path ends with 'trainvaltest',
            and {} otherwise (no labels are available).
        """
        # TODO add support for loading dataset from S3
        # TODO

    def preprocess_features(self, raw_feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed feature tensor from the raw tensor. The
        preprocessed features are how training/validation/test as well as
        prediction data are fed into downstream models. The preprocessed tensors
        are of shape m x h x w x c, where m is the number of images, h is the
        image height, w is the image width, and c is the number of channels
        (3 for RGB), with all values in the interval [0, 1].

        :param raw_feature_tensor: The raw features to be preprocessed.
        :return: The preprocessed feature tensor. This tensor is ready for
            downstream model consumption.
        """
        # TODO divide by 255, make sure result is float32

    def preprocess_labels(self, raw_label_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed label tensor from the raw tensor. The
        preprocessed labels are how training/validation/test as well as
        prediction data are fed into downstream models. Preprocessed labels are
        tensors of shape m x k, where m is the number of examples, and k is the
        number of classes, where each of the k-length vectors are binary,
        multi-label encoded for a minimum of 1 and a maximum of 2 entries per
        vector.

        :param raw_label_tensor: The raw labels to be preprocessed.
        :return: The preprocessed label tensor. This tensor is ready for
            downstream model consumption.
        """
        # TODO encode labels

    def unpreprocess_features(self, feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw feature tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model inputs into real-world values.

        :param feature_tensor: The preprocessed features to be inverted.
        :return: The raw feature tensor.
        """
        # TODO

    def unpreprocess_labels(self, label_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw label tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model outputs into real-world values.

        :param label_tensor: The preprocessed labels to be inverted.
        :return: The raw label tensor.
        """
        # TODO
