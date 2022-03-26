"""Contains the PokemonClassificationDataProcessor class."""

import os
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from matplotlib.image import imread
from mlops.dataset.invertible_data_processor import InvertibleDataProcessor
from mlops.examples.image.classification.errors import LabelsNotFoundError

DEFAULT_DATASET_TRAINVALTEST_PATH = os.path.join('sample_data', 'pokemon',
                                                 'trainvaltest')
DEFAULT_DATASET_PRED_PATH = os.path.join('sample_data', 'pokemon', 'pred')
IMAGES_DIRNAME = 'images'
LABELS_FILENAME = 'pokemon.csv'
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.2
CLASSES = ['Normal', 'Fire', 'Water', 'Grass', 'Electric', 'Ice', 'Fighting',
           'Poison', 'Ground', 'Flying', 'Psychic', 'Bug', 'Rock', 'Ghost',
           'Dark', 'Dragon', 'Steel', 'Fairy']
HEIGHT = 120
WIDTH = 120
CHANNELS = 3
THRESHOLD = 0.5


class PokemonClassificationDataProcessor(InvertibleDataProcessor):
    """Transforms the pokemon dataset at sample_data/pokemon into features and
    labels for classification."""

    def get_raw_features_and_labels(self, dataset_path: str) -> \
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Returns the raw feature and label tensors from the dataset path. This
        method is specifically used for the train/val/test sets and not input
        data for prediction, because in some cases the features and labels need
        to be read simultaneously to ensure proper ordering of features and
        labels.

        Raw features are tensors of shape m x h x w x c, where m is the number
        of images, h is the image height, w is the image width, and c is the
        number of channels (3 for RGB), with all values in the interval
        [0, 1]. Raw labels are tensors of shape m x 2, where m is the number of
        examples. All entries are strings from CLASSES indicating 1 or 2 (if
        multi-typed) types belonging to the sample. Types are not ordered.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset, specifically
            train/val/test and not prediction data.
        :return: A 2-tuple of the features dictionary and labels dictionary,
            with matching keys and ordered tensors.
        """
        # pylint: disable=too-many-locals
        try:
            df = pd.read_csv(os.path.join(dataset_path, LABELS_FILENAME))
        except FileNotFoundError as exc:
            raise LabelsNotFoundError from exc
        X = []
        y = []
        for _, row in df.iterrows():
            name, type1, type2 = row
            full_path = os.path.join(dataset_path,
                                     IMAGES_DIRNAME,
                                     f'{name}.png')
            try:
                # Discard alpha channel.
                tensor = imread(full_path)[:, :, :3]
            except FileNotFoundError:
                # Some pokemon images are jpg--we will not use these.
                tensor = None
            if tensor is not None:
                if not isinstance(type2, str):
                    label = type1, None
                else:
                    label = type1, type2
                X.append(tensor)
                y.append(label)
        X = np.array(X)
        y = np.array(y)
        num_train = int(len(X) * TRAIN_SPLIT)
        num_val = int(len(X) * VAL_SPLIT)
        X_train = X[:num_train]
        y_train = y[:num_train]
        X_val = X[num_train:num_train + num_val]
        y_val = y[num_train:num_train + num_val]
        X_test = X[num_train + num_val:]
        y_test = y[num_train + num_val:]
        return ({'X_train': X_train, 'X_val': X_val, 'X_test': X_test},
                {'y_train': y_train, 'y_val': y_val, 'y_test': y_test})

    def get_raw_features(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """Returns the raw feature tensors from the prediction dataset path. Raw
        features are tensors of shape m x h x w x c, where m is the number of
        images, h is the image height, w is the image width, and c is the number
        of channels (3 for RGB), with all values in the interval [0, 1]. The
        features are already scaled because PNG images load into float32 instead
        of uint8.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. The returned keys will be {'X_train', 'X_val', 'X_test'}
            if the directory indicated by dataset_path ends with 'trainvaltest',
            and {'X_pred'} otherwise.
        """
        X = []
        for filename in os.listdir(os.path.join(dataset_path, IMAGES_DIRNAME)):
            full_path = os.path.join(dataset_path, IMAGES_DIRNAME, filename)
            # Discard alpha channel.
            tensor = imread(full_path)[:, :, :3]
            X.append(tensor)
        X = np.array(X)
        features = {}
        if dataset_path.rstrip('/').endswith('trainvaltest'):
            num_train = int(len(X) * TRAIN_SPLIT)
            num_val = int(len(X) * VAL_SPLIT)
            features['X_train'] = X[:num_train]
            features['X_val'] = X[num_train:num_train + num_val]
            features['X_test'] = X[num_train + num_val:]
        else:
            features['X_pred'] = X
        return features

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
        # PNG images, when loaded by imread, are already scaled into [0, 1].
        return raw_feature_tensor.copy()

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
        preprocessed = []
        for label_strs in raw_label_tensor:
            label_arr = PokemonClassificationDataProcessor._get_label_arr(
                *label_strs)
            preprocessed.append(label_arr)
        return np.array(preprocessed)

    def unpreprocess_features(self, feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw feature tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model inputs into real-world values.

        :param feature_tensor: The preprocessed features to be inverted.
        :return: The raw feature tensor.
        """
        # No preprocessing was necessary.
        return feature_tensor.copy()

    def unpreprocess_labels(self, label_tensor: np.ndarray) -> np.ndarray:
        """Returns the raw label tensor from the preprocessed tensor; inverts
        preprocessing. Improves model interpretability by enabling users to
        transform model outputs into real-world values.

        :param label_tensor: The preprocessed labels to be inverted.
        :return: The raw label tensor.
        """
        unpreprocessed = []
        for label_arr in label_tensor:
            type_strs = PokemonClassificationDataProcessor._get_label_strs(
                label_arr)
            unpreprocessed.append(type_strs)
        return np.array(unpreprocessed)

    @staticmethod
    def _get_label_arr(type1: str, type2: Optional[str]) -> np.ndarray:
        """Returns the label binary array for the given type pair.

        :param type1: The pokemon's first type.
        :param type2: The pokemon's second type, if any (may be None).
        :return: The label for the given type pair; a tensor of shape k where k
            is the number of classes. The tensor will have one 1 if type2 is
            None, otherwise it will have two. The indices of these will be the
            indices of the type names in CLASSES.
        """
        res = [0] * len(CLASSES)
        res[CLASSES.index(type1)] = 1
        if type2:
            res[CLASSES.index(type2)] = 1
        return np.array(res).astype(np.float32)

    @staticmethod
    def _get_label_strs(label_arr: np.ndarray) -> np.ndarray:
        """Returns the label string array for the given type pair.

        :param label_arr: The pokemon's binary label array.
        :return: The string type labels for the given binary label array; a
            tensor of shape 2. The tensor will have one 1 string entry and 1 NA
            if the pokemon is single-typed; otherwise the tensor will have two
            string entries.
        """
        type_strs = []
        for idx in np.where(label_arr == 1)[0]:
            type_strs.append(CLASSES[idx])
        if len(type_strs) == 1:
            type_strs.append(None)
        return np.array(type_strs)

    @staticmethod
    def get_valid_prediction(pred_arr: np.ndarray,
                             threshold: float = THRESHOLD) -> np.ndarray:
        """Returns a valid binary prediction from the raw prediction tensor. A
        valid prediction has one or two 1s, and all other entries are 0. The
        highest value in the prediction array is automatically converted to a 1,
        and the second-highest is converted to a 1 if the value is higher than
        the given decision threshold.

        :param pred_arr: The raw model predictions; a tensor of shape m x k,
            where m is the number of examples and k is the number of classes.
            All entries are in the range [0, 1].
        :param threshold: The decision threshold, in the range [0, 1]. If the
            second-highest value in pred_arr is greater than this threshold, it
            will be converted to a 1. The highest value is automatically
            converted to a 1 (Pokemon have at least 1 type).
        :return: The valid binary predictions; a tensor of shape m x k, where m
            is the number of example and k is the number of classes. All entries
            are in the set {0, 1}, and in each example there are 1 or 2 ones.
        """
        valid_arr = np.zeros_like(pred_arr)
        top_indices = pred_arr.argsort(axis=1)[:, ::-1]
        for idx in range(len(pred_arr)):
            valid_arr[idx, top_indices[idx, 0]] = 1
            if pred_arr[idx, top_indices[idx, 1]] >= threshold:
                valid_arr[idx, top_indices[idx, 1]] = 1
        return valid_arr
