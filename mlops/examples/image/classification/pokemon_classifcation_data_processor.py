"""Contains the PokemonClassificationDataProcessor class."""

import numpy as np
from mlops.dataset.invertible_data_processor import InvertibleDataProcessor


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

