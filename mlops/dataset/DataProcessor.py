"""Contains the DataProcessor class."""

from mlops.dataset import FeaturesAndOptionalLabels


# TODO abstract class
# TODO this object should allow dataset_path to also point to an S3 bucket.
class DataProcessor:
    """Transforms raw data from a file or directory into features for downstream
    model training, prediction, etc."""

    def __init__(self,
                 dataset_path: str | None) -> None:
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

    def _calibrate(self, dataset_path: str) -> None:
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

    # TODO abstract
    def path_to_features_and_labels(self,
                                    dataset_path: str,
                                    endpoint: str = ENDPOINT_LOCAL) -> \
            dict[str, FeaturesAndOptionalLabels]:
        """Transforms the raw data at the given file or directory into features
        and labels that can be used by downstream models.

        :param dataset_path: The path to the file or directory on the local
            filesystem containing the dataset.
        :return: A dictionary whose keys
        """
