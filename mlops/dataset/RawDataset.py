"""Contains the RawDataset class."""


# TODO abstract class
class RawDataset:
    """A serializable representation of a raw dataset. This representation could
    be the dataset itself, or a link to the dataset if the raw data are stored
    in their own versioned repository."""

    # TODO abstract method
    def save(self, path: str, endpoint: str = ENDPOINT_LOCAL) -> None:
        """Saves the object to the given path.

        TODO params
        """

    # TODO abstract method
    @staticmethod
    def load(path: str, endpoint: str = ENDPOINT_LOCAL) -> 'RawDataset':
        """Loads the RawDataset object from the given path. The data in the
        directory may be the training/validation/test data, or it may be a batch
        of user data that is intended for prediction, or data in some other
        format.

        TODO params
        """

    # TODO abstract method
    def __eq__(self, other: 'RawDataset') -> bool:
        """Returns True if this object is equal to the other.

        TODO params
        """
