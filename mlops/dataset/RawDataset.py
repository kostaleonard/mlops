"""Contains the RawDataset class."""


# TODO abstract class
class RawDataset:
    """A serializable raw dataset object."""

    # TODO abstract method
    def save(self, path: str, endpoint: str = ENDPOINT_LOCAL) -> None:
        """Saves the object to the given path.

        TODO params
        """

    # TODO abstract method
    @staticmethod
    def load(path: str, endpoint: str = ENDPOINT_LOCAL) -> 'RawDataset':
        """Loads the RawDataset object from the given path.

        TODO params
        """

    # TODO abstract method
    def __eq__(self, other: 'RawDataset') -> bool:
        """Returns True if this object is equal to the other.

        TODO params
        """
