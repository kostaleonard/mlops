"""Contains custom errors."""


class PublicationPathAlreadyExistsError(IOError):
    """Raised when a VersionedDatasetBuilder or VersionedModelBuilder object
    calls its publish method, and the path to which the object is published
    already exists."""
