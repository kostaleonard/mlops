"""Contains custom errors."""


class PublicationPathAlreadyExistsError(FileExistsError):
    """Raised when a VersionedDatasetBuilder or VersionedModelBuilder object
    calls its publish method, and the path to which the object is published
    already exists."""


class InvalidDatasetCopyStrategyError(ValueError):
    """Raised when a VersionedDatasetBuilder calls its publish method, and the
    dataset copy strategy is not one of the valid options."""
