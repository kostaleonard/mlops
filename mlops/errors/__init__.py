"""Contains custom errors."""


class PublicationPathAlreadyExistsError(FileExistsError):
    """Raised when a VersionedDatasetBuilder or VersionedModelBuilder object
    calls its publish method, and the path to which the object is published
    already exists."""


class InvalidDatasetCopyStrategyError(ValueError):
    """Raised when a VersionedDatasetBuilder calls its publish method, and the
    dataset copy strategy is not one of the valid options."""


class GeneratorMappingInvalidInputShapeError(ValueError):
    """Raised when a GeneratorMapping object is built with an invalid input
    shape based on its constructor arguments."""


class UnknownActivationFunctionError(ValueError):
    """Raised when an unrecognized object is passed to create the activation
    function for a layer."""
