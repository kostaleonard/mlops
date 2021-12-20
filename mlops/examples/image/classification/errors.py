"""Contains custom errors for the Pokemon classification example."""


class LabelsNotFoundError(FileNotFoundError):
    """Raised when a PokemonClassificationDataProcessor attempts to load labels
    for prediction data, an unlabeled data source."""


class NoModelPathsSuppliedError(ValueError):
    """Raised when a non-empty collection of strings representing paths to
    models is expected, but an empty collection is passed instead."""
