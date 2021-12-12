"""Contains custom errors for the Pokemon classification example."""


class LabelsNotFoundError(FileNotFoundError):
    """Raised when a PokemonClassificationDataProcessor attempts to load labels
    for prediction data, an unlabeled data source."""
