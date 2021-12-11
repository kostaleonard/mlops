"""Contains custom errors for the Pokemon classification example."""


class AttemptedToLoadPredictionLabelsError(ValueError):
    """Raised when a PokemonClassificationDataProcessor attempts to load labels
    for prediction data, an unlabeled data source."""
