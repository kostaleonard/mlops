"""Contains custom errors for the Pokemon classification example."""


class TrainValTestFeaturesAndLabelsNotLoadedSimultaneouslyError(ValueError):
    """Raised when a PokemonClassificationDataProcessor attempts to load train/
    val/test features or labels one at a time rather than simultaneously,
    leading to improperly ordered examples in feature/label tensors."""


class AttemptedToLoadPredictionLabelsError(ValueError):
    """Raised when a PokemonClassificationDataProcessor attempts to load labels
    for prediction data, an unlabeled data source."""
