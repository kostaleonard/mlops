"""Contains the ImageDataProcessor class."""

from mlops.dataset.invertible_data_processor import InvertibleDataProcessor


class PokemonClassificationDataProcessor(InvertibleDataProcessor):
    """Transforms the pokemon dataset at sample_data/pokemon into features and
    labels."""
