"""Contains the ImageClassificationDataProcessor class."""

from mlops.dataset.invertible_data_processor import InvertibleDataProcessor


class ImageClassificationDataProcessor(InvertibleDataProcessor):
    """Transforms the pokemon dataset at sample_data/pokemon into features and
    labels for classification."""
