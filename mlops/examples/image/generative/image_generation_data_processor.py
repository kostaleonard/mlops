"""Contains the ImageGenerationDataProcessor class."""

from mlops.dataset.invertible_data_processor import InvertibleDataProcessor


class ImageGenerationDataProcessor(InvertibleDataProcessor):
    """Transforms the pokemon dataset at sample_data/pokemon into features for
    image generation."""
