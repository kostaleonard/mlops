"""Contains the PresetDataProcessor class."""

from mlops.dataset.data_processor import DataProcessor


class PresetDataProcessor(DataProcessor):
    """Processes a preset dataset, with no file I/O."""
