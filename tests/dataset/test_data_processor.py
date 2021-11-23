"""Tests data_processor.py."""

import pytest
from mlops.dataset import data_processor as dp


def test_data_processor_is_abstract() -> None:
    """Tests that the DataProcessor object is abstract."""
    with pytest.raises(TypeError):
        _ = dp.DataProcessor()
