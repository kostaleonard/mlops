"""Contains dataset modules."""

import numpy as np

ENDPOINT_LOCAL = 'local'
ENDPOINT_S3 = 's3'


# TODO classes should probably be in their own files.
class Tensor:
    """Represents a multi-dimensional array, or tensor."""

    def __init__(self, name: str, values: np.ndarray) -> None:
        """TODO docstring"""
        self.name = name
        self.values = values


class Features(Tensor):
    """Represents a feature tensor."""


class Labels(Tensor):
    """Represents a label tensor."""


FeaturesAndOptionalLabels = (Features, Labels | None)
