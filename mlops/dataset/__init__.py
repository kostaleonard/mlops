"""Contains dataset modules."""

import numpy as np

Features = np.ndarray
Labels = np.ndarray
FeaturesAndOptionalLabels = (Features, Labels | None)

ENDPOINT_LOCAL = 'local'
ENDPOINT_S3 = 's3'
