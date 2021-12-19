# MLOps

mlops is a package that enables software engineering within an MLOps framework by
providing:

* Seamless dataset and model versioning
* Publication of versioned datasets and models to local and Cloud filesystems
* Serialization and reproducibility of the data processing pipeline for each
versioned dataset, so that new prediction or training data can be pre- and
post-processed according to the arbitrary transformations that produced the
original dataset
* Capturing of dataset and model metadata, including the linking of models to the
datasets on which they were trained/validated/tested
* A common framework for data processing and model prototyping

## Installation

```bash
pip install mlops
```

## Usage

### Extend class `DataProcessor`

First, create a concrete subclass of abstract `DataProcessor` that tells `DatasetBuilder` objects how to transform raw data into features.

```python
import numpy as np
from mlops import DataProcessor


class MNISTDataProcessor(DataProcessor):
    """Transforms raw pixel data into features for downstream models."""
    
    def raw_to_features(dataset_path: str) -> (np.ndarray, np.ndarray):
        """Transforms raw image files into tensors"""
```


```python
from mlops import DatasetBuilder

class MNISTDatasetBuilder(DatasetBuilder):
    """Builds the MNIST dataset and publishes to S3."""
    
    def __init__(self, raw_filename: str) -> None:
        """Instantiates the object."""
        super().__init__(raw_filename, MNISTDataProcessor())
```
