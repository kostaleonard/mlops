# MLOps

mlops is a package that enables software engineering within an MLOps framework by
providing:

* Seamless dataset and model versioning
* Publication of versioned datasets and models to local and Cloud filesystems
* Serialization and reproducibility of the data processing pipeline for each
versioned dataset, so that new prediction or training data can be pre- and
post-processed according to the arbitrary transformations that produced the
original dataset
* Capturing of dataset and model metadata, including the connection of models to
the datasets on which they were trained/validated/tested
* A common framework for data processing and model prototyping

## Installation

```bash
pip install mlops
```

## Usage

### Extend class `DataProcessor`

First, create a concrete subclass of abstract `DataProcessor` that tells
`VersionedDatasetBuilder` objects how to transform raw data files into feature
and label tensors.

You can also extend `InvertibleDataProcessor` if you want to be able to invert
preprocessing transformations. This capability helps with model input and output
interpretability, since it's not always readily apparent what the raw input and
output of a model mean to human beings.

```python
import numpy as np
from mlops.dataset.data_processor import DataProcessor


class MyDataProcessor(DataProcessor):
    """Transforms a raw dataset into features and labels for downstream model
    training, prediction, etc."""

    def get_raw_features_and_labels(self, dataset_path: str) -> \
            (dict[str, np.ndarray], dict[str, np.ndarray]):
        """Returns the raw feature and label tensors from the dataset path. This
        method is specifically used for the train/val/test sets and not input
        data for prediction, because in some cases the features and labels need
        to be read simultaneously to ensure proper ordering of features and
        labels.

        For example, when handling image data, the raw features would likely be
        tensors of shape m x h x w x c, where m is the number of images, h is
        the image height, w is the image width, and c is the number of channels
        (3 for RGB), with all values in the interval [0, 255]. The raw labels
        may be tensors of shape m, where m is the number of examples, with all
        values in the set {0, ..., k - 1} indicating the class.

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset, specifically
            train/val/test and not prediction data.
        :return: A 2-tuple of the features dictionary and labels dictionary,
            with matching keys and ordered tensors.
        """
        # Your code here.
        pass

    def get_raw_features(self, dataset_path: str) -> dict[str, np.ndarray]:
        """Returns the raw feature tensors from the dataset path. The raw
        features are how training/validation/test as well as prediction data
        enter the data pipeline. For example, when handling image data, the raw
        features would likely be tensors of shape m x h x w x c, where m is the
        number of images, h is the image height, w is the image width, and c is
        the number of channels (3 for RGB), with all values in the interval
        [0, 255].

        :param dataset_path: The path to the file or directory on the local or
            remote filesystem containing the dataset.
        :return: A dictionary whose values are feature tensors and whose
            corresponding keys are the names by which those tensors should be
            referenced. For example, the training features (value) may be called
            'X_train' (key).
        """
        # Your code here.
        pass

    def preprocess_features(self, raw_feature_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed feature tensor from the raw tensor. The
        preprocessed features are how training/validation/test as well as
        prediction data are fed into downstream models. For example, when
        handling image data, the preprocessed features would likely be tensors
        of shape m x h x w x c, where m is the number of images, h is the image
        height, w is the image width, and c is the number of channels (3 for
        RGB), with all values in the interval [0, 1].

        :param raw_feature_tensor: The raw features to be preprocessed.
        :return: The preprocessed feature tensor. This tensor is ready for
            downstream model consumption.
        """
        # Your code here.
        pass

    def preprocess_labels(self, raw_label_tensor: np.ndarray) -> np.ndarray:
        """Returns the preprocessed label tensor from the raw tensor. The
        preprocessed labels are how training/validation/test as well as
        prediction data are fed into downstream models. For example, in a
        classification task, the preprocessed labels may be tensors of shape
        m x k, where m is the number of examples, and k is the number of
        classes, where each of the k-length vectors are one-hot encoded.

        :param raw_label_tensor: The raw labels to be preprocessed.
        :return: The preprocessed label tensor. This tensor is ready for
            downstream model consumption.
        """
        # Your code here.
        pass
```

### Publish a dataset

Use a `VersionedDatasetBuilder` to build and publish a dataset. You can publish the
dataset to the local filesystem or a cloud store like S3. `VersionedDatasetBuilder`
objects publish versioned datasets, which can be loaded directly from their paths.
This relationship ensures that every instantiated `VersionedDataset` object is in
the dataset repository (i.e., you can't create an "unversioned"
`VersionedDataset`).

During publication, the `VersionedDatasetBuilder` serializes the data processor
and adds it to the dataset repository as an artifact. The serialized data
processor captures the data pre- and post-processing instructions at the time of
dataset creation, which may not be tied to any commit in the project VCS (saving
the commit at which the dataset was built is not sufficient for reproducing the
data processing pipeline). If you decide to change your data processor class
definition to output data in a new schema, previous versioned datasets still
"know" how to transform data into a format consistent with the
training/validation/test datasets through the `data_processor` property.

```python
from mlops.dataset.versioned_dataset_builder import VersionedDatasetBuilder

processor = MyDataProcessor()
builder = VersionedDatasetBuilder('path/to/my/dataset', processor)
builder.publish('s3://my-bucket/datasets', 'v1', tags=['image', 'classification'])
```

TODO show a screenshot of the published dataset files here.

### Publish a model

Now train and publish a model using the versioned dataset. The `VersionedDataset`
is important because it standardizes the training/validation/test datasets,
providing a common point of comparison between models. It also captures the
transformations required to feed data into models using the `data_processor`
property. Every prototype model should be published so that any results achieved
can be reproduced.

The `TrainingConfig` object saves the training history and hyperparameters for
experiment tracking. Both of these items are stored in the model metadata.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.training_config import TrainingConfig
from mlops.model.versioned_model_builder import VersionedModelBuilder

versioned_dataset = VersionedDataset('s3://my-bucket/datasets/v1')
model = Sequential([Dense(versioned_dataset.y_train.shape[1],
                          input_shape=versioned_dataset.X_train.shape[1:])])
model.compile('adam', loss='mse')
train_kwargs = {'epochs': 5,
                'batch_size': 8}
history = model.fit(x=versioned_dataset.X_train,
                    y=versioned_dataset.y_train,
                    **train_kwargs)
training_config = TrainingConfig(history, train_kwargs)
builder = VersionedModelBuilder(versioned_dataset, model, training_config)
builder.publish('s3://my-bucket/models', tags=['prototype'])
```

TODO show a screenshot of the published model files here.

### Predict using VersionedDataset and VersionedModel

Now that both datasets and models are versioned, use VersionedDataset and
VersionedModel objects to process new data and run prediction.

```python
from mlops.model.versioned_model import VersionedModel

versioned_model = VersionedModel(
    's3://my-bucket/models/2021-12-19T06:59:00.451852')
features = versioned_dataset.data_processor.get_preprocessed_features(
    'path/to/new/data/for/prediction')
predictions = versioned_model.model.predict(features['X_pred'])
```
