# Image classification

This example demonstrates dataset and model versioning as part of an MLOps workflow.

**Note: All modules are designed to be run from the project root.**

Task: Classify images of Pokemon as belonging to one or two of 18 types.

Dataset: A few examples are available in the [sample_data directory](../../../../sample_data/pokemon) for ease of use;
the full dataset is available [on Kaggle](https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types).

## Modules

1. [pokemon_classification_data_processor.py](pokemon_classification_data_processor.py): Defines how to translate the
raw images and CSV data at the raw dataset directory into feature and label tensors.
2. [publish_dataset.py](publish_dataset.py): Publishes a new version of the dataset. This module only needs to be run if
the data processor changes.
3. [train_model.py](train_model.py): Trains and publishes a new model.
4. [model_prediction.py](model_prediction.py): Uses versioned models to run prediction on unseen data.
5. [errors.py](errors.py): Custom errors.

## Notebooks

1. [prototype_model.ipynb](notebooks/prototype_model.ipynb): Prototypes models and investigates results. Contains the
full ML pipeline, including versioned dataset publication, versioned model publication, and prediction.

## Run instructions

### Notebook

The easiest way to run and understand this example is to run [prototype_model.ipynb](notebooks/prototype_model.ipynb),
which contains all steps of the ML pipeline. The notebook first publishes a versioned dataset if one does not already
exist, creates a new prototype TensorFlow model, trains and publishes the versioned model, and examines the results on
unseen data.

### Python scripts

You can also run each of the scripts individually from the project root.

#### 1. (Optional) publish a versioned dataset

This step is optional because the training script automatically publishes a versioned dataset if none exist. You can
publish a dataset by running the following. If a
dataset of the same version as defined in the script already exists, this will raise a
`PublicationPathAlreadyExistsError`. If you make any changes to `pokemon_classification_data_processor.py`, publish a
new dataset by incrementing the version in `publish_dataset.py` and running the script again. This script exists as a
standalone module because you may want to publish a dataset without training a model.

```bash
PYTHONPATH=. python mlops/examples/image/classification/publish_dataset.py
```

At the project root directory, you will now see `datasets/pokemon/v1` containing the published versioned dataset. You
can change the arguments to `publish()` to change the path (to a local or remote--e.g., S3--destination), version, and
tags.

#### 2. Train and publish a versioned model

Train and publish a versioned model with the following. If no versioned dataset exists, one will be created.

```bash
PYTHONPATH=. python mlops/examples/image/classification/train_model.py
```

Again at the project root directory, there will now be a `models/pokemon/versioned` directory containing models
versioned by timestamp. The `checkpoints` directory contains model checkpoints in case training is interrupted.

#### 3. Run model prediction

Retrieve the best versioned model (based on last epoch validation loss), compute the test error, and run prediction on
the 3 sample prediction images with the following.

```bash
PYTHONPATH=. python mlops/examples/image/classification/model_prediction.py
```

## Summary

You've just created versioned datasets and models for all of your experiments, and standardized your data processing
pipeline. You now have a record of which of your trained models performed the best, which dataset each model used, and
which combinations of architectures and hyperparameters you tried. You can roll back the dataset and model to any
arbitrary version and reproduce the results you achieved before.
