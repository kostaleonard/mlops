"""Contains test fixtures."""
# pylint: disable=no-name-in-module

import os
import pytest
import boto3
from moto import mock_s3
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from mlops.dataset.versioned_dataset import VersionedDataset
from mlops.model.training_config import TrainingConfig
from tests.dataset.test_versioned_dataset import _publish_test_dataset_local, \
    TEST_PUBLICATION_PATH_LOCAL as TEST_DATASET_PUBLICATION_PATH_LOCAL, \
    _publish_test_dataset_s3, \
    TEST_PUBLICATION_PATH_S3 as TEST_DATASET_PUBLICATION_PATH_S3

TEST_BUCKET = 'kosta-mlops'
TEST_REGION = 'us-east-2'


@pytest.fixture(name='dataset')
def fixture_dataset() -> VersionedDataset:
    """Returns the versioned dataset fixture for testing.

    :return: The versioned dataset fixture.
    """
    _publish_test_dataset_local()
    return VersionedDataset(os.path.join(TEST_DATASET_PUBLICATION_PATH_LOCAL,
                                         'v1'))


@pytest.fixture(name='dataset_s3')
def fixture_dataset_s3(mocked_s3: None) -> VersionedDataset:
    """Returns the S3 versioned dataset fixture for testing.

    :param mocked_s3: A mocked S3 bucket for testing.
    :return: The S3 versioned dataset fixture.
    """
    # pylint: disable=unused-argument
    _publish_test_dataset_s3()
    return VersionedDataset(os.path.join(TEST_DATASET_PUBLICATION_PATH_S3,
                                         'v1'))


@pytest.fixture(name='model')
def fixture_model(dataset: VersionedDataset) -> Model:
    """Returns the model fixture for testing.

    :param dataset: The versioned dataset.
    :return: The model fixture.
    """
    mod = Sequential([Dense(dataset.y_train.shape[1],
                            input_shape=dataset.X_train.shape[1:])])
    mod.compile('adam', loss='mse')
    return mod


@pytest.fixture(name='training_config')
def fixture_training_config(dataset: VersionedDataset,
                            model: Model) -> TrainingConfig:
    """Returns the training configuration fixture for testing.

    :param dataset: The versioned dataset.
    :param model: The model.
    :return: The training configuration fixture.
    """
    train_kwargs = {'epochs': 5,
                    'batch_size': 8}
    history = model.fit(x=dataset.X_train,
                        y=dataset.y_train,
                        **train_kwargs)
    return TrainingConfig(history, train_kwargs)


@pytest.fixture(name='aws_credentials', scope='session')
def fixture_aws_credentials() -> None:
    """Mocked AWS credentials for moto."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'
    os.environ['AWS_DEFAULT_REGION'] = TEST_REGION


@pytest.fixture(name='mocked_s3', scope='session')
def fixture_mocked_s3(aws_credentials: None) -> None:
    """Creates a mocked S3 bucket for tests.

    :param aws_credentials: Mocked AWS credentials.
    """
    # pylint: disable=unused-argument
    with mock_s3():
        conn = boto3.resource('s3', region_name=TEST_REGION)
        conn.create_bucket(
            Bucket=TEST_BUCKET,
            CreateBucketConfiguration={
                'LocationConstraint': TEST_REGION
            }
        )
        yield
