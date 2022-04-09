"""Tests connection to S3."""

import os
import pytest
import boto3
from botocore.exceptions import ClientError

PROJECT_S3_BUCKET = 'kosta-mlops'
TEST_FILENAME = 's3_conn_test.txt'
TEST_FILE_PATH = os.path.join('/tmp', TEST_FILENAME)


@pytest.mark.awstest
def test_connection_to_project_s3_bucket(mocked_s3: None) -> None:
    """Tests that the project S3 bucket (s3://kosta-mlops) can be reached.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    s3 = boto3.client('s3')
    response = s3.list_buckets()
    bucket_matches = [bucket_dict for bucket_dict in response['Buckets']
                      if bucket_dict['Name'] == PROJECT_S3_BUCKET]
    assert len(bucket_matches) == 1


@pytest.mark.awstest
def test_project_s3_bucket_read_write(mocked_s3: None) -> None:
    """Tests that the project S3 bucket (s3://kosta-mlops) can be read from and
    written to.

    :param mocked_s3: A mocked S3 bucket for testing.
    """
    s3 = boto3.client('s3')
    # Remove test file from local filesystem.
    try:
        os.remove(TEST_FILE_PATH)
    except FileNotFoundError:
        pass
    # Remove test file from S3.
    s3.delete_object(Bucket=PROJECT_S3_BUCKET, Key=TEST_FILENAME)
    # Create test file on local filesystem.
    with open(TEST_FILE_PATH, 'w', encoding='utf-8') as outfile:
        outfile.write(TEST_FILENAME)
    # Attempt to upload to S3.
    try:
        _ = s3.upload_file(TEST_FILE_PATH, PROJECT_S3_BUCKET, TEST_FILENAME)
    except ClientError as exc:
        raise exc
    # Remove test file from local filesystem.
    os.remove(TEST_FILE_PATH)
    # Read test file from S3.
    s3.download_file(PROJECT_S3_BUCKET, TEST_FILENAME, TEST_FILE_PATH)
    # Remove test file from S3.
    s3.delete_object(Bucket=PROJECT_S3_BUCKET, Key=TEST_FILENAME)
    # Check that contents of downloaded file match what we sent for upload.
    with open(TEST_FILE_PATH, 'r', encoding='utf-8') as infile:
        assert infile.read() == TEST_FILENAME
