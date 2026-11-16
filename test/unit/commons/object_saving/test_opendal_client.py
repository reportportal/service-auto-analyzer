#  Copyright 2026 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
import pickle
import subprocess
import sys
import time
from unittest import mock

import boto3
import opendal

# noinspection PyPackageRequirements
import pytest
import requests
from botocore.exceptions import ClientError

from app.commons.model.launch_objects import ApplicationConfig
from app.commons.object_saving.object_saver import ObjectSaver
from app.commons.object_saving.opendal_client import DATASTORE_TYPES, OpenDalClient, build_operator_config
from test import random_alphanumeric

SERVER_PORT = 5125
REGION = "us-west-1"
ACCESS_KEY = "test"
SECRET_KEY = "test"
DEFAULT_BUCKET = "rp-bucket"
SINGLE_BUCKET = "test-bucket"
MISSING_BUCKET = "missing-bucket"
# Mute Sonar warning about hardcoded HTTP URL, since this is a test setup
# noinspection HttpUrlsUsage
ENDPOINT = f"http://localhost:{SERVER_PORT}"  # NOSONAR

# Bucket prefix, the bucket where objects are stored and the folder of project "2" inside it
BUCKET_CASES = [
    ("prj-", DEFAULT_BUCKET, "prj-2"),
    (f"{SINGLE_BUCKET}/prj-", SINGLE_BUCKET, "prj-2"),
    (f"{SINGLE_BUCKET}/reportportal/prj-", SINGLE_BUCKET, "reportportal/prj-2"),
]
BUCKET_PREFIXES = [case[0] for case in BUCKET_CASES]
BUCKET_NAMES = {case[0]: case[1] for case in BUCKET_CASES}


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )


@pytest.fixture(autouse=True, scope="module")
def run_s3():
    # OpenDAL listing blocks Python threads while it waits for the response, so the in-process
    # `ThreadedMotoServer` is not able to answer. Run the server in a separate process instead.
    server = subprocess.Popen(
        [sys.executable, "-m", "moto.server", "-p", str(SERVER_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                requests.get(f"{ENDPOINT}/moto-api/", timeout=1)
                break
            except requests.exceptions.ConnectionError:
                time.sleep(0.1)
        s3_client = get_s3_client()
        for bucket_name in (DEFAULT_BUCKET, SINGLE_BUCKET):
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": REGION})
        yield
    finally:
        server.terminate()
        server.wait(timeout=10)


def create_storage_client(
    bucket_prefix: str = "prj-", bucket_postfix: str = "", bucket_name: str | None = None
) -> OpenDalClient:
    return OpenDalClient(
        ApplicationConfig(
            datastoreType="s3-compatible",
            datastoreEndpoint=ENDPOINT,
            datastoreRegion=REGION,
            datastoreBucketPrefix=bucket_prefix,
            datastoreBucketPostfix=bucket_postfix,
            datastoreDefaultBucketName=bucket_name or BUCKET_NAMES.get(bucket_prefix, DEFAULT_BUCKET),
            datastoreAccessKey=ACCESS_KEY,
            datastoreSecretKey=SECRET_KEY,
        )
    )


def get_stored_object(bucket_name: str, key: str) -> bytes | None:
    try:
        return get_s3_client().get_object(Bucket=bucket_name, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            return None
        raise


@pytest.mark.parametrize("bucket_prefix", BUCKET_PREFIXES)
def test_object_not_exists(bucket_prefix):
    client = create_storage_client(bucket_prefix)

    assert not client.does_object_exists("2", f"{random_alphanumeric(16)}.pickle")


@pytest.mark.parametrize("bucket_prefix", BUCKET_PREFIXES)
def test_object_exists(bucket_prefix):
    object_name = f"{random_alphanumeric(16)}.pickle"
    client = create_storage_client(bucket_prefix)

    client.put_project_object({"test": True}, "2", object_name)

    assert client.does_object_exists("2", object_name)


@pytest.mark.parametrize("bucket_prefix, bucket_name, project_folder", BUCKET_CASES)
def test_json_write(bucket_prefix, bucket_name, project_folder):
    object_name = f"{random_alphanumeric(16)}.json"
    client = create_storage_client(bucket_prefix)

    client.put_project_object({"test": True}, "2", object_name, using_json=True)

    key = f"{project_folder}/{object_name}"
    assert get_stored_object(bucket_name, key) == b'{"test": true}'
    assert get_s3_client().head_object(Bucket=bucket_name, Key=key)["ContentType"] == "application/json"


@pytest.mark.parametrize("bucket_prefix, bucket_name, project_folder", BUCKET_CASES)
def test_pickle_write(bucket_prefix, bucket_name, project_folder):
    object_name = f"{random_alphanumeric(16)}.pickle"
    client = create_storage_client(bucket_prefix)

    client.put_project_object({"test": [1, 2]}, "2", object_name)

    key = f"{project_folder}/{object_name}"
    assert pickle.loads(get_stored_object(bucket_name, key)) == {"test": [1, 2]}
    assert get_s3_client().head_object(Bucket=bucket_name, Key=key)["ContentType"] == "application/octet-stream"


@pytest.mark.parametrize("bucket_prefix, bucket_name, project_folder", BUCKET_CASES)
def test_json_read(bucket_prefix, bucket_name, project_folder):
    object_name = f"{random_alphanumeric(16)}.json"
    get_s3_client().put_object(
        Bucket=bucket_name, Key=f"{project_folder}/{object_name}", Body=json.dumps({"test": True}).encode("utf-8")
    )
    client = create_storage_client(bucket_prefix)

    result = client.get_project_object("2", object_name, using_json=True)

    assert result == {"test": True}


@pytest.mark.parametrize("bucket_prefix", BUCKET_PREFIXES)
def test_pickle_round_trip(bucket_prefix):
    object_name = f"{random_alphanumeric(16)}.pickle"
    client = create_storage_client(bucket_prefix)

    client.put_project_object({"test": (1, "a")}, "2", object_name)

    assert client.get_project_object("2", object_name) == {"test": (1, "a")}


@pytest.mark.parametrize("bucket_prefix, bucket_name, project_folder", BUCKET_CASES)
def test_not_existing_file_get(bucket_prefix, bucket_name, project_folder):
    object_name = f"{random_alphanumeric(16)}.json"
    client = create_storage_client(bucket_prefix)

    with pytest.raises(ValueError) as exc:
        client.get_project_object("2", object_name)
    assert exc.value.args[0] == f'Unable to get file with path "{project_folder}/{object_name}"'


@pytest.mark.parametrize("bucket_prefix", BUCKET_PREFIXES)
def test_remove_not_existing_folder(bucket_prefix):
    client = create_storage_client(bucket_prefix)

    assert not client.remove_folder_objects("3", "test-remove-not-existing")


@pytest.mark.parametrize("bucket_prefix, bucket_name, project_folder", BUCKET_CASES)
def test_remove_existing_folder(bucket_prefix, bucket_name, project_folder):
    folder = f"folder-{random_alphanumeric(8)}"
    resources = [f"{folder}/my_test_file_folder.json", f"{folder}/nested/my_nested_file.json"]
    client = create_storage_client(bucket_prefix)
    for resource in resources:
        client.put_project_object({"test": True}, "2", resource)

    assert client.remove_folder_objects("2", folder)

    for resource in resources:
        assert get_stored_object(bucket_name, f"{project_folder}/{resource}") is None


def test_remove_folder_keeps_other_folders():
    folder = f"folder-{random_alphanumeric(8)}"
    other_resource = f"{folder}-other/my_test_file.json"
    client = create_storage_client()
    client.put_project_object({"test": True}, "2", f"{folder}/my_test_file.json")
    client.put_project_object({"test": True}, "2", other_resource)

    assert client.remove_folder_objects("2", folder)

    assert client.does_object_exists("2", other_resource)


@pytest.mark.parametrize("bucket_prefix", BUCKET_PREFIXES)
def test_list_not_existing_folder(bucket_prefix):
    client = create_storage_client(bucket_prefix)

    assert client.get_folder_objects("4", "test") == []


@pytest.mark.parametrize("bucket_prefix", BUCKET_PREFIXES)
def test_list_existing_folder(bucket_prefix):
    resource = f"list-{random_alphanumeric(8)}/my_test_file_list.json"
    client = create_storage_client(bucket_prefix)
    client.put_project_object({"test": True}, "6", resource, using_json=True)

    assert client.get_folder_objects("6", resource.split("/")[0]) == [resource]


@pytest.mark.parametrize("bucket_prefix", BUCKET_PREFIXES)
def test_list_folder_not_recursive(bucket_prefix):
    """Listing a folder should return only its direct children, not objects of nested folders."""
    path = f"suggestion_model_{random_alphanumeric(8)}/"
    model_folder = f"{path}suggestion_model_2026-07-01"
    client = create_storage_client(bucket_prefix)
    client.put_project_object({"test": True}, "10", f"{model_folder}/boost_model.pickle", using_json=True)
    client.put_project_object({"test": True}, "10", f"{model_folder}/data_features_config.pickle", using_json=True)

    assert client.get_folder_objects("10", path) == [model_folder]


def test_list_dir_separators():
    resource = f"test-{random_alphanumeric(8)}/{random_alphanumeric(16)}.json"
    client = create_storage_client()
    client.put_project_object({"test": True}, "7", resource, using_json=True)

    assert client.get_folder_objects("7", resource.split("/")[0] + "/") == [resource]


@pytest.mark.parametrize("bucket_prefix", BUCKET_PREFIXES)
def test_remove_project_objects(bucket_prefix):
    resource = f"objects/{random_alphanumeric(16)}.json"
    client = create_storage_client(bucket_prefix)
    client.put_project_object({"test": True}, "8", resource, using_json=True)
    assert client.get_project_object("8", resource, using_json=True) == {"test": True}

    client.remove_project_objects("8", [resource, "objects/not_existing.json"])

    with pytest.raises(ValueError):
        client.get_project_object("8", resource)


@pytest.mark.parametrize(
    "bucket_prefix, bucket_postfix, bucket_name, project_folder",
    [
        ("prj-", "-data", DEFAULT_BUCKET, "prj-9-data"),
        (f"{SINGLE_BUCKET}/prj-", "-storage", SINGLE_BUCKET, "prj-9-storage"),
    ],
)
def test_bucket_postfix(bucket_prefix, bucket_postfix, bucket_name, project_folder):
    object_name = f"{random_alphanumeric(16)}.json"
    client = create_storage_client(bucket_prefix, bucket_postfix)

    client.put_project_object({"test_postfix": True}, "9", object_name, using_json=True)

    assert get_stored_object(bucket_name, f"{project_folder}/{object_name}") == b'{"test_postfix": true}'
    assert client.does_object_exists("9", object_name)
    assert client.get_project_object("9", object_name, using_json=True) == {"test_postfix": True}


def test_missing_bucket():
    client = create_storage_client(f"{MISSING_BUCKET}/prj-", bucket_name=MISSING_BUCKET)

    assert not client.does_object_exists("2", "test.json")
    assert client.get_folder_objects("2", "test") == []
    assert not client.remove_folder_objects("2", "test")
    client.remove_project_objects("2", ["test.json"])
    with pytest.raises(ValueError):
        client.get_project_object("2", "test.json")
    with pytest.raises(opendal.exceptions.ConfigInvalid):
        client.put_project_object({"test": True}, "2", "test.json")


def create_memory_client(bucket_prefix: str = "prj-") -> OpenDalClient:
    return OpenDalClient(
        ApplicationConfig(datastoreType="s3-compatible", datastoreBucketPrefix=bucket_prefix),
        operator=opendal.Operator("memory"),
    )


def test_memory_empty_project():
    client = create_memory_client()

    client.put_project_object({"test": True}, "", "folder/test.json", using_json=True)

    assert client.does_object_exists("", "folder/test.json")
    assert client.get_folder_objects("", "folder") == ["folder/test.json"]
    assert client.get_project_object("", "folder/test.json", using_json=True) == {"test": True}
    assert client.remove_folder_objects("", "folder")
    assert not client.does_object_exists("", "folder/test.json")


def test_memory_projects_are_separated():
    client = create_memory_client()

    client.put_project_object({"project": 1}, "1", "model/data.json", using_json=True)
    client.put_project_object({"project": 2}, "2", "model/data.json", using_json=True)
    assert client.remove_folder_objects("1", "model")

    assert not client.does_object_exists("1", "model/data.json")
    assert client.get_project_object("2", "model/data.json", using_json=True) == {"project": 2}


@pytest.mark.parametrize(
    "app_config, expected_scheme, expected_options",
    [
        (ApplicationConfig(datastoreType="s3-compatible"), "s3", {"bucket": "rp-bucket"}),
        (
            ApplicationConfig(
                datastoreType="s3-compatible",
                datastoreEndpoint="https://s3.example.com",
                datastoreRegion="eu-west-1",
                datastoreAccessKey="access",
                datastoreSecretKey="secret",
                datastoreBucketPrefix="my-bucket/prj-",
                datastoreDefaultBucketName="my-bucket",
            ),
            "s3",
            {
                "bucket": "my-bucket",
                "region": "eu-west-1",
                "endpoint": "https://s3.example.com",
                "access_key_id": "access",
                "secret_access_key": "secret",
            },
        ),
        (
            ApplicationConfig(
                datastoreType="aws-s3", datastoreEndpoint="https://s3.example.com", datastoreRegion="us-east-1"
            ),
            "s3",
            {"bucket": "rp-bucket", "region": "us-east-1"},
        ),
        (
            ApplicationConfig(datastoreType="aws-s3", datastoreRegion="eu-west-1", datastoreAccessKey="access"),
            "s3",
            {"bucket": "rp-bucket", "region": "eu-west-1"},
        ),
        (
            ApplicationConfig(
                datastoreType="aws-s3",
                datastoreRegion="eu-west-1",
                datastoreAccessKey="access",
                datastoreSecretKey="secret",
            ),
            "s3",
            {"bucket": "rp-bucket", "region": "eu-west-1", "access_key_id": "access", "secret_access_key": "secret"},
        ),
        (
            ApplicationConfig(
                datastoreType="azure",
                datastoreAzureAccountName="account",
                datastoreAzureAccountKey="key",
            ),
            "azblob",
            {"account_name": "account", "account_key": "key"},
        ),
        (
            ApplicationConfig(
                datastoreType="azure",
                datastoreAzureAccountName="account",
                datastoreAzureAccountKey="key",
                datastoreAzureEndpoint="https://account.blob.core.windows.net",
                datastoreAzureContainer="container",
                datastoreAccessKey="ignored",
            ),
            "azblob",
            {
                "container": "container",
                "account_name": "account",
                "account_key": "key",
                "endpoint": "https://account.blob.core.windows.net",
            },
        ),
        (ApplicationConfig(datastoreType="gcs"), "gcs", {}),
        (
            ApplicationConfig(
                datastoreType="gcs",
                datastoreGcsBucket="gcs-bucket",
                datastoreGcsCredentialsPath="/etc/gcs/key.json",
                datastoreGcsEndpoint="https://storage.example.com",
            ),
            "gcs",
            {
                "bucket": "gcs-bucket",
                "credential_path": "/etc/gcs/key.json",
                "endpoint": "https://storage.example.com",
            },
        ),
        (
            ApplicationConfig(
                datastoreType="s3-compatible",
                datastoreEndpoint="http://minio:9000",
                datastoreRegion="us-east-1",
                datastoreOpendalOptions={"endpoint": "http://localhost:9000", "enable_virtual_host_style": "true"},
            ),
            "s3",
            {
                "bucket": "rp-bucket",
                "region": "us-east-1",
                "endpoint": "http://localhost:9000",
                "enable_virtual_host_style": "true",
            },
        ),
    ],
)
def test_build_operator_config(app_config, expected_scheme, expected_options):
    assert build_operator_config(app_config) == (expected_scheme, expected_options)


@pytest.mark.parametrize("datastore_type", ["filesystem", "minio", "s3", "opendal"])
def test_build_operator_config_unsupported_type(datastore_type):
    with pytest.raises(ValueError):
        build_operator_config(ApplicationConfig(datastoreType=datastore_type))


@pytest.mark.parametrize(
    "datastore_type, expected_scheme",
    [("s3-compatible", "s3"), ("aws-s3", "s3"), ("azure", "azblob"), ("gcs", "gcs")],
)
def test_object_saver_creates_opendal_client(datastore_type, expected_scheme):
    assert datastore_type in DATASTORE_TYPES
    with mock.patch("app.commons.object_saving.opendal_client.opendal.Operator") as operator_class:
        object_saver = ObjectSaver(ApplicationConfig(datastoreType=datastore_type))

    assert isinstance(object_saver.storage, OpenDalClient)
    operator_class.assert_called_once()
    assert operator_class.call_args.args == (expected_scheme,)
