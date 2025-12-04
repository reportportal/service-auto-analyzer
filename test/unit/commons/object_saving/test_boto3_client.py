#  Copyright 2023 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# noinspection PyPackageRequirements
import pytest

# noinspection PyPackageRequirements
from moto.server import ThreadedMotoServer

from app.commons.model.launch_objects import ApplicationConfig
from app.commons.object_saving.boto3_client import Boto3Client
from test import random_alphanumeric

SERVER_PORT = 5124
REGION = "us-west-1"
BUCKET_PREFIX = "prj-"
SERVER_HOST = f"localhost:{SERVER_PORT}"


@pytest.fixture(autouse=True, scope="session")
def run_s3():
    server = ThreadedMotoServer(port=SERVER_PORT, verbose=True)
    server.start()
    yield
    server.stop()


def create_storage_client(bucket_prefix=BUCKET_PREFIX):
    return Boto3Client(
        ApplicationConfig(
            s3Endpoint=f"http://{SERVER_HOST}",
            s3Region=REGION,
            bucketPrefix=bucket_prefix,
            s3AccessKey="test",
            s3SecretKey="test",
        )
    )


@pytest.mark.parametrize(
    "bucket_prefix, bucket, object_name",
    [
        (BUCKET_PREFIX, "2", f"{random_alphanumeric(16)}.pickle"),
        (f"test/{BUCKET_PREFIX}", "2", f"{random_alphanumeric(16)}.json"),
    ],
)
def test_object_not_exists(bucket_prefix, bucket, object_name):
    boto3_client = create_storage_client(bucket_prefix)

    assert not boto3_client.does_object_exists("2", object_name)


@pytest.mark.parametrize(
    "bucket_prefix, bucket, object_name",
    [
        (BUCKET_PREFIX, "2", f"{random_alphanumeric(16)}.pickle"),
        (f"test/{BUCKET_PREFIX}", "2", f"{random_alphanumeric(16)}.json"),
    ],
)
def test_object_exists(bucket_prefix, bucket, object_name):
    boto3_client = create_storage_client(bucket_prefix)

    boto3_client.put_project_object({"test": True}, "2", object_name)

    assert boto3_client.does_object_exists("2", object_name)


@pytest.mark.parametrize(
    "bucket_prefix, bucket, bucket_name, object_name, object_path",
    [
        (BUCKET_PREFIX, "2", f"{BUCKET_PREFIX}2", "my_test_file_write.json", "my_test_file_write.json"),
        (f"test/{BUCKET_PREFIX}", "2", "test", "my_test_file_write.json", f"{BUCKET_PREFIX}2/my_test_file_write.json"),
        (
            f"test/reportportal/{BUCKET_PREFIX}",
            "2",
            "test",
            "my_test_file_write.json",
            f"reportportal/{BUCKET_PREFIX}2/my_test_file_write.json",
        ),
    ],
)
def test_json_write(bucket_prefix, bucket, bucket_name, object_name, object_path):
    boto3_client = create_storage_client(bucket_prefix)

    boto3_client.put_project_object({"test": True}, bucket, object_name, using_json=True)

    # Verify by reading back through the client
    result = boto3_client.get_project_object(bucket, object_name, using_json=True)
    assert isinstance(result, dict)
    assert result["test"] is True


@pytest.mark.parametrize(
    "bucket_prefix, bucket, bucket_name, object_name, object_path",
    [
        (BUCKET_PREFIX, "2", f"{BUCKET_PREFIX}2", "my_test_file_read.json", "my_test_file_read.json"),
        (f"test/{BUCKET_PREFIX}", "2", "test", "my_test_file_read.json", f"{BUCKET_PREFIX}2/my_test_file_read.json"),
        (
            f"test/reportportal/{BUCKET_PREFIX}",
            "2",
            "test",
            "my_test_file_write.json",
            f"reportportal/{BUCKET_PREFIX}2/my_test_file_write.json",
        ),
    ],
)
def test_json_read(bucket_prefix, bucket, bucket_name, object_name, object_path):
    boto3_client = create_storage_client(bucket_prefix)

    # Write through the client to set up the test
    boto3_client.put_project_object({"test": True}, bucket, object_name, using_json=True)

    result = boto3_client.get_project_object(bucket, object_name, using_json=True)
    assert isinstance(result, dict)
    assert result["test"] is True


@pytest.mark.parametrize(
    "bucket_prefix, bucket, bucket_name, object_name, object_path",
    [
        (
            BUCKET_PREFIX,
            "2",
            f"{BUCKET_PREFIX}2",
            "my_test_file.json",
            "my_test_file.json",
        ),
        (
            f"test/{BUCKET_PREFIX}",
            "2",
            "test",
            "my_test_file.json",
            f"{BUCKET_PREFIX}2/my_test_file.json",
        ),
    ],
)
def test_not_existing_file_get(bucket_prefix, bucket, bucket_name, object_name, object_path):
    boto3_client = create_storage_client(bucket_prefix)

    with pytest.raises(ValueError) as exc:
        boto3_client.get_project_object("2", object_name)
    assert exc.value.args[0] == f'Unable to get file in bucket "{bucket_name}" with path "{object_path}"'


@pytest.mark.parametrize("bucket", ["2", "3"])
def test_remove_not_existing_folder(bucket):
    path = "test-remove-not-existing"
    boto3_client = create_storage_client()

    assert not boto3_client.remove_folder_objects(bucket, path)


@pytest.mark.parametrize(
    "bucket_prefix, bucket, bucket_name, object_name, path, object_path",
    [
        (
            BUCKET_PREFIX,
            "5",
            f"{BUCKET_PREFIX}5",
            "my_test_file_folder.json",
            "folder",
            "folder/my_test_file_folder.json",
        ),
        (
            f"test/{BUCKET_PREFIX}",
            "5",
            "test",
            "my_test_file_folder.json",
            "folder",
            f"{BUCKET_PREFIX}5/folder/my_test_file_folder.json",
        ),
    ],
)
def test_get_existing_folder(bucket_prefix, bucket, bucket_name, object_name, path, object_path):
    resource = "/".join([path, object_name])

    boto3_client = create_storage_client(bucket_prefix)
    boto3_client.put_project_object({"test": True}, bucket, resource)

    # Verify the object exists
    assert boto3_client.does_object_exists(bucket, resource)


@pytest.mark.parametrize(
    "bucket_prefix, bucket, bucket_name, object_name, path, object_path",
    [
        (
            BUCKET_PREFIX,
            "5",
            f"{BUCKET_PREFIX}5",
            "my_test_file_folder.json",
            "folder",
            "folder/my_test_file_folder.json",
        ),
        (
            f"test/{BUCKET_PREFIX}",
            "5",
            "test",
            "my_test_file_folder.json",
            "folder",
            f"{BUCKET_PREFIX}5/folder/my_test_file_folder.json",
        ),
    ],
)
def test_remove_existing_folder(bucket_prefix, bucket, bucket_name, object_name, path, object_path):
    resource = "/".join([path, object_name])

    boto3_client = create_storage_client(bucket_prefix)
    boto3_client.put_project_object({"test": True}, bucket, resource)

    assert boto3_client.remove_folder_objects(bucket, path)
    # Verify the object was removed
    assert not boto3_client.does_object_exists(bucket, resource)


def test_list_not_existing_folder():
    path = "test"
    boto3_client = create_storage_client()

    assert boto3_client.get_folder_objects("4", path) == []


@pytest.mark.parametrize(
    "bucket_prefix, bucket, bucket_name, object_name, path, object_path",
    [
        (BUCKET_PREFIX, "6", f"{BUCKET_PREFIX}6", "my_test_file_list.json", "list", "list/my_test_file_list.json"),
        (
            f"test/{BUCKET_PREFIX}",
            "6",
            "test",
            "my_test_file_list.json",
            "list",
            f"{BUCKET_PREFIX}6/list/my_test_file_list.json",
        ),
    ],
)
def test_list_existing_folder(bucket_prefix, bucket, bucket_name, object_name, path, object_path):
    resource = "/".join([path, object_name])

    boto3_client = create_storage_client(bucket_prefix)
    boto3_client.put_project_object({"test": True}, bucket, resource, using_json=True)

    assert boto3_client.get_folder_objects(bucket, path) == [resource]


def test_list_dir_separators():
    bucket = "7"
    object_name = f"{random_alphanumeric(16)}.json"
    path = "test/"
    resource = path + object_name

    boto3_client = create_storage_client()
    boto3_client.put_project_object({"test": True}, bucket, resource, using_json=True)

    assert boto3_client.get_folder_objects(bucket, path) == [resource]


@pytest.mark.parametrize(
    "bucket_prefix, bucket, bucket_name, object_name, path, object_path",
    [
        (
            BUCKET_PREFIX,
            "8",
            f"{BUCKET_PREFIX}8",
            "my_test_file_objects.json",
            "objects",
            "objects/my_test_file_objects.json",
        ),
        (
            f"test/{BUCKET_PREFIX}",
            "8",
            "test",
            "my_test_file_objects.json",
            "objects",
            f"{BUCKET_PREFIX}8/objects/my_test_file_objects.json",
        ),
    ],
)
def test_remove_project_objects(bucket_prefix, bucket, bucket_name, object_name, path, object_path):
    resource = "/".join([path, object_name])

    boto3_client = create_storage_client(bucket_prefix)
    boto3_client.put_project_object({"test": True}, bucket, resource, using_json=True)

    result = boto3_client.get_project_object(bucket, resource, using_json=True)
    assert isinstance(result, dict)
    assert result["test"] is True

    boto3_client.remove_project_objects(bucket, [resource])
    with pytest.raises(ValueError):
        boto3_client.get_project_object(bucket, resource)
