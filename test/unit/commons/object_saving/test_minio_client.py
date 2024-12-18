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
import requests
# noinspection PyPackageRequirements
from moto.server import ThreadedMotoServer

from app.commons.model.launch_objects import ApplicationConfig
from app.commons.object_saving.minio_client import MinioClient
from test import random_alphanumeric

SERVER_PORT = 5123
REGION = 'us-west-1'
BUCKET_PREFIX = 'prj-'
SERVER_HOST = f'localhost:{SERVER_PORT}'


@pytest.fixture(autouse=True, scope='session')
def run_s3():
    server = ThreadedMotoServer(port=SERVER_PORT, verbose=True)
    server.start()
    yield
    server.stop()


def create_storage_client(bucket_prefix=BUCKET_PREFIX):
    return MinioClient(ApplicationConfig(minioHost=SERVER_HOST, minioRegion=REGION, bucketPrefix=bucket_prefix,
                                         minioAccessKey='minio', minioSecretKey='minio', minioUseTls=False))


@pytest.mark.parametrize('bucket_prefix, bucket, object_name', [
    (BUCKET_PREFIX, '2', f'{random_alphanumeric(16)}.pickle'),
    (f'test/{BUCKET_PREFIX}', '2', f'{random_alphanumeric(16)}.json'),
])
def test_object_not_exists(bucket_prefix, bucket, object_name):
    minio_client = create_storage_client(bucket_prefix)

    assert not minio_client.does_object_exists('2', object_name)


@pytest.mark.parametrize('bucket_prefix, bucket, object_name', [
    (BUCKET_PREFIX, '2', f'{random_alphanumeric(16)}.pickle'),
    (f'test/{BUCKET_PREFIX}', '2', f'{random_alphanumeric(16)}.json'),
])
def test_object_exists(bucket_prefix, bucket, object_name):
    minio_client = create_storage_client(bucket_prefix)

    minio_client.put_project_object({'test': True}, '2', object_name)

    assert minio_client.does_object_exists('2', object_name)


def get_url(bucket, object_name):
    # noinspection HttpUrlsUsage
    return f'http://{SERVER_HOST}/{bucket}/{object_name}'


@pytest.mark.parametrize('bucket_prefix, bucket, bucket_name, object_name, object_path', [
    (BUCKET_PREFIX, '2', f'{BUCKET_PREFIX}2', 'my_test_file_write.json', 'my_test_file_write.json',),
    (f'test/{BUCKET_PREFIX}', '2', 'test', 'my_test_file_write.json', f'{BUCKET_PREFIX}2/my_test_file_write.json',),
])
def test_json_write(bucket_prefix, bucket, bucket_name, object_name, object_path):
    minio_client = create_storage_client(bucket_prefix)

    minio_client.put_project_object({'test': True}, bucket, object_name, using_json=True)

    headers = {
        'x-amz-date': '20231124T123217Z',
        'x-amz-content-sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
        'authorization': 'AWS4-HMAC-SHA256 Credential=minio/20231124/us-west-1/s3/aws4_request, '
                         'SignedHeaders=host;user-agent;x-amz-content-sha256;x-amz-date, '
                         'Signature=dc971726ff2b266f208b250089b2ba0be86352efad2858145b33c2ae085e7d71'
    }
    response = requests.get(get_url(bucket_name, object_path), headers=headers)
    assert response.text == '{"test": true}'


@pytest.mark.parametrize('bucket_prefix, bucket, bucket_name, object_name, object_path', [
    (BUCKET_PREFIX, '2', f'{BUCKET_PREFIX}2', 'my_test_file_read.json', 'my_test_file_read.json',),
    (f'test/{BUCKET_PREFIX}', '2', 'test', 'my_test_file_read.json', f'{BUCKET_PREFIX}2/my_test_file_read.json',),
])
def test_json_read(bucket_prefix, bucket, bucket_name, object_name, object_path):
    minio_client = create_storage_client(bucket_prefix)

    headers = {
        'x-amz-date': '20231124T124147Z',
        'x-amz-content-sha256': '80f65706d935d3b928d95207937dd81bad43ab56cd4d3b7ed41772318e734168',
        'authorization': 'AWS4-HMAC-SHA256 Credential=minio/20231124/us-west-1/s3/aws4_request, '
                         'SignedHeaders=content-length;content-type;host;user-agent;x-amz-content-sha256;x-amz-date, '
                         'Signature=d592f084a4f9fd46a8624a37323b5be843120bd9e7c075c925faea573f00511e'
    }
    requests.put(get_url(bucket_name, object_path), headers=headers, data='{"test": true}'.encode('utf-8'))

    result = minio_client.get_project_object(bucket, object_name, using_json=True)
    assert isinstance(result, dict)
    assert result['test'] is True


@pytest.mark.parametrize('bucket_prefix, bucket, bucket_name, object_name, object_path', [
    (BUCKET_PREFIX, '2', f'{BUCKET_PREFIX}2', 'my_test_file.json', 'my_test_file.json',),
    (f'test/{BUCKET_PREFIX}', '2', 'test', 'my_test_file.json', f'{BUCKET_PREFIX}2/my_test_file.json',),
])
def test_not_existing_file_get(bucket_prefix, bucket, bucket_name, object_name, object_path):
    minio_client = create_storage_client(bucket_prefix)

    with pytest.raises(ValueError) as exc:
        minio_client.get_project_object('2', object_name)
    assert exc.value.args[0] == f'Unable to get file in bucket "{bucket_name}" with path "{object_path}"'


@pytest.mark.parametrize('bucket', ['2', '3'])
def test_remove_not_existing_folder(bucket):
    path = 'test-remove-not-existing'
    minio_client = create_storage_client()

    assert not minio_client.remove_folder_objects(bucket, path)


@pytest.mark.parametrize('bucket_prefix, bucket, bucket_name, object_name, path, object_path', [
    (BUCKET_PREFIX, '5', f'{BUCKET_PREFIX}5', 'my_test_file_folder.json', 'folder', 'folder/my_test_file_folder.json'),
    (f'test/{BUCKET_PREFIX}', '5', 'test', 'my_test_file_folder.json', 'folder',
     f'{BUCKET_PREFIX}5/folder/my_test_file_folder.json'),
])
def test_get_existing_folder(bucket_prefix, bucket, bucket_name, object_name, path, object_path):
    resource = '/'.join([path, object_name])

    minio_client = create_storage_client(bucket_prefix)
    minio_client.put_project_object({'test': True}, bucket, resource)

    headers = {
        'x-amz-date': '20231124T123217Z',
        'x-amz-content-sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
        'authorization': 'AWS4-HMAC-SHA256 Credential=minio/20231124/us-west-1/s3/aws4_request, '
                         'SignedHeaders=host;user-agent;x-amz-content-sha256;x-amz-date, '
                         'Signature=dc971726ff2b266f208b250089b2ba0be86352efad2858145b33c2ae085e7d71'
    }
    resource_url = get_url(bucket_name, object_path)
    response = requests.get(resource_url, headers=headers)
    assert response.status_code == 200, f'Failed to get file from {resource_url}'


@pytest.mark.parametrize('bucket_prefix, bucket, bucket_name, object_name, path, object_path', [
    (BUCKET_PREFIX, '5', f'{BUCKET_PREFIX}5', 'my_test_file_folder.json', 'folder', 'folder/my_test_file_folder.json'),
    (f'test/{BUCKET_PREFIX}', '5', 'test', 'my_test_file_folder.json', 'folder',
     f'{BUCKET_PREFIX}5/folder/my_test_file_folder.json'),
])
def test_remove_existing_folder(bucket_prefix, bucket, bucket_name, object_name, path, object_path):
    resource = '/'.join([path, object_name])

    minio_client = create_storage_client(bucket_prefix)
    minio_client.put_project_object({'test': True}, bucket, resource)

    assert minio_client.remove_folder_objects(bucket, path)
    headers = {
        'x-amz-date': '20231124T123217Z',
        'x-amz-content-sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
        'authorization': 'AWS4-HMAC-SHA256 Credential=minio/20231124/us-west-1/s3/aws4_request, '
                         'SignedHeaders=host;user-agent;x-amz-content-sha256;x-amz-date, '
                         'Signature=dc971726ff2b266f208b250089b2ba0be86352efad2858145b33c2ae085e7d71'
    }
    resource_url = get_url(bucket_name, object_path)
    response = requests.get(resource_url, headers=headers)
    assert response.status_code == 404, f'Resource {resource_url} was not removed'


def test_list_not_existing_folder():
    path = 'test'
    minio_client = create_storage_client()

    assert minio_client.get_folder_objects('4', path) == []


@pytest.mark.parametrize('bucket_prefix, bucket, bucket_name, object_name, path, object_path', [
    (BUCKET_PREFIX, '6', f'{BUCKET_PREFIX}6', 'my_test_file_list.json', 'list', 'list/my_test_file_list.json'),
    (f'test/{BUCKET_PREFIX}', '6', 'test', 'my_test_file_list.json', 'list',
     f'{BUCKET_PREFIX}6/list/my_test_file_list.json'),
])
def test_list_existing_folder(bucket_prefix, bucket, bucket_name, object_name, path, object_path):
    resource = '/'.join([path, object_name])

    minio_client = create_storage_client(bucket_prefix)
    minio_client.put_project_object({'test': True}, bucket, resource, using_json=True)

    assert minio_client.get_folder_objects(bucket, path) == [resource]


def test_list_dir_separators():
    bucket = '7'
    object_name = f'{random_alphanumeric(16)}.json'
    path = 'test/'
    resource = path + object_name

    minio_client = create_storage_client()
    minio_client.put_project_object({'test': True}, bucket, resource, using_json=True)

    assert minio_client.get_folder_objects(bucket, path) == [resource]


@pytest.mark.parametrize('bucket_prefix, bucket, bucket_name, object_name, path, object_path', [
    (BUCKET_PREFIX, '8', f'{BUCKET_PREFIX}8', 'my_test_file_objects.json', 'objects',
     'objects/my_test_file_objects.json'),
    (f'test/{BUCKET_PREFIX}', '8', 'test', 'my_test_file_objects.json', 'objects',
     f'{BUCKET_PREFIX}8/objects/my_test_file_objects.json'),
])
def test_remove_project_objects(bucket_prefix, bucket, bucket_name, object_name, path, object_path):
    resource = '/'.join([path, object_name])

    minio_client = create_storage_client(bucket_prefix)
    minio_client.put_project_object({'test': True}, bucket, resource, using_json=True)

    result = minio_client.get_project_object(bucket, resource, using_json=True)
    assert isinstance(result, dict)
    assert result['test'] is True

    minio_client.remove_project_objects(bucket, [resource])
    with pytest.raises(ValueError):
        minio_client.get_project_object(bucket, resource)
