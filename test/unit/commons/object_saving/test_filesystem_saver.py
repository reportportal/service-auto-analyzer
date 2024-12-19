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

import os
# noinspection PyPackageRequirements
import pytest

from app.commons.object_saving.filesystem_saver import FilesystemSaver
from app.commons.model.launch_objects import ApplicationConfig
from test import random_alphanumeric


CREATED_FILES_AND_FOLDERS = []


def create_storage_client(path, bucket_prefix=None):
    if bucket_prefix is None:
        return FilesystemSaver(ApplicationConfig(filesystemDefaultPath=path))
    else:
        return FilesystemSaver(ApplicationConfig(filesystemDefaultPath=path, bucketPrefix=bucket_prefix))


def test_object_not_exists():
    base_path = f'test_{random_alphanumeric(16)}'
    object_name = f'{random_alphanumeric(16)}.pickle'
    file_system = create_storage_client(base_path)

    assert not file_system.does_object_exists('', object_name)


def test_object_exists():
    base_path = f'test_{random_alphanumeric(16)}'
    object_name = f'{random_alphanumeric(16)}.pickle'
    file_system = create_storage_client(base_path)
    expected_path = os.path.join(base_path, object_name)
    CREATED_FILES_AND_FOLDERS.append(expected_path)
    CREATED_FILES_AND_FOLDERS.append(base_path)

    file_system.put_project_object({'test': True}, '', object_name)

    assert file_system.does_object_exists('', object_name)


def test_json_write():
    base_path = 'test'
    object_name = f'{random_alphanumeric(16)}.json'
    file_system = create_storage_client(base_path)

    expected_path = os.path.join(base_path, object_name)

    CREATED_FILES_AND_FOLDERS.append(expected_path)
    CREATED_FILES_AND_FOLDERS.append(base_path)

    file_system.put_project_object({'test': True}, '', object_name, using_json=True)

    with open(expected_path, 'r') as f:
        assert f.readline() == '{"test": true}'


def test_json_read():
    base_path = 'test'
    object_name = f'{random_alphanumeric(16)}.json'
    file_system = create_storage_client(base_path)
    expected_path = os.path.join(base_path, object_name)

    CREATED_FILES_AND_FOLDERS.append(expected_path)
    CREATED_FILES_AND_FOLDERS.append(base_path)

    with open(expected_path, 'w') as f:
        f.writelines(['{"test": true}'])

    result = file_system.get_project_object('', object_name, using_json=True)
    assert isinstance(result, dict)
    assert result['test'] is True


def test_not_existing_file_get():
    base_path = 'test'
    object_name = f'{random_alphanumeric(16)}.json'
    file_system = create_storage_client(base_path)
    expected_path = os.path.join(base_path, object_name)

    with pytest.raises(ValueError) as exc:
        file_system.get_project_object('', object_name)
    assert exc.value.args[0] == f'Unable to get file: {expected_path}'


def test_remove_not_existing_folder():
    base_path = f'test_{random_alphanumeric(16)}'
    path = 'test'
    file_system = create_storage_client(base_path)

    assert not file_system.remove_folder_objects('', path)


def test_remove_existing_folder():
    base_path = f'test_{random_alphanumeric(16)}'
    path = 'test'
    expected_path = os.path.join(base_path, path)
    os.makedirs(expected_path)
    CREATED_FILES_AND_FOLDERS.append(base_path)

    file_system = create_storage_client(base_path)

    assert file_system.remove_folder_objects('', path)
    assert not os.path.exists(expected_path)


def test_list_not_existing_folder():
    base_path = f'test_{random_alphanumeric(16)}'
    path = 'test'
    file_system = create_storage_client(base_path)

    assert file_system.get_folder_objects('', path) == []


def test_list_existing_folder():
    bucket = '6'
    base_path = f'test_{random_alphanumeric(16)}'
    object_name = f'{random_alphanumeric(16)}.json'
    path = 'test'
    resource = '/'.join([path, object_name])
    CREATED_FILES_AND_FOLDERS.append(os.path.join(base_path, f'prj-{bucket}', path, object_name))
    CREATED_FILES_AND_FOLDERS.append(os.path.join(base_path, f'prj-{bucket}', path))
    CREATED_FILES_AND_FOLDERS.append(os.path.join(base_path, f'prj-{bucket}'))
    CREATED_FILES_AND_FOLDERS.append(base_path)

    file_system = create_storage_client(base_path)
    file_system.put_project_object({'test': True}, bucket, resource, using_json=True)

    assert file_system.get_folder_objects(bucket, path) == [path]


def test_list_dir_separators():
    bucket = '7'
    object_name = f'{random_alphanumeric(16)}.json'
    path = 'test/'
    resource = path + object_name
    CREATED_FILES_AND_FOLDERS.append(os.path.join(f'prj-{bucket}', path, object_name))
    CREATED_FILES_AND_FOLDERS.append(os.path.join(f'prj-{bucket}', path))
    CREATED_FILES_AND_FOLDERS.append(os.path.join(f'prj-{bucket}'))

    file_system = create_storage_client('')
    file_system.put_project_object({'test': True}, bucket, resource, using_json=True)

    assert file_system.get_folder_objects(bucket, path) == [resource]


def test_remove_project_objects():
    bucket = '8'
    object_name = f'{random_alphanumeric(16)}.json'
    path = 'test/'
    resource = path + object_name

    file_system = create_storage_client('')
    file_system.put_project_object({'test': True}, bucket, resource, using_json=True)
    CREATED_FILES_AND_FOLDERS.append(os.path.join(f'prj-{bucket}', path))
    CREATED_FILES_AND_FOLDERS.append(os.path.join(f'prj-{bucket}'))

    file_system.remove_project_objects(bucket, [resource])
    with pytest.raises(ValueError):
        file_system.get_project_object(bucket, resource)


@pytest.mark.parametrize('base_path', ['test_base_path', ''])
def test_base_path(base_path):
    object_name = f'{random_alphanumeric(16)}.pickle'
    file_system = create_storage_client(base_path)

    file_system.put_project_object({'test': True}, '', object_name)

    if base_path:
        expected_path = os.path.join(base_path, object_name)
        expected_directory = os.path.join(os.getcwd(), base_path)
        CREATED_FILES_AND_FOLDERS.append(base_path)
    else:
        expected_path = object_name
        expected_directory = os.getcwd()
    CREATED_FILES_AND_FOLDERS.append(expected_path)
    assert os.path.exists(expected_path)
    assert os.path.isfile(expected_path)

    result = file_system.get_project_object('', object_name)
    assert isinstance(result, dict)
    assert result['test']

    assert file_system.get_folder_objects('', '') == os.listdir(expected_directory)


def test_bucket_prefix():
    bucket = '8'
    base_path = f'storage_{random_alphanumeric(16)}'
    bucket_prefix = f'test_{random_alphanumeric(16)}/prj-'
    object_name = f'{random_alphanumeric(16)}.json'

    file_system = create_storage_client(base_path, bucket_prefix)
    file_system.put_project_object({'test': True}, bucket, object_name, using_json=True)

    resource = os.path.join(base_path, f'{bucket_prefix}{bucket}', object_name)
    CREATED_FILES_AND_FOLDERS.append(resource)
    CREATED_FILES_AND_FOLDERS.append(os.path.join(base_path, f'{bucket_prefix}{bucket}'))
    CREATED_FILES_AND_FOLDERS.append(base_path)

    assert file_system.get_folder_objects(bucket, '') == [object_name]
    assert file_system.does_object_exists(bucket, object_name)
    assert file_system.get_project_object(bucket, object_name, using_json=True) == {'test': True}
    assert os.path.exists(resource)


@pytest.fixture(autouse=True, scope='session')
def clean_up():
    yield
    for file in CREATED_FILES_AND_FOLDERS:
        if os.path.exists(file):
            if os.path.isdir(file):
                try:
                    os.removedirs(file)
                except OSError:
                    pass
            else:
                os.remove(file)
