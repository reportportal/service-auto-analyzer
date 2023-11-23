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
import pytest

from app.commons.object_saving.filesystem_saver import FilesystemSaver
from test import random_alphanumeric


@pytest.mark.parametrize('base_path', ['test_base_path', '', None])
def test_base_path(base_path):
    object_name = f'{random_alphanumeric(16)}.pickle'
    path = random_alphanumeric(16)
    file_system = FilesystemSaver({'filesystemDefaultPath': base_path})

    assert not file_system.does_object_exists('', object_name)

    file_system.put_project_object({'test': True}, '', object_name)
    file_system.put_project_object({'test': True}, path, object_name)

    if base_path:
        expected_path = os.path.join(base_path, object_name)
        expected_directory = os.path.join(os.getcwd(), base_path)
    else:
        expected_path = object_name
        expected_directory = os.getcwd()
    assert os.path.exists(expected_path)
    assert os.path.isfile(expected_path)
    assert file_system.does_object_exists('', object_name)

    result = file_system.get_project_object('', object_name)
    assert isinstance(result, dict)
    assert result['test']

    assert file_system.get_folder_objects('', '') == os.listdir(expected_directory)

    assert file_system.remove_folder_objects('', path)
    assert not os.path.exists(os.path.join(expected_directory, path))

    file_system.remove_project_objects('', [object_name])
    assert not os.path.exists(expected_path)


def test_json_write():
    base_path = 'test'
    object_name = f'{random_alphanumeric(16)}.json'
    file_system = FilesystemSaver({'filesystemDefaultPath': base_path})

    expected_path = os.path.join(base_path, object_name)
    file_system.put_project_object({'test': True}, '', object_name, using_json=True)

    with open(expected_path, 'r') as f:
        assert f.readline() == '{"test": true}'


def test_json_read():
    base_path = 'test'
    object_name = f'{random_alphanumeric(16)}.json'
    file_system = FilesystemSaver({'filesystemDefaultPath': base_path})
    expected_path = os.path.join(base_path, object_name)

    with open(expected_path, 'w') as f:
        f.writelines(['{"test": true}'])

    result = file_system.get_project_object('', object_name, using_json=True)
    assert isinstance(result, dict)
    assert result['test'] is True


def test_not_existing_file_get():
    base_path = 'test'
    object_name = f'{random_alphanumeric(16)}.json'
    file_system = FilesystemSaver({'filesystemDefaultPath': base_path})
    expected_path = os.path.join(base_path, object_name)

    with pytest.raises(ValueError) as exc:
        file_system.get_project_object('', object_name)
    assert exc.value.args[0] == f'Unable to get file: {expected_path}'


def test_remove_not_existing_folder():
    base_path = f'test_{random_alphanumeric(16)}'
    path = 'test'
    file_system = FilesystemSaver({'filesystemDefaultPath': base_path})

    assert not file_system.remove_folder_objects('', path)


def test_list_not_existing_folder():
    base_path = f'test_{random_alphanumeric(16)}'
    path = 'test'
    file_system = FilesystemSaver({'filesystemDefaultPath': base_path})

    assert file_system.get_folder_objects('', path) == []
