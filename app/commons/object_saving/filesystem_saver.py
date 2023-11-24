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

import json
import os
import pickle
import shutil
from typing import Any

from app.commons import logging
from app.utils import utils
from app.commons.object_saving.storage import Storage

logger = logging.getLogger('analyzerApp.filesystemSaver')


class FilesystemSaver(Storage):
    _base_path: str

    def __init__(self, app_config: dict[str, Any]) -> None:
        self._base_path = app_config['filesystemDefaultPath'] or ''

    def remove_project_objects(self, path: str, object_names: list[str]) -> None:
        for filename in object_names:
            object_name_full = os.path.join(self._base_path, path, filename).replace("\\", "/")
            if os.path.exists(object_name_full):
                os.remove(object_name_full)

    def put_project_object(self, data: Any, path: str, object_name: str, using_json: bool = False) -> None:
        folder_to_save = os.path.join(self._base_path, path, os.path.dirname(object_name)).replace("\\", "/")
        filename = os.path.join(self._base_path, path, object_name).replace("\\", "/")
        if folder_to_save:
            os.makedirs(folder_to_save, exist_ok=True)
        with open(filename, "wb") as f:
            if using_json:
                f.write(json.dumps(data).encode("utf-8"))
            else:
                pickle.dump(data, f)
        logger.debug("Saved into folder '%s' with name '%s': %s", path, object_name, data)

    def get_project_object(self, path: str, object_name: str, using_json: bool = False) -> object | None:
        filename = os.path.join(self._base_path, path, object_name).replace("\\", "/")
        if not utils.validate_file(filename):
            raise ValueError(f'Unable to get file: {filename}')
        with open(filename, "rb") as f:
            return json.loads(f.read()) if using_json else pickle.load(f)

    def does_object_exists(self, path: str, object_name: str) -> bool:
        return os.path.exists(os.path.join(self._base_path, path, object_name).replace("\\", "/"))

    def get_folder_objects(self, path: str, folder: str) -> list[str]:
        root_path = self._base_path
        if not root_path and not path:
            root_path = os.getcwd()
        if folder.endswith('/'):
            folder_to_check = os.path.join(root_path, path, folder).replace("\\", "/")
            if os.path.exists(folder_to_check):
                return [os.path.join(folder, file_name) for file_name in os.listdir(folder_to_check)]
        else:
            folder_to_check = os.path.join(root_path, path).replace("\\", "/")
            if os.path.exists(folder_to_check):
                return [file_name for file_name in os.listdir(folder_to_check) if file_name.startswith(folder)]
        return []

    def remove_folder_objects(self, path: str, folder: str) -> bool:
        folder_name = os.path.join(self._base_path, path, folder).replace("\\", "/")
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name, ignore_errors=True)
            return True
        return False
