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
from app.commons.model.launch_objects import ApplicationConfig
from app.commons.object_saving.storage import Storage, unify_path_separator
from app.utils import utils

logger = logging.getLogger("analyzerApp.filesystemSaver")


class FilesystemSaver(Storage):
    _base_path: str

    def __init__(self, app_config: ApplicationConfig) -> None:
        super().__init__(app_config)
        self._base_path = app_config.filesystemDefaultPath or os.getcwd()

    def get_path(self, object_name: str, bucket) -> str:
        return unify_path_separator(str(os.path.join(self._base_path, self._get_project_name(bucket), object_name)))

    def remove_project_objects(self, bucket: str, object_names: list[str]) -> None:
        for filename in object_names:
            object_name_full = self.get_path(filename, bucket)
            if os.path.exists(object_name_full):
                os.remove(object_name_full)

    def put_project_object(self, data: Any, bucket: str, object_name: str, using_json: bool = False) -> None:
        path = self.get_path(object_name, bucket)
        folder_to_save = unify_path_separator(os.path.dirname(path))
        if folder_to_save:
            os.makedirs(folder_to_save, exist_ok=True)
        with open(path, "wb") as f:
            if using_json:
                f.write(json.dumps(data).encode("utf-8"))
            else:
                # noinspection PyTypeChecker
                pickle.dump(data, f)
        logger.debug('Saved into folder "%s" with name "%s": %s', bucket, object_name, data)

    def get_project_object(self, bucket: str, object_name: str, using_json: bool = False) -> object | None:
        filename = self.get_path(object_name, bucket)
        if not utils.validate_file(filename):
            raise ValueError(f"Unable to get file: {filename}")
        with open(filename, "rb") as f:
            return json.loads(f.read()) if using_json else pickle.load(f)

    def does_object_exists(self, bucket: str, object_name: str) -> bool:
        return os.path.exists(self.get_path(object_name, bucket))

    def get_folder_objects(self, bucket: str, folder: str) -> list[str]:
        path = self.get_path(folder, bucket)
        if unify_path_separator(folder).endswith("/"):
            # The "folder" is a folder, list it
            if os.path.exists(path):
                return [os.path.join(folder, file_name) for file_name in os.listdir(path)]
        else:
            # The "folder" is a filename pattern, list base folder and filter files
            folder_to_check = unify_path_separator(os.path.dirname(path))
            if os.path.exists(folder_to_check):
                return [file_name for file_name in os.listdir(folder_to_check) if file_name.startswith(folder)]
        return []

    def remove_folder_objects(self, bucket: str, folder: str) -> bool:
        folder_name = self.get_path(folder, bucket)
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name, ignore_errors=True)
            return True
        return False
