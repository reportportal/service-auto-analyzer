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
from app.commons.interfaces.storage import Storage

logger = logging.getLogger("analyzerApp.filesystemSaver")


class FilesystemSaver(Storage):
    app_config: dict[str, Any]
    folder_storage: str

    def __init__(self, app_config: dict[str, Any]) -> None:
        self.app_config = app_config
        self.folder_storage = self.app_config["filesystemDefaultPath"]

    def remove_project_objects(self, project_id, object_names) -> None:
        for filename in object_names:
            object_name_full = os.path.join(self.folder_storage, project_id, filename).replace("\\", "/")
            if os.path.exists(object_name_full):
                os.remove(object_name_full)

    def put_project_object(self, data, project_id, object_name, using_json=False) -> None:
        folder_to_save = os.path.join(self.folder_storage, project_id, os.path.dirname(object_name)).replace("\\", "/")
        filename = os.path.join(self.folder_storage, project_id, object_name).replace("\\", "/")
        os.makedirs(folder_to_save, exist_ok=True)
        with open(filename, "wb") as f:
            if using_json:
                f.write(json.dumps(data).encode("utf-8"))
            else:
                pickle.dump(data, f)
        logger.debug("Saved into folder '%s' with name '%s': %s", project_id, object_name, data)

    def get_project_object(self, project_id, object_name, using_json=False) -> object | None:
        filename = os.path.join(self.folder_storage, project_id, object_name).replace("\\", "/")
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return json.loads(f.read()) if using_json else pickle.load(f)

    def does_object_exists(self, project_id, object_name) -> bool:
        return os.path.exists(os.path.join(self.folder_storage, project_id, object_name).replace("\\", "/"))

    def get_folder_objects(self, project_id, folder) -> list[str]:
        folder_to_check = os.path.join(self.folder_storage, project_id, folder).replace("\\", "/")
        if os.path.exists(folder_to_check):
            return [os.path.join(folder, file_name) for file_name in os.listdir(folder_to_check)]
        return []

    def remove_folder_objects(self, project_id, folder) -> bool:
        folder_name = os.path.join(self.folder_storage, project_id, folder).replace("\\", "/")
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name, ignore_errors=True)
            return True
        return False
