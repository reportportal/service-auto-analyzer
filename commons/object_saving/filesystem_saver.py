"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""
import logging
import pickle
import os
import shutil
import json

logger = logging.getLogger("analyzerApp.filesystemSaver")


class FilesystemSaver:

    def __init__(self, app_config):
        self.app_config = app_config
        self.folder_storage = self.app_config["filesystemDefaultPath"]

    def remove_project_objects(self, project_id, object_names):
        try:
            for filename in object_names:
                object_name_full = os.path.join(
                    self.folder_storage, project_id, filename).replace("\\", "/")
                if os.path.exists(object_name_full):
                    os.remove(object_name_full)
        except Exception as err:
            logger.error(err)

    def put_project_object(self, data, project_id, object_name, using_json=False):
        try:
            folder_to_save = os.path.join(
                self.folder_storage, project_id, os.path.dirname(object_name)).replace("\\", "/")
            filename = os.path.join(
                self.folder_storage, project_id, object_name).replace("\\", "/")
            os.makedirs(folder_to_save, exist_ok=True)
            with open(filename, "wb") as f:
                if using_json:
                    f.write(json.dumps(data).encode("utf-8"))
                else:
                    pickle.dump(data, f)
            logger.debug(
                "Saved into folder '%s' with name '%s': %s", project_id, object_name, data)
        except Exception as err:
            logger.error(err)

    def get_project_object(self, project_id, object_name, using_json=False):
        try:
            filename = os.path.join(
                self.folder_storage, project_id, object_name).replace("\\", "/")
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    return json.loads(f.read()) if using_json else pickle.load(f)
        except Exception as err:
            logger.error(err)
        return {}

    def does_object_exists(self, project_id, object_name):
        return os.path.exists(
            os.path.join(self.folder_storage, project_id, object_name).replace("\\", "/"))

    def get_folder_objects(self, project_id, folder):
        folder_to_check = os.path.join(
            self.folder_storage, project_id, folder).replace("\\", "/")
        if os.path.exists(folder_to_check):
            return [
                os.path.join(folder, file_name) for file_name in os.listdir(folder_to_check)]
        return []

    def remove_folder_objects(self, project_id, folder):
        try:
            folder_name = os.path.join(self.folder_storage,
                                       project_id, folder).replace("\\", "/")
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name, ignore_errors=True)
                return 1
            return 0
        except Exception as err:
            logger.error(err)
            return 0
