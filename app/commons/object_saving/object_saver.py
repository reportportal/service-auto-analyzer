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

from typing import Callable

from app.commons import logging
from app.commons.object_saving.filesystem_saver import FilesystemSaver
from app.commons.object_saving.minio_client import MinioClient
from app.commons.protocols import Storage

logger = logging.getLogger("analyzerApp.objectSaver")


def create_minio_client(app_config) -> Storage:
    return MinioClient(app_config)


def create_filesystem_client(app_config) -> Storage:
    return FilesystemSaver(app_config)


STORAGE_FACTORIES: dict[str, Callable[[dict], Storage]] = {
    'minio': create_minio_client,
    'filesystem': create_filesystem_client
}


class ObjectSaver:
    storage: Storage

    def __init__(self, app_config: dict) -> None:
        self.app_config = app_config
        if "binaryStoreType" in self.app_config and self.app_config["binaryStoreType"] in STORAGE_FACTORIES:
            self.storage = STORAGE_FACTORIES[self.app_config["binaryStoreType"]](app_config)

    def get_bucket_name(self, project_id):
        return self.app_config["minioBucketPrefix"] + str(project_id)

    def remove_project_objects(self, project_id, object_names):
        self.storage.remove_project_objects(
            self.get_bucket_name(project_id), object_names)

    def put_project_object(self, data, project_id, object_name, using_json=False):
        self.storage.put_project_object(data, self.get_bucket_name(project_id), object_name, using_json=using_json)

    def get_project_object(self, project_id, object_name, using_json=False):
        return self.storage.get_project_object(self.get_bucket_name(project_id), object_name, using_json=using_json)

    def does_object_exists(self, project_id, object_name):
        return self.storage.does_object_exists(self.get_bucket_name(project_id), object_name)

    def get_folder_objects(self, project_id, folder):
        return self.storage.get_folder_objects(self.get_bucket_name(project_id), folder)

    def remove_folder_objects(self, project_id, folder):
        return self.storage.remove_folder_objects(self.get_bucket_name(project_id), folder)
