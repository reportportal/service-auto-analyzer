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

from typing import Any, Callable

from app.commons import logging
from app.commons.object_saving import Storage
from app.commons.object_saving.filesystem_saver import FilesystemSaver
from app.commons.object_saving.minio_client import MinioClient

logger = logging.getLogger("analyzerApp.objectSaver")


def create_minio_client(app_config: dict[str, Any]) -> Storage:
    return MinioClient(app_config)


def create_filesystem_client(app_config: dict[str, Any]) -> Storage:
    return FilesystemSaver(app_config)


STORAGE_FACTORIES: dict[str, Callable[[dict], Storage]] = {
    'minio': create_minio_client,
    'filesystem': create_filesystem_client
}

CONFIG_KEY = 'binaryStoreType'


class ObjectSaver:
    app_config: dict[str, Any]
    storage: Storage
    project_id: str | int | None = None

    def __init__(self, app_config: dict[str, Any], project_id: str | int | None = None) -> None:
        self.app_config = app_config
        self.project_id = project_id
        if CONFIG_KEY in self.app_config and self.app_config[CONFIG_KEY] in STORAGE_FACTORIES:
            self.storage = STORAGE_FACTORIES[self.app_config[CONFIG_KEY]](app_config)
        else:
            raise ValueError(
                f'Storage "{self.app_config.get(CONFIG_KEY, None)}" is not supported, possible types are: '
                + str(STORAGE_FACTORIES.keys())
            )

    def get_bucket_name(self, project_id: str | int | None = None) -> str:
        id_str = ''
        if project_id is None:
            if self.project_id is None:
                return id_str
            else:
                id_str = str(self.project_id)
        else:
            id_str = str(project_id)
        return self.app_config['minioBucketPrefix'] + id_str

    def remove_project_objects(self, object_names: list[str], project_id: str | int | None = None) -> None:
        self.storage.remove_project_objects(self.get_bucket_name(project_id), object_names)

    def put_project_object(self, data: object, object_name: str, project_id: str | int | None = None,
                           using_json: bool = False) -> None:
        self.storage.put_project_object(data, self.get_bucket_name(project_id), object_name, using_json=using_json)

    def get_project_object(self, object_name: str, project_id: str | int | None = None,
                           using_json: bool = False) -> object:
        return self.storage.get_project_object(self.get_bucket_name(project_id), object_name, using_json=using_json)

    def does_object_exists(self, object_name: str, project_id: str | int | None = None) -> bool:
        return self.storage.does_object_exists(self.get_bucket_name(project_id), object_name)

    def get_folder_objects(self, folder: str, project_id: str | int | None = None) -> list:
        return self.storage.get_folder_objects(self.get_bucket_name(project_id), folder)

    def remove_folder_objects(self, folder: str, project_id: str | int | None = None) -> bool:
        return self.storage.remove_folder_objects(self.get_bucket_name(project_id), folder)
