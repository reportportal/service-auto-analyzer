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
from typing import Any, Callable

from app.commons import logging
from commons.model.launch_objects import ApplicationConfig
from app.commons.object_saving.filesystem_saver import FilesystemSaver
from app.commons.object_saving.minio_client import MinioClient
from app.commons.object_saving.storage import Storage

logger = logging.getLogger("analyzerApp.objectSaver")


def create_minio_client(app_config: ApplicationConfig) -> Storage:
    return MinioClient(app_config)


def create_filesystem_client(app_config: ApplicationConfig) -> Storage:
    return FilesystemSaver(app_config)


STORAGE_FACTORIES: dict[str, Callable[[ApplicationConfig], Storage]] = {
    'minio': create_minio_client,
    'filesystem': create_filesystem_client
}


class ObjectSaver:
    storage: Storage
    project_id: str | int | None = None
    path: str

    def __init__(self, app_config: ApplicationConfig, project_id: str | int | None = None,
                 path: str | None = None) -> None:
        self.project_id = project_id
        self.path = path or ""
        if app_config.binaryStoreType in STORAGE_FACTORIES:
            self.storage = STORAGE_FACTORIES[app_config.binaryStoreType](app_config)
        else:
            raise ValueError(
                f'Storage "{app_config.binaryStoreType}" is not supported, possible types are: '
                + str(STORAGE_FACTORIES.keys())
            )

    def get_project_id(self, project_id: str | int | None):
        if project_id is not None:
            return str(project_id)
        if self.project_id is not None:
            return str(self.project_id)
        return ""

    def get_object_name(self, object_names: str) -> str:
        return os.path.join(self.path, object_names)

    def remove_project_objects(self, object_names: list[str], project_id: str | int | None = None) -> None:
        self.storage.remove_project_objects(self.get_project_id(project_id),
                                            [self.get_object_name(n) for n in object_names])

    def put_project_object(self, data: Any, object_name: str, project_id: str | int | None = None,
                           using_json: bool = False) -> None:
        self.storage.put_project_object(data, self.get_project_id(project_id), self.get_object_name(object_name),
                                        using_json=using_json)

    def get_project_object(self, object_name: str, project_id: str | int | None = None,
                           using_json: bool = False) -> Any:
        return self.storage.get_project_object(self.get_project_id(project_id), self.get_object_name(object_name),
                                               using_json=using_json)

    def does_object_exists(self, object_name: str, project_id: str | int | None = None) -> bool:
        return self.storage.does_object_exists(self.get_project_id(project_id), self.get_object_name(object_name))

    def get_folder_objects(self, folder: str, project_id: str | int | None = None) -> list:
        return self.storage.get_folder_objects(self.get_project_id(project_id), self.get_object_name(folder))

    def remove_folder_objects(self, folder: str, project_id: str | int | None = None) -> bool:
        return self.storage.remove_folder_objects(self.get_project_id(project_id), self.get_object_name(folder))
