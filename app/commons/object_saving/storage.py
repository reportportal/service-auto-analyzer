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

"""Common interface class for Storage types."""
from abc import ABCMeta, abstractmethod
from typing import Any

from app.commons.model.launch_objects import ApplicationConfig


def unify_path_separator(path: str) -> str:
    return path.replace("\\", "/")


class Storage(metaclass=ABCMeta):
    _bucket_prefix: str

    def __init__(self, app_config: ApplicationConfig) -> None:
        self._bucket_prefix = app_config.bucketPrefix

    def _get_project_name(self, project_id: str | None) -> str:
        if not project_id:
            return ""
        return self._bucket_prefix + project_id

    @abstractmethod
    def remove_project_objects(self, path: str, object_names: list[str]) -> None:
        raise NotImplementedError('"remove_project_objects" method is not implemented!')

    @abstractmethod
    def put_project_object(self, data: Any, path: str, object_name: str, using_json: bool = False) -> None:
        raise NotImplementedError('"put_project_object" method is not implemented!')

    @abstractmethod
    def get_project_object(self, path: str, object_name: str, using_json: bool = False) -> object | None:
        raise NotImplementedError('"get_project_object" method is not implemented!')

    @abstractmethod
    def does_object_exists(self, path: str, object_name: str) -> bool:
        raise NotImplementedError('"does_object_exists" method is not implemented!')

    @abstractmethod
    def get_folder_objects(self, path: str, folder: str) -> list[str]:
        raise NotImplementedError('"get_folder_objects" method is not implemented!')

    @abstractmethod
    def remove_folder_objects(self, path: str, folder: str) -> bool:
        raise NotImplementedError('"remove_folder_objects" method is not implemented!')
