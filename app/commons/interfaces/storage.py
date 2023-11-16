"""Interface classes related to data storing/reading."""
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

from abc import ABCMeta, abstractmethod


class Storage(metaclass=ABCMeta):
    @abstractmethod
    def remove_project_objects(self, project_id, object_names) -> None:
        raise NotImplementedError('"remove_project_objects" method is not implemented!')

    @abstractmethod
    def put_project_object(self, data, project_id, object_name, using_json: bool = False) -> None:
        raise NotImplementedError('"put_project_object" method is not implemented!')

    @abstractmethod
    def get_project_object(self, project_id, object_name, using_json: bool = False) -> object | None:
        raise NotImplementedError('"get_project_object" method is not implemented!')

    @abstractmethod
    def does_object_exists(self, project_id, object_name) -> bool:
        raise NotImplementedError('"does_object_exists" method is not implemented!')

    @abstractmethod
    def get_folder_objects(self, project_id, folder) -> list[str]:
        raise NotImplementedError('"get_folder_objects" method is not implemented!')

    @abstractmethod
    def remove_folder_objects(self, project_id, folder) -> bool:
        raise NotImplementedError('"remove_folder_objects" method is not implemented!')
