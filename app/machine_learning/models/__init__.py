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

"""Common package for ML models."""

import os
from abc import ABCMeta, abstractmethod
from typing import Any

from app.commons.object_saving.object_saver import ObjectSaver, CONFIG_KEY


class MlModel(metaclass=ABCMeta):
    app_config: dict[str, Any]
    tags: list[str]
    folder: str
    object_saver: ObjectSaver

    def __init__(self, folder: str, tags: str, *, object_saver: ObjectSaver = None,
                 app_config: dict[str, Any] = None) -> None:
        self.folder = folder
        self.tags = [tag.strip() for tag in tags.split(',')]
        self.app_config = app_config
        if object_saver:
            self.object_saver = object_saver
        else:
            if app_config:
                self.object_saver = ObjectSaver(app_config)
            else:
                self.object_saver = ObjectSaver({CONFIG_KEY: 'filesystem', 'filesystemDefaultPath': ''})

    def _load_models(self, model_files: list[str]) -> list[Any]:
        result = []
        for file in model_files:
            model = self.object_saver.get_project_object(os.path.join(self.folder, file), using_json=False)
            if model is None:
                raise ValueError(f'Unable to load model "{file}".')
            result.append(model)
        return result

    def _save_models(self, data: dict[str, Any] | list[tuple[str, Any]]) -> None:
        for file_name, object_to_save in dict(data).items():
            self.object_saver.put_project_object(object_to_save, os.path.join(self.folder, file_name),
                                                 using_json=False)

    def get_model_info(self) -> list[str]:
        folder_name = os.path.basename(self.folder.strip("/").strip("\\")).strip()
        tags = self.tags
        if folder_name:
            tags = [folder_name] + self.tags
        return tags

    @abstractmethod
    def load_model(self) -> None:
        raise NotImplementedError('"load_model" method is not implemented!')

    @abstractmethod
    def save_model(self) -> None:
        raise NotImplementedError('"save_model" method is not implemented!')
