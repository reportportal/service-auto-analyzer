#  Copyright 2024 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from abc import ABCMeta, abstractmethod
from typing import Any, Iterable

from app.commons.object_saving.object_saver import ObjectSaver


class MlModel(metaclass=ABCMeta):
    """Base class for ML models."""
    tags: list[str]
    object_saver: ObjectSaver

    def __init__(self, object_saver: ObjectSaver, tags: str) -> None:
        self.tags = [tag.strip() for tag in tags.split(',')]
        self.object_saver = object_saver

    def _load_models(self, model_files: list[str]) -> list[Any]:
        result = []
        for file in model_files:
            model = self.object_saver.get_project_object(file, using_json=False)
            if model is None:
                raise ValueError(f'Unable to load model "{file}".')
            result.append(model)
        return result

    def _save_models(self, data: dict[str, Any] | Iterable[tuple[str, Any]]) -> None:
        for file_name, object_to_save in dict(data).items():
            self.object_saver.put_project_object(object_to_save, file_name, using_json=False)

    def get_model_info(self) -> list[str]:
        folder_name = self.object_saver.path.strip("/").strip("\\").strip()
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

    @property
    @abstractmethod
    def loaded(self) -> bool:
        raise NotImplementedError('"loaded" property is not implemented!')
