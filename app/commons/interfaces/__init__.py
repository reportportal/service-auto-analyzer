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
                self.object_saver = ObjectSaver({CONFIG_KEY: 'filesystem', 'filesystemDefaultPath': folder})

    def get_model_info(self):
        folder_name = os.path.basename(self.folder.strip("/").strip("\\")).strip()
        tags = self.tags
        if folder_name:
            tags = [folder_name] + self.tags
        return tags

    @abstractmethod
    def load_model(self):
        raise NotImplementedError('"load_model" method is not implemented!')

    @abstractmethod
    def save_model(self):
        raise NotImplementedError('"save_model" method is not implemented!')
