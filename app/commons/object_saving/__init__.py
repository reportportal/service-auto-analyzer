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

"""Common package for different Storage services (Minio, Filesystem, etc.)."""

from typing import Any

from app.commons.object_saving.object_saver import ObjectSaver, CONFIG_KEY


def create(app_config: dict[str, Any], project_id: str | int | None = None, path: str | None = None) -> ObjectSaver:
    return ObjectSaver(app_config=app_config, project_id=project_id, path=path)


def create_filesystem(base_path: str, project_id: str | int | None = None, path: str | None = None) -> ObjectSaver:
    return ObjectSaver(
        app_config={CONFIG_KEY: 'filesystem', 'filesystemDefaultPath': base_path, 'minioBucketPrefix': ''},
        project_id=project_id,
        path=path
    )
