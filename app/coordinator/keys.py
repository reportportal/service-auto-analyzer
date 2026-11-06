#  Copyright 2026 EPAM Systems
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

"""Lane keys: which project a message belongs to, so messages of one project are processed in order."""

import json
from typing import Any, Final, Optional

from app.commons import logging

LOGGER = logging.getLogger("analyzerApp.coordinator.keys")

# Routes whose body is a list of launches, which may belong to different projects
LAUNCH_LIST_ROUTES: Final[frozenset[str]] = frozenset({"index", "analyze", "namespace_finder"})
# Routes whose body is the project ID itself
PROJECT_BODY_ROUTES: Final[frozenset[str]] = frozenset({"delete", "suggest_patterns"})
# Routes whose body is an object with the project ID in the "projectId" field
PROJECT_ID_FIELD_ROUTES: Final[frozenset[str]] = frozenset({"search"})
# Routes whose body is an object with the project ID in the "project" field
PROJECT_FIELD_ROUTES: Final[frozenset[str]] = frozenset(
    {
        "defect_update",
        "clean",
        "item_remove",
        "launch_remove",
        "remove_by_launch_start_time",
        "remove_by_log_time",
        "suggest",
        "cluster",
        "train_models",
    }
)
PROJECT_FIELD: Final[str] = "project"
PROJECT_ID_FIELD: Final[str] = "projectId"


def _normalize(project: Any) -> Optional[str]:
    """Normalize a project ID to a lane key, so 1 and "1" share a lane.

    :param project: Project ID from a message body
    :return: The lane key, or None if there's no usable project ID
    """
    if project is None or isinstance(project, (bool, dict, list)):
        return None
    key = str(project).strip()
    return key or None


def _field(body: Any, field_name: str) -> Optional[str]:
    if not isinstance(body, dict):
        return None
    return _normalize(body.get(field_name))


def lane_key(routing_key: str, body: Any) -> Optional[str]:
    """Get the lane key of a message body.

    For a launch list it's the project of the first launch: use `split_by_project` for those routes, which splits
    lists with several projects first.

    :param routing_key: Routing key of the message
    :param body: Parsed JSON body of the message
    :return: The lane key, or None if the message doesn't need ordering (deprecated, test and unknown routes, or a body
        without a project ID)
    """
    key: Optional[str] = None
    if routing_key in LAUNCH_LIST_ROUTES:
        if isinstance(body, list) and body:
            key = _field(body[0], PROJECT_FIELD)
    elif routing_key in PROJECT_BODY_ROUTES:
        key = _normalize(body)
    elif routing_key in PROJECT_ID_FIELD_ROUTES:
        key = _field(body, PROJECT_ID_FIELD)
    elif routing_key in PROJECT_FIELD_ROUTES:
        key = _field(body, PROJECT_FIELD)
    else:
        return None
    if key is None:
        LOGGER.warning(f"No project ID in a '{routing_key}' message, it will be processed without ordering")
    return key


def split_by_project(routing_key: str, raw: bytes) -> list[tuple[Optional[str], bytes]]:
    """Split a message into one part per project, each with its lane key and raw body.

    Only launch lists are split. A message with one project keeps its original body, without re-serializing it.

    :param routing_key: Routing key of the message
    :param raw: Raw body of the message
    :return: Parts as (lane key, raw body) pairs, in the order of the first launch of each project
    """
    body = json.loads(raw, strict=False)
    if routing_key not in LAUNCH_LIST_ROUTES or not isinstance(body, list):
        return [(lane_key(routing_key, body), raw)]
    groups: dict[Optional[str], list[Any]] = {}
    for launch in body:
        groups.setdefault(_field(launch, PROJECT_FIELD), []).append(launch)
    if len(groups) <= 1:
        return [(lane_key(routing_key, body), raw)]
    projects = ", ".join(str(key) for key in groups)
    LOGGER.info(f"Splitting a '{routing_key}' message with {len(body)} launches by projects: {projects}")
    return [(key, json.dumps(launches).encode("utf-8")) for key, launches in groups.items()]
