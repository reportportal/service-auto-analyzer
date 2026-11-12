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

"""File and path helpers without heavy dependencies, safe to import from the coordinator process."""

import json
import os
from typing import Any


def read_file(folder: str, filename: str) -> str:
    """Read file content as string (UTF-8).

    :param folder: Folder of the file
    :param filename: Name of the file
    :return: File content
    """
    with open(os.path.join(folder, filename), "r") as file:
        return file.read()


def read_json_file(folder: str, filename: str, to_json: bool = False) -> Any:
    """Read a file, optionally parsing it as JSON.

    :param folder: Folder of the file
    :param filename: Name of the file
    :param to_json: Whether to parse the content as JSON
    :return: File content, or the parsed JSON object
    """
    content = read_file(folder, filename)
    return content if not to_json else json.loads(content)


def read_resource_file(filename: str, to_json: bool = False) -> Any:
    """Read a file from the "res" folder, optionally parsing it as JSON.

    :param filename: Name of the file
    :param to_json: Whether to parse the content as JSON
    :return: File content, or the parsed JSON object
    """
    return read_json_file("res", filename, to_json)


def strip_path(path: str) -> str:
    """Strip whitespaces and trailing slashes from a path.

    :param path: Path to strip
    :return: Stripped path
    """
    return path.strip().rstrip("/").rstrip("\\")
