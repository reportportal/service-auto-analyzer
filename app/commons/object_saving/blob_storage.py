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

"""Base class for blob storage implementations (S3, Minio, etc.)."""
import io
import json
import os.path
import pickle
from abc import ABCMeta
from typing import Any

from app.commons.model.launch_objects import ApplicationConfig
from app.commons.object_saving.storage import Storage, unify_path_separator

PATH_SEPARATOR = "/"  # Despite the possibility of deploying on Windows, all our configurations use Unix-like paths


class BlobStorage(Storage, metaclass=ABCMeta):
    """Base class for blob storage implementations with common bucket and path handling logic."""

    def __init__(self, app_config: ApplicationConfig) -> None:
        super().__init__(app_config)

    def get_bucket(self, bucket_id: str | None) -> str:
        """Extract bucket name from project path.

        Args:
            bucket_id: Project identifier

        Returns:
            Bucket name (first part of the path)
        """
        path = self._get_project_name(bucket_id)
        if not path:
            return path

        basename = os.path.basename(path)
        if basename == path:
            return path
        return os.path.normpath(path).split(PATH_SEPARATOR)[0]

    def get_path(self, object_name: str, bucket_name: str, bucket_id: str | None) -> str:
        """Get object path within bucket.

        Args:
            object_name: Name of the object
            bucket_name: Name of the bucket
            bucket_id: Project identifier

        Returns:
            Full path to the object within the bucket
        """
        path = self._get_project_name(bucket_id)
        if not path or path == bucket_name:
            return object_name

        path_octets = os.path.normpath(path).split(PATH_SEPARATOR)[1:]
        return unify_path_separator(str(os.path.join(path_octets[0], *path_octets[1:], object_name)))

    def serialize_data(self, data: Any, using_json: bool = False) -> tuple[bytes, str]:
        """Serialize data to bytes with appropriate content type.

        Args:
            data: Data to serialize
            using_json: Whether to use JSON serialization (default: pickle)

        Returns:
            Tuple of (serialized data, content type)
        """
        if using_json:
            data_to_save = json.dumps(data).encode("utf-8")
            content_type = "application/json"
        else:
            data_to_save = pickle.dumps(data)
            content_type = "application/octet-stream"
        return data_to_save, content_type

    def deserialize_data(self, data: bytes, using_json: bool = False) -> Any:
        """Deserialize data from bytes.

        Args:
            data: Serialized data
            using_json: Whether to use JSON deserialization (default: pickle)

        Returns:
            Deserialized data
        """
        return json.loads(data) if using_json else pickle.loads(data)

    def create_data_stream(self, data: bytes) -> io.BytesIO:
        """Create a BytesIO stream from data.

        Args:
            data: Data to wrap in stream

        Returns:
            BytesIO stream positioned at the start
        """
        data_stream = io.BytesIO(data)
        data_stream.seek(0)
        return data_stream

    def get_prefix(self, path: str) -> str:
        """Get prefix for folder operations, ensuring it ends with separator.

        Args:
            path: Folder path

        Returns:
            Path with trailing separator
        """
        return path.endswith(PATH_SEPARATOR) and path or path + PATH_SEPARATOR

    def extract_object_name(self, full_path: str, folder: str, path: str) -> str:
        """Extract object name from full path, removing bucket prefix if needed.

        Args:
            full_path: Full object path
            folder: Original folder parameter
            path: Computed path with bucket prefix

        Returns:
            Object name relative to the folder
        """
        object_name = full_path.strip(PATH_SEPARATOR)
        if folder != path:
            # Bucket prefix includes path to the project
            prefix_len = len(path) - len(folder)
            object_name = object_name[prefix_len:]
        return object_name
