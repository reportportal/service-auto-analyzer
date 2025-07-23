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

import io
import json
import os.path
import pickle
from typing import Any

from minio import Minio
from minio.error import MinioException, S3Error

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig
from app.commons.object_saving.storage import Storage, unify_path_separator

LOGGER = logging.getLogger("analyzerApp.minioClient")
PATH_SEPARATOR = "/"  # Despite the possibility of deploying on Windows, all our configurations use Unix-like paths


class MinioClient(Storage):
    region: str
    minio_client: Minio

    def __init__(self, app_config: ApplicationConfig) -> None:
        super().__init__(app_config)
        minio_host = app_config.minioHost
        self.region = app_config.minioRegion
        self.minio_client = Minio(
            minio_host,
            access_key=app_config.minioAccessKey,
            secret_key=app_config.minioSecretKey,
            secure=app_config.minioUseTls,
            region=self.region,
        )
        LOGGER.debug(f"Minio initialized {minio_host}")

    def get_bucket(self, bucket_id: str | None) -> str:
        path = self._get_project_name(bucket_id)
        if not path:
            return path

        basename = os.path.basename(path)
        if basename == path:
            return path
        return os.path.normpath(path).split(PATH_SEPARATOR)[0]

    def get_path(self, object_name: str, bucket_name: str, bucket_id: str | None) -> str:
        path = self._get_project_name(bucket_id)
        if not path or path == bucket_name:
            return object_name

        path_octets = os.path.normpath(path).split(PATH_SEPARATOR)[1:]
        return unify_path_separator(str(os.path.join(path_octets[0], *path_octets[1:], object_name)))

    def remove_project_objects(self, bucket: str, object_names: list[str]) -> None:
        bucket_name = self.get_bucket(bucket)
        if not self.minio_client.bucket_exists(bucket_name):
            return
        for object_name in object_names:
            path = self.get_path(object_name, bucket_name, bucket)
            self.minio_client.remove_object(bucket_name=bucket_name, object_name=path)

    def put_project_object(self, data: Any, bucket: str, object_name: str, using_json=False) -> None:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(object_name, bucket_name, bucket)
        if bucket_name and not self.minio_client.bucket_exists(bucket_name):
            LOGGER.debug("Creating minio bucket %s" % bucket_name)
            self.minio_client.make_bucket(bucket_name=bucket_name, location=self.region)
            LOGGER.debug("Created minio bucket %s" % bucket_name)
        if using_json:
            data_to_save = json.dumps(data).encode("utf-8")
            content_type = "application/json"
        else:
            data_to_save = pickle.dumps(data)
            content_type = "application/octet-stream"
        data_stream = io.BytesIO(data_to_save)
        data_stream.seek(0)
        result = self.minio_client.put_object(
            bucket_name=bucket_name,
            object_name=path,
            data=data_stream,
            length=len(data_to_save),
            content_type=content_type,
        )
        etag = result.etag
        LOGGER.debug(f'Saved into bucket "{bucket_name}" with path "{path}", etag "{etag}": {data}')

    def get_project_object(self, bucket: str, object_name: str, using_json=False) -> object | None:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(object_name, bucket_name, bucket)
        try:
            obj = self.minio_client.get_object(bucket_name=bucket_name, object_name=path)
        except MinioException as exc:
            raise ValueError(f'Unable to get file in bucket "{bucket_name}" with path "{path}"', exc)
        return json.loads(obj.data) if using_json else pickle.loads(obj.data)

    def does_object_exists(self, bucket: str, object_name: str) -> bool:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(object_name, bucket_name, bucket)
        if bucket_name and not self.minio_client.bucket_exists(bucket_name):
            return False
        try:
            self.minio_client.stat_object(bucket_name=bucket_name, object_name=path)
        except S3Error as e:
            if e.response.status == 404:
                return False
            raise e
        return True

    def get_folder_objects(self, bucket: str, folder: str) -> list[str]:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(folder, bucket_name, bucket)
        if bucket_name and not self.minio_client.bucket_exists(bucket_name):
            return []
        object_names = set()
        object_list = self.minio_client.list_objects(
            bucket_name, prefix=path.endswith(PATH_SEPARATOR) and path or path + PATH_SEPARATOR
        )
        for obj in object_list:
            object_name = obj.object_name.strip(PATH_SEPARATOR)
            if folder != path:
                # Bucket prefix includes path to the project
                prefix = path[0 : -(len(folder))]
                object_name = object_name[len(prefix) :]
            object_names.add(object_name)
        return sorted(object_names)

    def remove_folder_objects(self, bucket: str, folder: str) -> bool:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(folder, bucket_name, bucket)
        if bucket_name and not self.minio_client.bucket_exists(bucket_name):
            return False
        result = False
        for obj in self.minio_client.list_objects(
            bucket_name, prefix=path.endswith(PATH_SEPARATOR) and path or path + PATH_SEPARATOR
        ):
            self.minio_client.remove_object(bucket_name=bucket_name, object_name=obj.object_name)
            result = True
        return result
