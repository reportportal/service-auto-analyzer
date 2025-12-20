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

from typing import Any

from minio import Minio
from minio.error import MinioException, S3Error

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig
from app.commons.object_saving.blob_storage import BlobStorage

LOGGER = logging.getLogger("analyzerApp.minioClient")


class MinioClient(BlobStorage):
    region: str
    minio_client: Minio

    def __init__(self, app_config: ApplicationConfig) -> None:
        super().__init__(app_config)
        # noinspection HttpUrlsUsage
        endpoint = app_config.datastoreEndpoint or "http://minio:9000"  # NOSONAR
        minio_use_tls = endpoint.startswith("https://")
        # noinspection HttpUrlsUsage
        minio_host = endpoint.rstrip().rstrip("/").removeprefix("https://").removeprefix("http://")  # NOSONAR
        self.region = app_config.datastoreRegion
        self.minio_client = Minio(
            minio_host,
            access_key=app_config.datastoreAccessKey or "minio",
            secret_key=app_config.datastoreSecretKey or "minio123",
            secure=minio_use_tls,
            region=self.region,
        )
        LOGGER.debug(f"Minio initialized {minio_host}")

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
            LOGGER.debug(f"Creating minio bucket {bucket_name}")
            self.minio_client.make_bucket(bucket_name=bucket_name, location=self.region)
            LOGGER.debug(f"Created minio bucket {bucket_name}")

        data_to_save, content_type = self.serialize_data(data, using_json)
        data_stream = self.create_data_stream(data_to_save)

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
        return self.deserialize_data(obj.data, using_json)

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
        prefix = self.get_prefix(path)
        object_list = self.minio_client.list_objects(bucket_name, prefix=prefix)
        for obj in object_list:
            object_name = self.extract_object_name(obj.object_name, folder, path)
            object_names.add(object_name)
        return sorted(object_names)

    def remove_folder_objects(self, bucket: str, folder: str) -> bool:
        bucket_name = self.get_bucket(bucket)
        path = self.get_path(folder, bucket_name, bucket)
        if bucket_name and not self.minio_client.bucket_exists(bucket_name):
            return False
        result = False
        prefix = self.get_prefix(path)
        for obj in self.minio_client.list_objects(bucket_name, prefix=prefix):
            self.minio_client.remove_object(bucket_name=bucket_name, object_name=obj.object_name)
            result = True
        return result
