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
import pickle
from typing import Any

from minio import Minio
from minio.error import NoSuchKey

from app.commons import logging
from app.commons.object_saving.storage import Storage

logger = logging.getLogger("analyzerApp.minioClient")


class MinioClient(Storage):
    region: str
    bucket_prefix: str

    def __init__(self, app_config: dict[str, Any]) -> None:
        minio_host = app_config['minioHost']
        self.region = app_config['minioRegion']
        self.bucket_prefix = app_config['minioBucketPrefix']
        self.minioClient = Minio(
            minio_host,
            access_key=app_config['minioAccessKey'],
            secret_key=app_config['minioSecretKey'],
            secure=app_config['minioUseTls'],
            region=self.region
        )
        logger.info(f'Minio initialized {minio_host}')

    def get_bucket(self, bucket: str | None):
        if bucket:
            return self.bucket_prefix + bucket
        else:
            return ''

    def remove_project_objects(self, bucket: str, object_names: list[str]) -> None:
        bucket_name = self.get_bucket(bucket)
        if not self.minioClient.bucket_exists(bucket_name):
            return
        for object_name in object_names:
            self.minioClient.remove_object(bucket_name=bucket_name, object_name=object_name)

    def put_project_object(self, data: Any, bucket: str, object_name: str, using_json=False) -> None:
        bucket_name = self.get_bucket(bucket)
        if bucket_name:
            if not self.minioClient.bucket_exists(bucket_name):
                logger.debug("Creating minio bucket %s" % bucket_name)
                self.minioClient.make_bucket(bucket_name=bucket_name, location=self.region)
                logger.debug("Created minio bucket %s" % bucket_name)
        if using_json:
            data_to_save = json.dumps(data).encode("utf-8")
        else:
            data_to_save = pickle.dumps(data)
        data_stream = io.BytesIO(data_to_save)
        data_stream.seek(0)
        self.minioClient.put_object(
            bucket_name=bucket_name, object_name=object_name,
            data=data_stream, length=len(data_to_save))
        logger.debug("Saved into bucket '%s' with name '%s': %s", bucket_name, object_name, data)

    def get_project_object(self, bucket: str, object_name: str, using_json=False) -> object | None:
        bucket_name = self.get_bucket(bucket)
        try:
            obj = self.minioClient.get_object(bucket_name=bucket_name, object_name=object_name)
        except NoSuchKey as exc:
            raise ValueError(f'Unable to get file: {object_name}', exc)
        return json.loads(obj.data) if using_json else pickle.loads(obj.data)

    def does_object_exists(self, bucket: str, object_name: str) -> bool:
        bucket_name = self.get_bucket(bucket)
        if bucket_name:
            if not self.minioClient.bucket_exists(bucket_name):
                return False
        try:
            self.minioClient.stat_object(bucket_name=bucket_name, object_name=object_name)
        except NoSuchKey:
            return False
        return True

    def get_folder_objects(self, bucket: str, folder: str) -> list[str]:
        bucket_name = self.get_bucket(bucket)
        if bucket_name:
            if not self.minioClient.bucket_exists(bucket_name):
                return []
        object_names = []
        object_list = self.minioClient.list_objects(bucket_name, prefix=folder, recursive=True)
        for obj in object_list:
            object_names.append(obj.object_name)
        return object_names

    def remove_folder_objects(self, bucket: str, folder: str) -> bool:
        bucket_name = self.get_bucket(bucket)
        if bucket_name:
            if not self.minioClient.bucket_exists(bucket_name):
                return False
        for obj in self.minioClient.list_objects(bucket_name, prefix=folder):
            self.minioClient.remove_object(bucket_name=bucket_name, object_name=obj.object_name)
        return True
