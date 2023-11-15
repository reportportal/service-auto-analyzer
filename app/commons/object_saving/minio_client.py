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

from minio import Minio

from app.commons import logging

logger = logging.getLogger("analyzerApp.minioClient")


class MinioClient:

    def __init__(self, app_config: dict) -> None:
        self.app_config = app_config
        self.minioClient = None
        minio_host = app_config['minioHost']
        self.minioClient = Minio(
            minio_host,
            access_key=app_config['minioAccessKey'],
            secret_key=app_config['minioSecretKey'],
            secure=app_config['minioUseTls'],
            region=app_config['minioRegion']
        )
        logger.info(f'Minio initialized {minio_host}')

    def remove_project_objects(self, project_id, object_names) -> None:
        if self.minioClient is None:
            return
        bucket_name = project_id
        if not self.minioClient.bucket_exists(bucket_name):
            return
        for object_name in object_names:
            self.minioClient.remove_object(bucket_name=bucket_name, object_name=object_name)

    def put_project_object(self, data, project_id, object_name, using_json=False) -> None:
        if self.minioClient is None:
            return

        bucket_name = project_id
        if not self.minioClient.bucket_exists(bucket_name):
            logger.debug("Creating minio bucket %s" % bucket_name)
            self.minioClient.make_bucket(bucket_name=bucket_name, location=self.app_config["minioRegion"])
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

    def get_project_object(self, project_id, object_name, using_json=False) -> object | None:
        if self.minioClient is None or not self.minioClient.bucket_exists(project_id):
            return
        obj = self.minioClient.get_object(bucket_name=project_id, object_name=object_name)
        return json.loads(obj.data) if using_json else pickle.loads(obj.data)

    def does_object_exists(self, project_id, object_name) -> bool:
        if self.minioClient is None:
            return False
        if not self.minioClient.bucket_exists(project_id):
            return False
        self.minioClient.get_object(
            bucket_name=project_id, object_name=object_name)
        return True

    def get_folder_objects(self, project_id, folder) -> list[str]:
        if self.minioClient is None:
            return []
        object_names = []
        if not self.minioClient.bucket_exists(project_id):
            return []
        for obj in self.minioClient.list_objects(project_id, prefix=folder):
            object_names.append(obj.object_name)
        return object_names

    def remove_folder_objects(self, project_id, folder) -> bool:
        if self.minioClient is None or not self.minioClient.bucket_exists(project_id):
            return False
        for obj in self.minioClient.list_objects(project_id, prefix=folder):
            self.minioClient.remove_object(bucket_name=project_id, object_name=obj.object_name)
        return True
