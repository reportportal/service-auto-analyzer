"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

from minio import Minio
import json
import io
import logging
import pickle


logger = logging.getLogger("analyzerApp.minioClient")


class MinioClient:

    def __init__(self, app_config):
        self.minioClient = None
        try:
            self.minioClient = Minio(
                app_config["minioHost"],
                access_key=app_config["minioAccessKey"],
                secret_key=app_config["minioSecretKey"],
                secure=False,
            )
            logger.info("Minio intialized %s" % app_config["minioHost"])
        except Exception as err:
            logger.error(err)

    def get_bucket_name(self, project_id):
        return "prj-%s" % project_id

    def remove_project_objects(self, project_id, object_names):
        if self.minioClient is None:
            return
        try:
            bucket_name = self.get_bucket_name(project_id)
            if not self.minioClient.bucket_exists(bucket_name):
                return
            for object_name in object_names:
                self.minioClient.remove_object(
                    bucket_name=self.get_bucket_name(project_id), object_name=object_name)
        except Exception as err:
            logger.error(err)

    def put_project_object(self, data, project_id, object_name, using_json=False):
        if self.minioClient is None:
            return
        try:
            bucket_name = self.get_bucket_name(project_id)
            if not self.minioClient.bucket_exists(bucket_name):
                logger.debug("Creating minio bucket %s" % bucket_name)
                self.minioClient.make_bucket(bucket_name)
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
            logger.debug(
                "Saved into bucket '%s' with name '%s': %s", bucket_name, object_name, data)
        except Exception as err:
            logger.error(err)

    def get_project_object(self, project_id, object_name, using_json=False):
        if self.minioClient is None:
            return {}
        try:
            obj = self.minioClient.get_object(
                bucket_name=self.get_bucket_name(project_id), object_name=object_name)
            return json.loads(obj.data) if using_json else pickle.loads(obj.data)
        except Exception:
            return {}

    def does_object_exists(self, project_id, object_name):
        try:
            self.minioClient.get_object(
                bucket_name=self.get_bucket_name(project_id), object_name=object_name)
            return True
        except Exception:
            return False

    def get_folder_objects(self, project_id, folder):
        object_names = []
        for obj in self.minioClient.list_objects(
                self.get_bucket_name(project_id), prefix=folder):
            object_names.append(obj.object_name)
        return object_names

    def remove_folder_objects(self, project_id, folder):
        for obj in self.minioClient.list_objects(
                self.get_bucket_name(project_id), prefix=folder):
            self.minioClient.remove_object(
                bucket_name=self.get_bucket_name(project_id), object_name=obj.object_name)
