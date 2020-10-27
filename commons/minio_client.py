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

    def put_project_object(self, data, project_id, object_name):
        if self.minioClient is None:
            return
        try:
            bucket_name = self.get_bucket_name(project_id)
            if not self.minioClient.bucket_exists(bucket_name):
                logger.debug("Creating minio bucket %s" % bucket_name)
                self.minioClient.make_bucket(bucket_name)
                logger.debug("Created minio bucket %s" % bucket_name)
            logger.debug("Saving minio object")
            data = json.dumps(data).encode("utf-8")
            data_stream = io.BytesIO(data)
            data_stream.seek(0)
            self.minioClient.put_object(
                bucket_name=bucket_name, object_name=object_name,
                data=data_stream, length=len(data))
            logger.debug(
                "Saved into bucket '%s' with name '%s'", bucket_name, object_name)
        except Exception as err:
            logger.error(err)

    def get_project_object(self, project_id, object_name, defalt_obj={}):
        if self.minioClient is None:
            return defalt_obj
        try:
            obj = self.minioClient.get_object(
                bucket_name=self.get_bucket_name(project_id), object_name=object_name)
            return json.loads(obj.data)
        except Exception:
            return defalt_obj
