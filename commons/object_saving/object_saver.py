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
import logging
from commons.object_saving.minio_client import MinioClient
from commons.object_saving.filesystem_saver import FilesystemSaver


logger = logging.getLogger("analyzerApp.objectSaver")


class ObjectSaver:

    def __init__(self, app_config):
        self.app_config = app_config
        self.saving_strategy = {
            "minio": self.create_minio,
            "filesystem": self.create_fs
        }
        self.binarystore_type = "filesystem"
        if "binaryStoreType" in self.app_config and\
                self.app_config["binaryStoreType"] in self.saving_strategy:
            self.binarystore_type = self.app_config["binaryStoreType"]

    def create_minio(self):
        return MinioClient(self.app_config)

    def create_fs(self):
        return FilesystemSaver(self.app_config)

    def get_bucket_name(self, project_id):
        return self.app_config["minioBucketPrefix"] + str(project_id)

    def remove_project_objects(self, project_id, object_names):
        self.saving_strategy[self.binarystore_type]().remove_project_objects(
            self.get_bucket_name(project_id), object_names)

    def put_project_object(self, data, project_id, object_name, using_json=False):
        self.saving_strategy[self.binarystore_type]().put_project_object(
            data, self.get_bucket_name(project_id),
            object_name, using_json=using_json)

    def get_project_object(self, project_id, object_name, using_json=False):
        return self.saving_strategy[self.binarystore_type]().get_project_object(
            self.get_bucket_name(project_id), object_name, using_json=using_json)

    def does_object_exists(self, project_id, object_name):
        return self.saving_strategy[self.binarystore_type]().does_object_exists(
            self.get_bucket_name(project_id), object_name)

    def get_folder_objects(self, project_id, folder):
        return self.saving_strategy[self.binarystore_type]().get_folder_objects(
            self.get_bucket_name(project_id), folder)

    def remove_folder_objects(self, project_id, folder):
        return self.saving_strategy[self.binarystore_type]().remove_folder_objects(
            self.get_bucket_name(project_id), folder)
