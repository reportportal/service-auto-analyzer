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

from boosting_decision_making.defect_type_model import DefectTypeModel
from commons import minio_client


class CustomDefectTypeModel(DefectTypeModel):

    def __init__(self, app_config, project_id, folder=""):
        self.project_id = project_id
        self.minio_client = minio_client.MinioClient(app_config)
        super(CustomDefectTypeModel, self).__init__(folder=folder)
        self.is_global = False

    def load_model(self, folder):
        self.count_vectorizer_models = self.minio_client.get_project_object(
            self.project_id, folder + "count_vectorizer_models", using_json=False)
        self.models = self.minio_client.get_project_object(
            self.project_id, folder + "models", using_json=False)

    def save_model(self, folder):
        self.minio_client.put_project_object(
            self.count_vectorizer_models,
            self.project_id, folder + "count_vectorizer_models", using_json=False)
        self.minio_client.put_project_object(
            self.models,
            self.project_id, folder + "models", using_json=False)
