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
from commons import minio_client
from gensim.models.phrases import Phrases

logger = logging.getLogger("analyzerApp.namespace_finder")


class NamespaceFinder:

    def __init__(self, app_config):
        self.minio_client = minio_client.MinioClient(app_config)

    def remove_namespaces(self, project_id):
        self.minio_client.remove_project_objects(
            project_id, ["project_log_unique_words", "chosen_namespaces"])

    def get_chosen_namespaces(self, project_id):
        return self.minio_client.get_project_object(project_id, "chosen_namespaces")

    def update_namespaces(self, project_id, log_words):
        all_words = self.minio_client.get_project_object(project_id, "project_log_unique_words")
        for word in log_words:
            all_words[word] = 1
        self.minio_client.put_project_object(
            all_words, project_id, "project_log_unique_words")
        phrases = Phrases([w.split(".") for w in all_words], min_count=1, threshold=1)
        potential_project_namespaces = {}
        for word in all_words:
            potential_namespace = phrases[word.split(".")][0]
            if "_" not in potential_namespace:
                continue
            if potential_namespace not in potential_project_namespaces:
                potential_project_namespaces[potential_namespace] = 0
            potential_project_namespaces[potential_namespace] += 1
        chosen_namespaces = {}
        for item, cnt in potential_project_namespaces.items():
            if cnt > 10:
                chosen_namespaces[item.replace("_", ".")] = cnt
        self.minio_client.put_project_object(
            chosen_namespaces, project_id, "chosen_namespaces")
