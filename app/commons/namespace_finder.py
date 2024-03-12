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

from gensim.models.phrases import Phrases

from app.commons import logging
from app.commons.object_saving.object_saver import ObjectSaver

logger = logging.getLogger("analyzerApp.namespace_finder")


class NamespaceFinder:

    def __init__(self, app_config):
        self.object_saver = ObjectSaver(app_config)

    def remove_namespaces(self, project_id):
        self.object_saver.remove_project_objects(["project_log_unique_words", "chosen_namespaces"], project_id)

    def get_chosen_namespaces(self, project_id) -> dict:
        if self.object_saver.does_object_exists("chosen_namespaces", project_id):
            return self.object_saver.get_project_object("chosen_namespaces", project_id, using_json=True)
        else:
            return {}

    def update_namespaces(self, project_id, log_words) -> None:
        if self.object_saver.does_object_exists("project_log_unique_words", project_id):
            all_words = self.object_saver.get_project_object("project_log_unique_words", project_id, using_json=True)
        else:
            all_words = {}
        for word in log_words:
            all_words[word] = 1
        self.object_saver.put_project_object(all_words, "project_log_unique_words", project_id, using_json=True)
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
        logger.debug("Chosen namespaces %s", chosen_namespaces)
        self.object_saver.put_project_object(chosen_namespaces, "chosen_namespaces", project_id, using_json=True)
