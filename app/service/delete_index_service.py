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

from time import time
from typing import Any

from app.commons import logging, namespace_finder, trigger_manager
from app.commons.esclient import EsClient
from app.commons.launch_objects import SearchConfig
from app.utils import utils, text_processing
from app.commons.model_chooser import ModelChooser

logger = logging.getLogger("analyzerApp.deleteIndexService")


class DeleteIndexService:
    app_config: dict[str, Any]
    search_cfg: SearchConfig
    namespace_finder: namespace_finder.NamespaceFinder
    trigger_manager: trigger_manager.TriggerManager
    es_client: EsClient
    model_chooser: ModelChooser

    def __init__(self, model_chooser: ModelChooser, app_config: dict[str, Any], search_cfg: SearchConfig):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.namespace_finder = namespace_finder.NamespaceFinder(self.app_config)
        self.trigger_manager = trigger_manager.TriggerManager(
            model_chooser, self.search_cfg, app_config=self.app_config)
        self.es_client = EsClient(app_config=self.app_config)
        self.model_chooser = model_chooser

    @utils.ignore_warnings
    def delete_index(self, index_name):
        logger.info("Started deleting index")
        t_start = time()
        is_index_deleted = self.es_client.delete_index(text_processing.unite_project_name(
            str(index_name), self.app_config["esProjectIndexPrefix"]))
        self.namespace_finder.remove_namespaces(index_name)
        self.trigger_manager.delete_triggers(index_name)
        self.model_chooser.delete_all_custom_models(index_name)
        logger.info("Finished deleting index %.2f s", time() - t_start)
        return int(is_index_deleted)
