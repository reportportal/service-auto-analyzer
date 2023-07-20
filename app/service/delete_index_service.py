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

import logging
from app.utils import utils
from time import time
from app.commons import namespace_finder
from app.commons.esclient import EsClient
from app.commons import trigger_manager

logger = logging.getLogger("analyzerApp.deleteIndexService")


class DeleteIndexService:

    def __init__(self, model_chooser, app_config=None, search_cfg=None):
        self.app_config = app_config or {}
        self.search_cfg = search_cfg or {}
        self.namespace_finder = namespace_finder.NamespaceFinder(self.app_config)
        self.trigger_manager = trigger_manager.TriggerManager(
            model_chooser, app_config=self.app_config, search_cfg=self.search_cfg)
        self.es_client = EsClient(app_config=self.app_config, search_cfg=self.search_cfg)
        self.model_chooser = model_chooser

    @utils.ignore_warnings
    def delete_index(self, index_name):
        logger.info("Started deleting index")
        t_start = time()
        is_index_deleted = self.es_client.delete_index(utils.unite_project_name(
            str(index_name), self.app_config["esProjectIndexPrefix"]))
        self.namespace_finder.remove_namespaces(index_name)
        self.trigger_manager.delete_triggers(index_name)
        self.model_chooser.delete_all_custom_models(index_name)
        logger.info("Finished deleting index %.2f s", time() - t_start)
        return int(is_index_deleted)
