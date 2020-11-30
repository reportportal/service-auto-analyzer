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
import utils.utils as utils
from time import time
from commons import namespace_finder
from commons.esclient import EsClient
from commons.triggering_training.retraining_defect_type_triggering import RetrainingDefectTypeTriggering

logger = logging.getLogger("analyzerApp.deleteIndexService")


class DeleteIndexService:

    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.namespace_finder = namespace_finder.NamespaceFinder(app_config)
        self.model_training_triggering = {
            "defect_type": RetrainingDefectTypeTriggering(self.app_config)
        }
        self.es_client = EsClient(app_config=app_config, search_cfg=search_cfg)

    @utils.ignore_warnings
    def delete_index(self, index_name):
        logger.info("Started deleting index")
        t_start = time()
        is_index_deleted = self.es_client.delete_index(index_name)
        self.namespace_finder.remove_namespaces(index_name)
        for model_type in self.model_training_triggering:
            self.model_training_triggering[model_type].remove_triggering_info(
                {"project_id": index_name})
        logger.info("Finished deleting index %.2f s", time() - t_start)
        return int(is_index_deleted)
