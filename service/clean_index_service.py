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
from commons.esclient import EsClient
from service import suggest_service

logger = logging.getLogger("analyzerApp.cleanIndexService")


class CleanIndexService:

    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.es_client = EsClient(app_config=app_config, search_cfg=search_cfg)
        self.suggest_service = suggest_service.SuggestService(app_config=app_config, search_cfg=search_cfg)

    @utils.ignore_warnings
    def delete_logs(self, clean_index):
        logger.info("Started cleaning index")
        t_start = time()
        deleted_logs_cnt = self.es_client.delete_logs(clean_index)
        self.suggest_service.clean_suggest_info_logs(clean_index)
        logger.info("Finished cleaning index %.2f s", time() - t_start)
        return deleted_logs_cnt
