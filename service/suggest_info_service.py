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
import json
import utils.utils as utils
from time import time
from commons.esclient import EsClient
from amqp.amqp import AmqpClient
from datetime import datetime

logger = logging.getLogger("analyzerApp.suggestInfoService")


class SuggestInfoService:

    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.es_client = EsClient(app_config=app_config, search_cfg=search_cfg)
        self.rp_suggest_index_template = "suggestions_info"

    def build_index_name(self, project_id):
        return str(project_id) + "_suggest"

    @utils.ignore_warnings
    def index_suggest_info(self, suggest_info_list):
        logger.info("Started saving suggest_info_list")
        t_start = time()
        bodies = []
        for obj in suggest_info_list:
            obj_info = json.loads(obj.json())
            obj_info["savedDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            project_index_name = self.build_index_name(obj_info["project"])
            self.es_client.create_index_for_stats_info(
                self.rp_suggest_index_template,
                override_index_name=project_index_name)
            bodies.append({
                "_index": project_index_name,
                "_source": obj_info
            })
        self.es_client._bulk_index(bodies)
        logger.info("Finished saving %.2f s", time() - t_start)
