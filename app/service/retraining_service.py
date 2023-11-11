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
import json
from app.utils import utils
from time import time
from app.commons.esclient import EsClient
from app.commons import trigger_manager
from app.amqp.amqp import AmqpClient

logger = logging.getLogger("analyzerApp.retrainingService")


class RetrainingService:

    def __init__(self, model_chooser, app_config=None, search_cfg=None):
        self.app_config = app_config or {}
        self.search_cfg = search_cfg or {}
        self.trigger_manager = trigger_manager.TriggerManager(
            model_chooser, app_config=self.app_config, search_cfg=self.search_cfg)
        self.es_client = EsClient(app_config=self.app_config, search_cfg=self.search_cfg)

    @utils.ignore_warnings
    def train_models(self, train_info):
        logger.info("Started training")
        t_start = time()
        assert self.trigger_manager.does_trigger_exist(train_info["model_type"])

        _retraining_triggering, _retraining = self.trigger_manager.get_trigger_info(train_info["model_type"])
        is_model_trained = 0
        if _retraining_triggering.should_model_training_be_triggered(train_info):
            logger.debug("Should be trained ", train_info)
            try:
                gathered_data, training_log_info = _retraining.train(train_info)
                _retraining_triggering.clean_triggering_info(train_info, gathered_data)
                logger.debug(training_log_info)
                if "amqpUrl" in self.app_config and self.app_config["amqpUrl"].strip():
                    AmqpClient(self.app_config["amqpUrl"]).send_to_inner_queue(
                        self.app_config["exchangeName"], "stats_info", json.dumps(training_log_info))
                is_model_trained = 1
            except Exception as exc:
                logger.error("Training finished with errors")
                logger.exception(exc)
                is_model_trained = 0
        logger.info("Finished training %.2f s", time() - t_start)
        return is_model_trained
