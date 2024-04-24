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

import json
from time import time
from typing import Any

from app.amqp.amqp import AmqpClient
from app.commons import logging, trigger_manager
from app.commons.launch_objects import SearchConfig
from app.commons.model_chooser import ModelChooser
from app.utils import utils

logger = logging.getLogger("analyzerApp.retrainingService")


class RetrainingService:
    search_cfg: SearchConfig
    app_config: dict[str, Any]
    trigger_manager: trigger_manager.TriggerManager

    def __init__(self, model_chooser: ModelChooser, search_cfg: SearchConfig, app_config: dict[str, Any]):
        self.search_cfg = search_cfg
        self.app_config = app_config
        self.trigger_manager = trigger_manager.TriggerManager(
            model_chooser, self.search_cfg, app_config=self.app_config)

    @utils.ignore_warnings
    def train_models(self, train_info):
        assert self.trigger_manager.does_trigger_exist(train_info["model_type"])
        logger.info("Started training")
        t_start = time()
        _retraining_triggering, _retraining = self.trigger_manager.get_trigger_info(train_info["model_type"])
        if _retraining_triggering.should_model_training_be_triggered(train_info):
            logger.debug("Should be trained ", train_info)
            gathered_data, training_log_info = _retraining.train(train_info)
            _retraining_triggering.clean_triggering_info(train_info, gathered_data)
            logger.debug(training_log_info)
            if "amqpUrl" in self.app_config and self.app_config["amqpUrl"].strip():
                AmqpClient(self.app_config["amqpUrl"]).send_to_inner_queue(
                    self.app_config["exchangeName"], "stats_info", json.dumps(training_log_info))
        logger.info("Finished training %.2f s", time() - t_start)
