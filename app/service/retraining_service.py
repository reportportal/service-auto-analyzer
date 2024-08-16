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

from app.amqp.amqp import AmqpClient
from app.commons.model.ml import TrainInfo
from app.commons import logging, trigger_manager
from app.commons.model.launch_objects import SearchConfig, ApplicationConfig
from app.commons.model_chooser import ModelChooser
from app.utils import utils

logger = logging.getLogger("analyzerApp.retrainingService")


class RetrainingService:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    trigger_manager: trigger_manager.TriggerManager

    def __init__(self, model_chooser: ModelChooser, app_config: ApplicationConfig, search_cfg: SearchConfig):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.trigger_manager = trigger_manager.TriggerManager(model_chooser, app_config=self.app_config,
                                                              search_cfg=self.search_cfg)

    @utils.ignore_warnings
    def train_models(self, train_info: TrainInfo) -> None:
        assert self.trigger_manager.does_trigger_exist(train_info.model_type)
        logger.info('Started training')
        t_start = time()
        _retraining_triggering, _retraining = self.trigger_manager.get_trigger_info(train_info.model_type)
        if _retraining_triggering.should_model_training_be_triggered(train_info):
            logger.debug(f'Should be trained: {train_info.json()}')
            gathered_data, training_log_info = _retraining.train(train_info)
            _retraining_triggering.clean_triggering_info(train_info.project, gathered_data)
            logger.debug(training_log_info)
            if self.app_config.amqpUrl:
                AmqpClient(self.app_config.amqpUrl).send_to_inner_queue(
                    self.app_config.exchangeName, 'stats_info', json.dumps(training_log_info))
        logger.info("Finished training %.2f s", time() - t_start)
