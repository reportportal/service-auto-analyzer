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
from commons import namespace_finder
from commons.esclient import EsClient
from commons.triggering_training.retraining_triggering import RetrainingTriggering
from boosting_decision_making.training_models import training_defect_type_model
from amqp.amqp import AmqpClient

logger = logging.getLogger("analyzerApp.retrainingService")


class RetrainingService:

    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.namespace_finder = namespace_finder.NamespaceFinder(app_config)
        self.model_training_triggering = {
            "defect_type": (RetrainingTriggering(app_config, "defect_type_trigger_info",
                                                 start_number=100, accumulated_difference=100),
                            training_defect_type_model.DefectTypeModelTraining(
                                app_config, search_cfg)),
            "suggestions": (RetrainingTriggering(app_config, "suggestions_trigger_info",
                                                 start_number=100, accumulated_difference=50),
                            training_defect_type_model.DefectTypeModelTraining(
                                app_config, search_cfg))
        }
        self.es_client = EsClient(app_config=app_config, search_cfg=search_cfg)

    @utils.ignore_warnings
    def train_models(self, train_info):
        logger.info("Started training")
        t_start = time()
        assert train_info["model_type"] in self.model_training_triggering

        _retraining_triggering, _retraining = self.model_training_triggering[train_info["model_type"]]
        if _retraining_triggering.should_model_training_be_triggered(train_info):
            logger.debug("Should be trained ", train_info)
            try:
                gathered_data, training_log_info = _retraining.train(train_info)
                _retraining_triggering.clean_defect_type_triggering_info(
                    train_info, gathered_data)
                if "amqpUrl" in self.app_config and self.app_config["amqpUrl"].strip():
                    AmqpClient(self.app_config["amqpUrl"]).send_to_inner_queue(
                        self.app_config["exchangeName"], "stats_info", json.dumps(training_log_info))
            except Exception as err:
                logger.error("Training finished with errors")
                logger.error(err)
        logger.info("Finished training %.2f s", time() - t_start)
