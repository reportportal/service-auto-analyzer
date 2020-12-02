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
from boosting_decision_making.training_models import training_defect_type_model

logger = logging.getLogger("analyzerApp.retrainingService")


class RetrainingService:

    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.namespace_finder = namespace_finder.NamespaceFinder(app_config)
        self.model_training_triggering = {
            "defect_type": (RetrainingDefectTypeTriggering(app_config),
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
            print("Should be trained ", train_info)
            try:
                _retraining.train(train_info)
            except Exception as err:
                logger.error("Training finished with errors")
                logger.error(err)
        logger.info("Finished training %.2f s", time() - t_start)
