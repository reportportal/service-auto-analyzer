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
from commons.triggering_training.retraining_triggering import RetrainingTriggering
from boosting_decision_making.training_models import training_defect_type_model
from boosting_decision_making.training_models import training_analysis_model

logger = logging.getLogger("analyzerApp.triggerManager")


class TriggerManager:

    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.model_training_triggering = {
            "defect_type": (RetrainingTriggering(app_config, "defect_type_trigger_info",
                                                 start_number=100, accumulated_difference=100),
                            training_defect_type_model.DefectTypeModelTraining(
                                app_config, search_cfg)),
            "suggestion": (RetrainingTriggering(app_config, "suggestion_trigger_info",
                                                start_number=100, accumulated_difference=50),
                           training_analysis_model.AnalysisModelTraining(
                               app_config, search_cfg)),
            "auto_analysis": (RetrainingTriggering(app_config, "auto_analysis_trigger_info",
                                                   start_number=300, accumulated_difference=100),
                              training_analysis_model.AnalysisModelTraining(
                                  app_config, search_cfg))
        }

    def is_trigger_exists(self, name):
        return name in self.model_training_triggering

    def get_trigger_info(self, name):
        return self.model_training_triggering[name]

    def delete_triggers(self, project_id):
        for model_type in self.model_training_triggering:
            self.model_training_triggering[model_type][0].remove_triggering_info(
                {"project_id": project_id})
