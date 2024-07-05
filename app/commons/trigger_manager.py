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
from typing import Any

from app.commons import logging
from app.commons.model.launch_objects import SearchConfig, ApplicationConfig
from app.commons.model.ml import ModelType
from app.commons.model_chooser import ModelChooser
from app.commons.triggering_training.retraining_triggering import RetrainingTriggering
from app.machine_learning.training import training_defect_type_model, training_analysis_model

logger = logging.getLogger("analyzerApp.triggerManager")


class TriggerManager:
    model_training_triggering: dict[ModelType, tuple[RetrainingTriggering, Any]]

    def __init__(self, model_chooser: ModelChooser, app_config: ApplicationConfig, search_cfg: SearchConfig):
        self.model_training_triggering = {
            ModelType.defect_type: (RetrainingTriggering(app_config, 'defect_type_trigger_info',
                                                         start_number=100, accumulated_difference=100),
                                    training_defect_type_model.DefectTypeModelTraining(app_config, search_cfg,
                                                                                       model_chooser)),
            ModelType.suggestion: (RetrainingTriggering(app_config, 'suggestion_trigger_info',
                                                        start_number=100, accumulated_difference=50),
                                   training_analysis_model.AnalysisModelTraining(app_config, search_cfg,
                                                                                 model_chooser)),
            ModelType.auto_analysis: (RetrainingTriggering(app_config, 'auto_analysis_trigger_info',
                                                           start_number=300, accumulated_difference=100),
                                      training_analysis_model.AnalysisModelTraining(app_config, search_cfg,
                                                                                    model_chooser))
        }

    def does_trigger_exist(self, model: ModelType):
        return model in self.model_training_triggering

    def get_trigger_info(self, model: ModelType) -> tuple[RetrainingTriggering, Any]:
        return self.model_training_triggering[model]

    def delete_triggers(self, project_id: int):
        for trigger_info in self.model_training_triggering.values():
            trigger_info[0].remove_triggering_info(project_id)
