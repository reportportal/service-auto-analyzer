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

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig
from app.commons.model.ml import TrainInfo
from app.commons.object_saving.object_saver import ObjectSaver

METRIC_SINCE_TRAINING = "gathered_metric_since_training"
GATHERED_METRIC_TOTAL = "gathered_metric_total"
REQUIRED_FIELDS = [METRIC_SINCE_TRAINING, GATHERED_METRIC_TOTAL]

logger = logging.getLogger("analyzerApp.retraining_triggering")


class RetrainingTriggering:
    object_saver: ObjectSaver
    start_number: int
    accumulated_difference: int
    trigger_saving_name: str

    def __init__(
        self,
        app_config: ApplicationConfig,
        trigger_saving_name: str,
        start_number: int = 100,
        accumulated_difference: int = 100,
    ):
        self.object_saver = ObjectSaver(app_config)
        self.start_number = start_number
        self.accumulated_difference = accumulated_difference
        self.trigger_saving_name = trigger_saving_name

    def remove_triggering_info(self, project_id: int) -> None:
        self.object_saver.remove_project_objects([self.trigger_saving_name], project_id)

    def get_triggering_info(self, project_id: int) -> dict[str, int]:
        if not self.object_saver.does_object_exists(self.trigger_saving_name, project_id):
            self.clean_triggering_info(project_id, 0)
        obj = self.object_saver.get_project_object(self.trigger_saving_name, project_id, using_json=True)
        for required_field in REQUIRED_FIELDS:
            if required_field not in obj:
                return {}
        return obj

    def save_triggering_info(self, trigger_info: dict[str, int], project_id: int) -> None:
        self.object_saver.put_project_object(trigger_info, self.trigger_saving_name, project_id, using_json=True)

    def clean_triggering_info(self, project_id: int, gathered_metric_total: int) -> None:
        if self.object_saver.does_object_exists(self.trigger_saving_name, project_id):
            trigger_info = self.get_triggering_info(project_id)
        else:
            trigger_info = {}
        trigger_info[METRIC_SINCE_TRAINING] = 0
        trigger_info[GATHERED_METRIC_TOTAL] = gathered_metric_total
        self.save_triggering_info(trigger_info, project_id)

    def should_model_training_be_triggered(self, train_info: TrainInfo) -> bool:
        trigger_info = self.get_triggering_info(train_info.project)
        gathered_metric_total = trigger_info.get(GATHERED_METRIC_TOTAL, 0)
        trigger_info[GATHERED_METRIC_TOTAL] = gathered_metric_total + train_info.gathered_metric_total
        metric_since_training = trigger_info.get(METRIC_SINCE_TRAINING, 0)
        trigger_info[METRIC_SINCE_TRAINING] = metric_since_training + train_info.gathered_metric_total
        self.save_triggering_info(trigger_info, train_info.project)
        return (
            trigger_info[GATHERED_METRIC_TOTAL] >= self.start_number
            and trigger_info[METRIC_SINCE_TRAINING] >= self.accumulated_difference
        )
