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
from commons.object_saving.object_saver import ObjectSaver

logger = logging.getLogger("analyzerApp.retraining_triggering")


class RetrainingTriggering():

    def __init__(self, app_config, trigger_saving_name, start_number=100, accumulated_difference=100):
        self.object_saver = ObjectSaver(app_config)
        self.start_number = start_number
        self.accumulated_difference = accumulated_difference
        self.trigger_saving_name = trigger_saving_name
        self.required_fields = ["gathered_metric_since_training", "gathered_metric_total"]

    def remove_triggering_info(self, train_info):
        self.object_saver.remove_project_objects(
            train_info["project_id"], [self.trigger_saving_name])

    def get_triggering_info(self, train_info):
        obj = self.object_saver.get_project_object(
            train_info["project_id"], self.trigger_saving_name, using_json=True)
        for required_field in self.required_fields:
            if required_field not in obj:
                return {}
        return obj

    def save_triggering_info(self, trigger_info, train_info):
        self.object_saver.put_project_object(
            trigger_info, train_info["project_id"],
            self.trigger_saving_name, using_json=True)

    def clean_triggering_info(self, train_info, gathered_metric_total):
        trigger_info = self.get_triggering_info(train_info)
        trigger_info["gathered_metric_since_training"] = 0
        trigger_info["gathered_metric_total"] = gathered_metric_total
        self.save_triggering_info(trigger_info, train_info)

    def should_model_training_be_triggered(self, train_info):
        trigger_info = self.get_triggering_info(train_info)
        if "gathered_metric_total" not in trigger_info:
            trigger_info["gathered_metric_total"] = 0
        trigger_info["gathered_metric_total"] += train_info["gathered_metric_total"]
        if "gathered_metric_since_training" not in trigger_info:
            trigger_info["gathered_metric_since_training"] = 0
        trigger_info["gathered_metric_since_training"] += train_info["gathered_metric_total"]
        self.save_triggering_info(trigger_info, train_info)
        return trigger_info["gathered_metric_total"] >= self.start_number\
            and trigger_info["gathered_metric_since_training"] >= self.accumulated_difference
