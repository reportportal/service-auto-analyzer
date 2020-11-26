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
from commons import minio_client
from commons.triggering_training import abstract_triggering_training

logger = logging.getLogger("analyzerApp.retraining_defect_type_triggering")


class RetrainingDefectTypeTriggering(abstract_triggering_training.AbstractTrainingTrigger):

    def __init__(self, app_config, start_number=100, accumulated_difference=100):
        self.minio_client = minio_client.MinioClient(app_config)
        self.start_number = start_number
        self.accumulated_difference = accumulated_difference

    def remove_triggering_info(self, train_info):
        self.minio_client.remove_project_objects(
            train_info["project_id"], ["defect_type_trigger_info"])

    def get_triggering_info(self, train_info):
        return self.minio_client.get_project_object(
            train_info["project_id"], "defect_type_trigger_info", using_json=True)

    def save_triggering_info(self, trigger_info, train_info):
        self.minio_client.put_project_object(
            trigger_info, train_info["project_id"],
            "defect_type_trigger_info", using_json=True)

    def clean_defect_type_triggering_info(self, train_info):
        trigger_info = self.get_triggering_info(train_info)
        trigger_info["num_logs_with_defect_types_since_training"] = 0
        self.save_triggering_info(trigger_info, train_info)

    def should_model_training_be_triggered(self, train_info):
        trigger_info = self.get_triggering_info(train_info)
        if "num_logs_with_defect_types" not in trigger_info:
            trigger_info["num_logs_with_defect_types"] = 0
        trigger_info["num_logs_with_defect_types"] += train_info["num_logs_with_defect_types"]
        if "num_logs_with_defect_types_since_training" not in trigger_info:
            trigger_info["num_logs_with_defect_types_since_training"] = 0
        trigger_info["num_logs_with_defect_types_since_training"] += train_info["num_logs_with_defect_types"]
        self.save_triggering_info(trigger_info, train_info)
        return trigger_info["num_logs_with_defect_types"] >= self.start_number\
            and trigger_info["num_logs_with_defect_types_since_training"] >= self.accumulated_difference
