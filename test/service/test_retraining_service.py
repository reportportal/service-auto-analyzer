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

import unittest
from unittest.mock import MagicMock
from app.service import RetrainingService
from test.mock_service import TestService
from app.utils import utils


class TestRetrainingService(TestService):

    @utils.ignore_warnings
    def test_train_models_triggering(self):
        """Test train models triggering"""
        tests = [
            {
                "train_info": {"model_type": "defect_type",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "defect_type",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 120},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "defect_type",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 94, "gathered_metric_since_training": 94},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "defect_type",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 95, "gathered_metric_since_training": 95},
                "train_result": (123, {}),
                "is_model_trained": 1
            },
            {
                "train_info": {"model_type": "defect_type",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 120, "gathered_metric_since_training": 67},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "defect_type",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 85, "gathered_metric_since_training": 95},
                "train_result": (123, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "suggestion",
                               "project_id": 1,
                               "gathered_metric_total": 3},
                "trigger_info": {},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "suggestion",
                               "project_id": 1,
                               "gathered_metric_total": 3},
                "trigger_info": {"gathered_metric_total": 14},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "suggestion",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 90, "gathered_metric_since_training": 35},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "suggestion",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 120, "gathered_metric_since_training": 30},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "suggestion",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 95, "gathered_metric_since_training": 45},
                "train_result": (100, {}),
                "is_model_trained": 1
            },
            {
                "train_info": {"model_type": "suggestion",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 89, "gathered_metric_since_training": 55},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "auto_analysis",
                               "project_id": 1,
                               "gathered_metric_total": 3},
                "trigger_info": {},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "auto_analysis",
                               "project_id": 1,
                               "gathered_metric_total": 3},
                "trigger_info": {"gathered_metric_total": 14},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "auto_analysis",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 290, "gathered_metric_since_training": 92},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "auto_analysis",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 301, "gathered_metric_since_training": 93},
                "train_result": (0, {}),
                "is_model_trained": 0
            },
            {
                "train_info": {"model_type": "auto_analysis",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 295, "gathered_metric_since_training": 95},
                "train_result": (100, {}),
                "is_model_trained": 1
            },
            {
                "train_info": {"model_type": "auto_analysis",
                               "project_id": 1,
                               "gathered_metric_total": 5},
                "trigger_info": {"gathered_metric_total": 291, "gathered_metric_since_training": 95},
                "train_result": (0, {}),
                "is_model_trained": 0
            }
        ]
        for idx, test in enumerate(tests):
            try:
                _retraining_service = RetrainingService(self.model_chooser,
                                                        app_config=self.app_config,
                                                        search_cfg=self.get_default_search_config())
                model_triggering = _retraining_service.trigger_manager.model_training_triggering
                model_triggering = model_triggering[test["train_info"]["model_type"]]
                model_triggering[0].object_saver.get_project_object = MagicMock(
                    return_value=test["trigger_info"])
                model_triggering[1].train = MagicMock(
                    return_value=test["train_result"])
                response = _retraining_service.train_models(test["train_info"])
                assert test["is_model_trained"] == response
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}').\
                    with_traceback(err.__traceback__)


if __name__ == '__main__':
    unittest.main()
