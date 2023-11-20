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

import os
from typing import Any

from app.commons import logging
from app.machine_learning.models.boosting_decision_maker import BoostingDecisionMaker

logger = logging.getLogger("analyzerApp.custom_boosting_decision_maker")


class CustomBoostingDecisionMaker(BoostingDecisionMaker):
    project_id: int | str

    def __init__(self, folder: str, app_config: dict[str, Any], project_id: int | str):
        super().__init__(folder=folder, tags='custom boosting model', app_config=app_config)
        self.project_id = project_id

    def load_model(self):
        self.n_estimators, self.max_depth, self.xg_boost = self.object_saver.get_project_object(
            os.path.join(self.folder, "boost_model"), self.project_id, using_json=False)
        assert self.xg_boost is not None
        self.full_config, self.feature_ids, self.monotonous_features = self.object_saver.get_project_object(
            os.path.join(self.folder, "data_features_config"), self.project_id, using_json=False)
        assert len(self.full_config) > 0
        if self.object_saver.does_object_exists(os.path.join(self.folder, "features_dict_with_saved_objects"),
                                                self.project_id):
            features_dict_with_saved_objects = self.object_saver.get_project_object(
                os.path.join(self.folder, "features_dict_with_saved_objects"), self.project_id, using_json=False)
            self.features_dict_with_saved_objects = self.transform_feature_encoders_to_objects(
                features_dict_with_saved_objects)
        else:
            self.features_dict_with_saved_objects = {}

    def save_model(self):
        self.object_saver.put_project_object([self.n_estimators, self.max_depth, self.xg_boost],
                                             os.path.join(self.folder, "boost_model"), self.project_id,
                                             using_json=False)
        self.object_saver.put_project_object([self.full_config, self.feature_ids, self.monotonous_features],
                                             os.path.join(self.folder, "data_features_config"), self.project_id,
                                             using_json=False)
        self.object_saver.put_project_object(self.transform_feature_encoders_to_dict(),
                                             os.path.join(self.folder, "features_dict_with_saved_objects"),
                                             self.project_id, using_json=False)
