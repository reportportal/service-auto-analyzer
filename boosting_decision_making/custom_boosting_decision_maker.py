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

from boosting_decision_making.boosting_decision_maker import BoostingDecisionMaker
from commons.object_saving.object_saver import ObjectSaver
import os
import logging

logger = logging.getLogger("analyzerApp.custom_boosting_decision_maker")


class CustomBoostingDecisionMaker(BoostingDecisionMaker):

    def __init__(self, app_config, project_id, folder=""):
        self.project_id = project_id
        self.object_saver = ObjectSaver(app_config)
        super(CustomBoostingDecisionMaker, self).__init__(folder=folder)
        self.is_global = False

    def load_model(self, folder):
        self.n_estimators, self.max_depth, self.xg_boost = self.object_saver.get_project_object(
            self.project_id, os.path.join(folder, "boost_model"),
            using_json=False)
        assert self.xg_boost is not None
        self.full_config, self.feature_ids, self.monotonous_features = self.object_saver.get_project_object(
            self.project_id, os.path.join(folder, "data_features_config"),
            using_json=False)
        assert len(self.full_config) > 0
        if self.object_saver.does_object_exists(
                self.project_id, os.path.join(folder, "features_dict_with_saved_objects")):
            features_dict_with_saved_objects = self.object_saver.get_project_object(
                self.project_id, os.path.join(folder, "features_dict_with_saved_objects"),
                using_json=False)
            self.features_dict_with_saved_objects = self.transform_feature_encoders_to_objects(
                features_dict_with_saved_objects)
        else:
            self.features_dict_with_saved_objects = {}

    def save_model(self, folder):
        self.object_saver.put_project_object(
            [self.n_estimators, self.max_depth, self.xg_boost],
            self.project_id, os.path.join(folder, "boost_model"),
            using_json=False)
        self.object_saver.put_project_object(
            [self.full_config, self.feature_ids, self.monotonous_features],
            self.project_id, os.path.join(folder, "data_features_config"),
            using_json=False)
        self.object_saver.put_project_object(
            self.transform_feature_encoders_to_dict(),
            self.project_id, os.path.join(folder, "features_dict_with_saved_objects"),
            using_json=False)
