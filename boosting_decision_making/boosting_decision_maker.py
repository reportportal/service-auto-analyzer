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

from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle
import logging
from utils import utils
from boosting_decision_making import feature_encoder

logger = logging.getLogger("analyzerApp.boosting_decision_maker")


class BoostingDecisionMaker:

    def __init__(self, folder="", n_estimators=75, max_depth=5,
                 monotonous_features="", is_global=True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.folder = folder
        self.monotonous_features = utils.transform_string_feature_range_into_list(
            monotonous_features)
        self.is_global = is_global
        self.features_dict_with_saved_objects = {}
        if not folder.strip():
            self.xg_boost = XGBClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          random_state=43)
        else:
            self.load_model(folder)

    def get_model_info(self):
        folder_name = os.path.basename(self.folder.strip("/").strip("\\")).strip()
        if folder_name:
            tags = [folder_name]
            if not self.is_global:
                return tags + ["custom boosting model"]
            return tags + ["global boosting model"]
        return []

    def get_feature_ids(self):
        return utils.transform_string_feature_range_into_list(self.feature_ids)\
            if isinstance(self.feature_ids, str) else self.feature_ids

    def get_feature_names(self):
        feature_ids = self.get_feature_ids()
        feature_names = []
        for _id in feature_ids:
            if _id in self.features_dict_with_saved_objects:
                feature_names_from_encodings = self.features_dict_with_saved_objects[_id].get_feature_names()
                feature_names.extend(
                    [str(_id) + "_" + feature_name for feature_name in feature_names_from_encodings])
            else:
                feature_names.append(str(_id))
        return feature_names

    def add_config_info(self, full_config, features, monotonous_features):
        self.full_config = full_config
        self.feature_ids = features
        self.monotonous_features = monotonous_features

    def transform_feature_encoders_to_dict(self):
        features_dict_with_saved_objects = {}
        for feature in self.features_dict_with_saved_objects:
            feature_info = self.features_dict_with_saved_objects[feature].save_to_feature_info()
            features_dict_with_saved_objects[feature] = feature_info
        return features_dict_with_saved_objects

    def transform_feature_encoders_to_objects(self, features_dict_with_saved_objects):
        _features_dict_with_saved_objects = {}
        for feature in features_dict_with_saved_objects:
            _feature_encoder = feature_encoder.FeatureEncoder()
            _feature_encoder.load_from_feature_info(features_dict_with_saved_objects[feature])
            _features_dict_with_saved_objects[feature] = _feature_encoder
        return _features_dict_with_saved_objects

    def load_model(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "boost_model.pickle"), "rb") as f:
            self.n_estimators, self.max_depth, self.xg_boost = pickle.load(f)
        with open(os.path.join(folder, "data_features_config.pickle"), "rb") as f:
            self.full_config, self.feature_ids, self.monotonous_features = pickle.load(f)
        if os.path.exists(os.path.join(folder, "features_dict_with_saved_objects.pickle")):
            features_dict_with_saved_objects = {}
            with open(os.path.join(folder, "features_dict_with_saved_objects.pickle"), "rb") as f:
                features_dict_with_saved_objects = pickle.load(f)
            self.features_dict_with_saved_objects = self.transform_feature_encoders_to_objects(
                features_dict_with_saved_objects)
        else:
            self.features_dict_with_saved_objects = {}

    def save_model(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, "boost_model.pickle"), "wb") as f:
            pickle.dump([self.n_estimators, self.max_depth, self.xg_boost], f)
        with open(os.path.join(folder, "data_features_config.pickle"), "wb") as f:
            pickle.dump([self.full_config, self.feature_ids, self.monotonous_features], f)
        with open(os.path.join(folder, "features_dict_with_saved_objects.pickle"), "wb") as f:
            pickle.dump(self.transform_feature_encoders_to_dict(), f)

    def train_model(self, train_data, labels):
        mon_features = [
            (1 if feature in self.monotonous_features else 0) for feature in self.get_feature_ids()]
        mon_features_prepared = "(" + ",".join([str(f) for f in mon_features]) + ")"
        self.xg_boost = XGBClassifier(n_estimators=self.n_estimators,
                                      max_depth=self.max_depth, random_state=43,
                                      monotone_constraints=mon_features_prepared)
        self.xg_boost.fit(train_data, labels)
        logger.info("Train score: %s", self.xg_boost.score(train_data, labels))
        logger.info("Feature importances: %s", self.xg_boost.feature_importances_)

    def validate_model(self, valid_test_set, valid_test_labels):
        res, res_prob = self.predict(valid_test_set)
        f1_score = self.xg_boost.score(valid_test_set, valid_test_labels)
        logger.info("Valid dataset F1 score: %s", f1_score)
        logger.info(confusion_matrix(valid_test_labels, res))
        logger.info(classification_report(valid_test_labels, res))
        return f1_score

    def predict(self, data):
        if not len(data):
            return [], []
        return self.xg_boost.predict(data), self.xg_boost.predict_proba(data)
