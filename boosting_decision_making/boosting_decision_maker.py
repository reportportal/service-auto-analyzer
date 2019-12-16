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

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle
import logging

logger = logging.getLogger("analyzerApp.boosting_decision_maker")


class BoostingDecisionMaker:

    def __init__(self, folder="", n_estimators=50, max_depth=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.folder = folder
        if folder.strip() == "":
            self.xg_boost = XGBClassifier(n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          random_state=43)
        else:
            self.load_model(folder)

    def get_feature_ids(self):
        return self.feature_ids

    def add_config_info(self, full_config, features):
        self.full_config = full_config
        self.feature_ids = features

    def load_model(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "boost_model.pickle"), "rb") as f:
            self.n_estimators, self.max_depth, self.xg_boost = pickle.load(f)
        with open(os.path.join(folder, "data_features_config.pickle"), "rb") as f:
            self.full_config, self.feature_ids = pickle.load(f)

    def save_model(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, "boost_model.pickle"), "wb") as f:
            pickle.dump([self.n_estimators, self.max_depth, self.xg_boost], f)
        with open(os.path.join(folder, "data_features_config.pickle"), "wb") as f:
            pickle.dump([self.full_config, self.feature_ids], f)

    def train_model(self, train_data, labels):
        self.xg_boost.fit(train_data, labels)
        logger.info("Train score: ", self.xg_boost.score(train_data, labels))
        logger.info("Feature importances: ", self.xg_boost.feature_importances_)

    def validate_model(self, valid_test_set, valid_test_labels):
        res, res_prob = self.predict(valid_test_set)
        logger.info("Valid dataset F1 score: ",
                    self.xg_boost.score(valid_test_set, valid_test_labels))
        logger.info(confusion_matrix(valid_test_labels, res))
        logger.info(classification_report(valid_test_labels, res))

    def predict(self, data):
        return self.xg_boost.predict(data), self.xg_boost.predict_proba(data)
