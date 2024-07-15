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

from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from app.commons import logging
from app.commons.object_saving.object_saver import ObjectSaver
from app.machine_learning import feature_encoder
from app.machine_learning.models import MlModel
from app.utils import text_processing

logger = logging.getLogger("analyzerApp.boosting_decision_maker")

MODEL_FILES: list[str] = ['boost_model.pickle', 'data_features_config.pickle',
                          'features_dict_with_saved_objects.pickle']


class BoostingDecisionMaker(MlModel):
    _loaded: bool

    def __init__(self, object_saver: ObjectSaver, tags: str = 'global boosting model', *, n_estimators: int = 75,
                 max_depth: int = 5, monotonous_features: str = '') -> None:
        super().__init__(object_saver, tags)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.xg_boost = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=43)
        self.monotonous_features = text_processing.transform_string_feature_range_into_list(
            monotonous_features)
        self.features_dict_with_saved_objects = {}
        self.full_config = {}
        self.feature_ids = []
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def get_feature_ids(self) -> list[int]:
        return text_processing.transform_string_feature_range_into_list(self.feature_ids) \
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

    def load_model(self) -> None:
        if self.loaded:
            return
        boost_model, features_config, features_dict = self._load_models(MODEL_FILES)
        self.n_estimators, self.max_depth, self.xg_boost = boost_model
        self.full_config, self.feature_ids, self.monotonous_features = features_config
        self.features_dict_with_saved_objects = self.transform_feature_encoders_to_objects(features_dict)
        self._loaded = True

    def save_model(self):
        self._save_models(zip(MODEL_FILES, [[self.n_estimators, self.max_depth, self.xg_boost],
                                            [self.full_config, self.feature_ids, self.monotonous_features],
                                            self.transform_feature_encoders_to_dict()]))

    def train_model(self, train_data, labels):
        mon_features = [
            (1 if feature in self.monotonous_features else 0) for feature in self.get_feature_ids()]
        mon_features_prepared = "(" + ",".join([str(f) for f in mon_features]) + ")"
        self.xg_boost = XGBClassifier(n_estimators=self.n_estimators,
                                      max_depth=self.max_depth, random_state=43,
                                      monotone_constraints=mon_features_prepared)
        self.xg_boost.fit(train_data, labels)
        self._loaded = True
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
