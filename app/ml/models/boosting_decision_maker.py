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
import json
from typing import Any, Optional

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing_extensions import override
from xgboost import XGBClassifier

from app.commons import logging
from app.commons.object_saving.object_saver import ObjectSaver
from app.ml.models import MlModel
from app.utils import text_processing

LOGGER = logging.getLogger("analyzerApp.boosting_decision_maker")

MODEL_FILES: list[str] = ["boost_model.pickle", "data_features_config.pickle"]
DEFAULT_RANDOM_STATE = 43
DEFAULT_N_ESTIMATORS = 75
DEFAULT_MAX_DEPTH = 5
DEFAULT_LEARNING_RATE = 0.2


class BoostingDecisionMaker(MlModel):
    _loaded: bool
    n_estimators: int
    max_depth: int
    random_state: int
    learning_rate: float
    feature_ids: list[int]
    monotonous_features: set[int]
    boost_model: Any

    def __init__(
        self,
        object_saver: ObjectSaver,
        tags: str = "global boosting model",
        *,
        features: Optional[list[int]] = None,
        monotonous_features: Optional[list[int]] = None,
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        super().__init__(object_saver, tags)
        self.n_estimators = n_estimators if n_estimators is not None else DEFAULT_N_ESTIMATORS
        self.max_depth = max_depth if max_depth is not None else DEFAULT_MAX_DEPTH
        self.random_state = random_state if random_state is not None else DEFAULT_RANDOM_STATE
        self.learning_rate = learning_rate if learning_rate is not None else DEFAULT_LEARNING_RATE
        self.boost_model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=self.random_state
        )
        self.feature_ids = features if features else []
        self.monotonous_features = set(monotonous_features) if monotonous_features else set()
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    @override
    def is_custom(self) -> bool:
        """Indicates whether the model is custom or not."""
        return False

    @property
    def feature_importances(self) -> Optional[dict[int, float]]:
        if self.loaded:
            return dict(zip(self.feature_ids, self.boost_model.feature_importances_.tolist()))
        return None

    def load_model(self) -> None:
        if self.loaded:
            return
        boost_model, features_config = self._load_models(MODEL_FILES)
        if len(boost_model) > 3:
            # New model format
            self.n_estimators, self.max_depth, self.random_state, self.boost_model = boost_model
            self.feature_ids, self.monotonous_features = features_config
        else:
            # Old model format
            self.n_estimators, self.max_depth, self.boost_model = boost_model
            self.random_state = DEFAULT_RANDOM_STATE
            _, features, self.monotonous_features = features_config
            self.feature_ids = text_processing.transform_string_feature_range_into_list(features)
        self._loaded = True

    def save_model(self):
        self._save_models(
            zip(
                MODEL_FILES,
                [
                    [self.n_estimators, self.max_depth, self.random_state, self.boost_model],
                    [self.feature_ids, self.monotonous_features],
                ],
            )
        )

    def train_model(self, train_data: list[list[float]], labels: list[int]) -> float:
        mon_features = [(1 if feature in self.monotonous_features else 0) for feature in self.feature_ids]
        mon_features_prepared = "(" + ",".join([str(f) for f in mon_features]) + ")"
        self.boost_model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            monotone_constraints=mon_features_prepared,
            learning_rate=self.learning_rate,
        )
        self.boost_model.fit(train_data, labels)
        self._loaded = True
        res = self.boost_model.predict(train_data)
        f1 = f1_score(y_pred=res, y_true=labels)
        if f1 is None:
            f1 = 0.0
        LOGGER.debug(f"Train dataset F1 score: {f1:.5f}")
        LOGGER.debug(
            "Feature importances: %s",
            json.dumps(dict(zip(self.feature_ids, self.boost_model.feature_importances_.tolist()))),
        )
        return f1

    def predict(self, data: list[list[float]]) -> tuple[list[int], list[list[float]]]:
        if not len(data):
            return [], []
        return self.boost_model.predict(data).tolist(), self.boost_model.predict_proba(data).tolist()

    def validate_model(self, valid_test_set: list[list[float]], valid_test_labels: list[int]) -> float:
        res, _ = self.predict(valid_test_set)
        f1 = f1_score(y_pred=res, y_true=valid_test_labels)
        if f1 is None:
            f1 = 0.0
        LOGGER.debug(f"Valid dataset F1 score: {f1:.5f}")
        LOGGER.debug(f"\n{confusion_matrix(valid_test_labels, res)}")
        LOGGER.debug(f"\n{classification_report(valid_test_labels, res)}")
        return f1
