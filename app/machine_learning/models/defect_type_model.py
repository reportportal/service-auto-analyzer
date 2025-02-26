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

import re
from collections import Counter
from typing import Optional, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from app.commons import logging
from app.commons.object_saving.object_saver import ObjectSaver
from app.machine_learning.models import MlModel
from app.utils import text_processing
from app.utils.defaultdict import DefaultDict

LOGGER = logging.getLogger('analyzerApp.DefectTypeModel')
MODEL_FILES: list[str] = ['count_vectorizer_models.pickle', 'models.pickle']
DATA_FIELD = 'detected_message_without_params_extended'
BASE_DEFECT_TYPE_PATTERN = re.compile(r'^(?:([^_]+)_\S+|(\D+)\d+)$')
DEFAULT_N_ESTIMATORS = 10
DEFAULT_MIN_SAMPLES_LEAF = 1
DEFAULT_MAX_FEATURES = 'sqrt'


# noinspection PyMethodMayBeStatic
class DummyVectorizer:
    def transform(self, data: list[str]) -> csr_matrix:
        return csr_matrix(np.zeros((len(data), 1)))

    def get_feature_names_out(self) -> list[str]:
        return ['dummy']


# noinspection PyMethodMayBeStatic
class DummyClassifier:
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return np.zeros(data.shape[0])

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        return np.zeros((data.shape[0], 2))


DEFAULT_MODEL = DummyClassifier()
DEFAULT_VECTORIZER = DummyVectorizer()


def get_model(self: DefaultDict, model_name: str, default_value: any) -> Any:
    m = BASE_DEFECT_TYPE_PATTERN.match(model_name)
    if not m:
        raise KeyError(model_name)
    base_model_name = (m.group(1) or m.group(2)).strip()
    if not base_model_name:
        raise KeyError(model_name)
    if base_model_name in self:
        return self[base_model_name]
    else:
        return default_value


def get_vectorizer_model(self: Any, model_name: str) -> Any:
    return get_model(self, model_name, DEFAULT_VECTORIZER)


def get_classifier_model(self: Any, model_name: str) -> Any:
    return get_model(self, model_name, DEFAULT_MODEL)


class DefectTypeModel(MlModel):
    _loaded: bool
    count_vectorizer_models: DefaultDict[str, TfidfVectorizer | DummyVectorizer]
    models: DefaultDict[str, RandomForestClassifier | DummyClassifier]
    n_estimators: int

    def __init__(self, object_saver: ObjectSaver, tags: str = 'global defect type model', *,
                 n_estimators: Optional[int] = None) -> None:
        super().__init__(object_saver, tags)
        self._loaded = False
        self.count_vectorizer_models = DefaultDict(get_vectorizer_model)
        self.models = DefaultDict(get_classifier_model)
        self.n_estimators = n_estimators if n_estimators is not None else DEFAULT_N_ESTIMATORS

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        if self.loaded:
            return
        model = self._load_models(MODEL_FILES)
        self.count_vectorizer_models = DefaultDict(get_vectorizer_model, **model[0])
        self.models = DefaultDict(get_classifier_model, **model[1])
        self._loaded = True

    def save_model(self):
        self._save_models(zip(MODEL_FILES, [self.count_vectorizer_models, self.models]))

    def train_model(self, name: str, train_data_x: list[str], labels: list[int], random_state: int) -> float:
        self.count_vectorizer_models[name] = TfidfVectorizer(
            binary=True, min_df=5, analyzer=text_processing.preprocess_words)
        transformed_values = self.count_vectorizer_models[name].fit_transform(train_data_x)
        LOGGER.debug(f'Length of train data: {len(labels)}')
        LOGGER.debug(f'Train data label distribution: {Counter(labels)}')
        LOGGER.debug(f'Train model name: {name}; estimators number: {self.n_estimators}')
        model = RandomForestClassifier(self.n_estimators, class_weight='balanced',
                                       min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF, max_features=DEFAULT_MAX_FEATURES,
                                       random_state=random_state)
        x_train_values = pd.DataFrame(
            transformed_values.toarray(),
            columns=self.count_vectorizer_models[name].get_feature_names_out())
        model.fit(x_train_values, labels)
        self.models[name] = model
        self._loaded = True
        res = model.predict(x_train_values)
        f1 = f1_score(y_pred=res, y_true=labels)
        if f1 is None:
            f1 = 0.0
        LOGGER.debug(f'Train dataset F1 score: {f1:.5f}')
        return f1

    def validate_model(self, name: str, test_data_x: list[str], labels: list[int]) -> float:
        assert name in self.models
        LOGGER.debug(f'Validation data label distribution: {Counter(labels)}')
        LOGGER.debug(f'Validation model name: {name}')
        res, res_prob = self.predict(test_data_x, name)
        f1 = f1_score(y_pred=res, y_true=labels)
        if f1 is None:
            f1 = 0.0
        LOGGER.debug(f'Valid dataset F1 score: {f1:.5f}')
        LOGGER.debug(f'\n{confusion_matrix(y_pred=res, y_true=labels)}')
        LOGGER.debug(f'\n{classification_report(y_pred=res, y_true=labels)}')
        return f1

    def predict(self, data: list, model_name: str) -> tuple[list, list]:
        if len(data) == 0:
            return [], []
        transformed_values = self.count_vectorizer_models[model_name].transform(data)
        x_test_values = pd.DataFrame(
            transformed_values.toarray(), columns=self.count_vectorizer_models[model_name].get_feature_names_out())
        predicted_labels = self.models[model_name].predict(x_test_values)
        predicted_probs = self.models[model_name].predict_proba(x_test_values)
        return predicted_labels, predicted_probs
