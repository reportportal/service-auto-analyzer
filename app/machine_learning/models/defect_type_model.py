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

from collections import Counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, accuracy_score

from app.commons.object_saving.object_saver import ObjectSaver
from app.machine_learning.models import MlModel
from app.utils import text_processing

MODEL_FILES: list[str] = ['count_vectorizer_models.pickle', 'models.pickle']


class DefectTypeModel(MlModel):
    _loaded: bool
    count_vectorizer_models: dict[str, TfidfVectorizer]
    models: dict[str, RandomForestClassifier]

    def __init__(self, object_saver: ObjectSaver, tags: str = 'global defect type model') -> None:
        super().__init__(object_saver, tags)
        self._loaded = False
        self.count_vectorizer_models = {}
        self.models = {}

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load_model(self) -> None:
        if self.loaded:
            return
        model = self._load_models(MODEL_FILES)
        self.count_vectorizer_models, self.models = model
        self._loaded = True

    def save_model(self):
        self._save_models(zip(MODEL_FILES, [self.count_vectorizer_models, self.models]))

    def train_model(self, name: str, train_data_x, labels):
        self.count_vectorizer_models[name] = TfidfVectorizer(
            binary=True, min_df=5, analyzer=text_processing.preprocess_words)
        transformed_values = self.count_vectorizer_models[name].fit_transform(train_data_x)
        print("Length of train data: ", len(labels))
        print("Label distribution:", Counter(labels))
        model = RandomForestClassifier(class_weight='balanced')
        x_train_values = pd.DataFrame(
            transformed_values.toarray(),
            columns=self.count_vectorizer_models[name].get_feature_names_out())
        model.fit(x_train_values, labels)
        self.models[name] = model
        self._loaded = True

    def validate_model(self, name, test_data_x, labels):
        assert name in self.models
        print("Label distribution:", Counter(labels))
        print("Model name: %s" % name)
        res, res_prob = self.predict(test_data_x, name)
        print("Valid dataset F1 score: ", f1_score(y_pred=res, y_true=labels))
        print(confusion_matrix(y_pred=res, y_true=labels))
        print(classification_report(y_pred=res, y_true=labels))
        f1 = f1_score(y_pred=res, y_true=labels)
        if f1 is None:
            f1 = 0.0
        accuracy = accuracy_score(y_pred=res, y_true=labels)
        if accuracy is None:
            accuracy = 0.0
        return f1, accuracy

    def validate_models(self, test_data):
        results = []
        for name, test_data_x, labels in test_data:
            f1, accuracy = self.validate_model(
                name, test_data_x, labels)
            results.append((name, f1, accuracy))
        return results

    def predict(self, data, model_name):
        assert model_name in self.models
        if len(data) == 0:
            return [], []
        transformed_values = self.count_vectorizer_models[model_name].transform(data)
        x_test_values = pd.DataFrame(
            transformed_values.toarray(), columns=self.count_vectorizer_models[model_name].get_feature_names_out())
        predicted_labels = self.models[model_name].predict(x_test_values)
        predicted_probs = self.models[model_name].predict_proba(x_test_values)
        return predicted_labels, predicted_probs
