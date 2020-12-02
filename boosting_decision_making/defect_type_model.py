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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from utils import utils
import pandas as pd
import os
import pickle
from collections import Counter


class DefectTypeModel:

    def __init__(self, folder=""):
        self.folder = folder
        self.count_vectorizer_models = {}
        self.models = {}
        self.is_global = True
        if self.folder:
            self.load_model(folder)

    def get_model_info(self):
        folder_name = os.path.basename(self.folder.strip("/").strip("\\")).strip()
        if folder_name:
            tags = [folder_name]
            if not self.is_global:
                return tags + ["custom defect type model"]
            return tags + ["global defect type model"]
        return []

    def load_model(self, folder):
        self.count_vectorizer_models = pickle.load(
            open(os.path.join(folder, "count_vectorizer_models.pickle"), "rb"))
        self.models = pickle.load(open(os.path.join(folder, "models.pickle"), "rb"))

    def save_model(self, folder):
        os.makedirs(folder, exist_ok=True)
        pickle.dump(self.count_vectorizer_models,
                    open(os.path.join(folder, "count_vectorizer_models.pickle"), "wb"))
        pickle.dump(self.models, open(os.path.join(folder, "models.pickle"), "wb"))

    def train_model(self, name, train_data_x, labels):
        self.count_vectorizer_models[name] = TfidfVectorizer(
            binary=True, stop_words="english", min_df=5,
            token_pattern=r"[\w\._]+", analyzer=utils.preprocess_words)
        transformed_values = self.count_vectorizer_models[name].fit_transform(train_data_x)
        print("Length of train data: ", len(labels))
        print("Label distribution:", Counter(labels))
        model = RandomForestClassifier()
        x_train_values = pd.DataFrame(
            transformed_values.toarray(),
            columns=self.count_vectorizer_models[name].get_feature_names())
        model.fit(x_train_values, labels)
        self.models[name] = model

    def train_models(self, train_data):
        for name, train_data_x, labels in train_data:
            self.train_model(name, train_data_x, labels)

    def validate_model(self, name, test_data_x, labels):
        assert name in self.models
        print("Label distribution:", Counter(labels))
        print("Model name: %s" % name)
        res, res_prob = self.predict(test_data_x, name)
        print("Valid dataset F1 score: ", f1_score(y_pred=res, y_true=labels))
        print(confusion_matrix(y_pred=res, y_true=labels))
        print(classification_report(y_pred=res, y_true=labels))
        f1 = f1_score(y_pred=res, y_true=labels)
        accuracy = accuracy_score(y_pred=res, y_true=labels)
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
            transformed_values.toarray(),
            columns=self.count_vectorizer_models[model_name].get_feature_names())
        predicted_labels = self.models[model_name].predict(x_test_values)
        predicted_probs = self.models[model_name].predict_proba(x_test_values)
        return predicted_labels, predicted_probs
