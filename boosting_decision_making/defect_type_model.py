from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import pickle
import re


class DefectTypeModel:

    def __init__(self, folder="", is_global=True):
        self.folder = folder
        self.count_vectorizer_models = {}
        self.models = {}
        self.is_global = is_global
        if self.folder:
            self.load_model(folder)

    def preprocess_words(self, text):
        all_words = []
        for w in re.finditer(r"[\w\._]+", text):
            word_normalized = re.sub(r"^[\w]\.", "", w.group(0))
            word = word_normalized.replace("_", "")
            if len(word) >= 3:
                all_words.append(word.lower())
            split_parts = word_normalized.split("_")
            split_words = []
            if len(split_parts) > 2:
                for idx in range(len(split_parts)):
                    if idx != len(split_parts) - 1:
                        split_words.append("".join(split_parts[idx:idx + 2]).lower())
                all_words.extend(split_words)
            if "." not in word_normalized:
                split_words = []
                split_parts = [s.strip() for s in re.split("([A-Z][^A-Z]+)", word) if s.strip()]
                if len(split_parts) > 2:
                    for idx in range(len(split_parts)):
                        if idx != len(split_parts) - 1:
                            if len("".join(split_parts[idx:idx + 2]).lower()) > 3:
                                split_words.append("".join(split_parts[idx:idx + 2]).lower())
                all_words.extend(split_words)
        return all_words

    def get_model_info(self):
        folder_name = os.path.basename(self.folder).strip()
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

    def train_model(self, train_data):
        for name, train_data_x, labels in train_data:
            self.count_vectorizer_models[name] = TfidfVectorizer(
                binary=True, stop_words="english", min_df=5,
                token_pattern=r"[\w\._]+", analyzer=self.preprocess_words)
            transformed_values = self.count_vectorizer_models[name].fit_transform(train_data_x)
            model = RandomForestClassifier()
            x_train_values = pd.DataFrame(
                transformed_values.toarray(),
                columns=self.count_vectorizer_models[name].get_feature_names())
            model.fit(x_train_values, labels)
            self.models[name] = model

    def validate_model(self, test_data):
        results = []
        for name, test_data_x, labels in test_data:
            assert name in self.models
            print("Model name: %s" % name)
            res, res_prob = self.predict(test_data_x, name)
            print("Valid dataset F1 score: ", f1_score(y_pred=res, y_true=labels))
            print(confusion_matrix(y_pred=res, y_true=labels))
            print(classification_report(y_pred=res, y_true=labels))
            results.append(
                (name,
                 f1_score(y_pred=res, y_true=labels),
                 accuracy_score(y_pred=res, y_true=labels)))
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
