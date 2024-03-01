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

import logging
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import re

from app.utils import text_processing

logger = logging.getLogger("analyzerApp.feature_encoder")


class FeatureEncoder:

    def __init__(self, field_name="", encoding_type="", max_features=50, ngram_max=2):
        self.field_name = field_name
        self.encoding_type = encoding_type
        self.max_features = max_features
        self.additional_info = {}
        self.ngram_max = ngram_max
        self.encoder = None
        self.prepare_text_functions = {
            "launch_name": FeatureEncoder.prepare_text_launch_name,
            "detected_message": FeatureEncoder.prepare_text_message,
            "stacktrace": FeatureEncoder.prepare_stacktrace,
            "test_item_name": FeatureEncoder.prepare_test_item_name,
            "found_exceptions": FeatureEncoder.prepare_found_exceptions
        }

    @staticmethod
    def add_default_value(texts, default_value):
        return [(text if text.strip() else default_value) for text in texts]

    @staticmethod
    def prepare_text_message(data):
        messages = [" ".join(text_processing.split_words(text)).replace(".", "_") for text in data]
        return FeatureEncoder.add_default_value(messages, "nomessage")

    @staticmethod
    def prepare_stacktrace(data):
        stacktraces = [
            " ".join([w for w in text_processing.split_words(text) if "." in w]).replace(".", "_") for text in data]
        return FeatureEncoder.add_default_value(stacktraces, "nostacktrace")

    @staticmethod
    def prepare_found_exceptions(data):
        found_exceptions = [text.replace(".", "_") for text in data]
        return FeatureEncoder.add_default_value(found_exceptions, "noexception")

    @staticmethod
    def prepare_text_launch_name(data):
        launch_names = [re.sub(r"\d+", " ", text.replace("-", " ").replace("_", " ")) for text in data]
        return FeatureEncoder.add_default_value(launch_names, "nolaunchname")

    @staticmethod
    def prepare_test_item_name(data):
        test_item_names = [re.sub(r"\d+", " ", text).replace(".", "_") for text in data]
        return FeatureEncoder.add_default_value(test_item_names, "notestitemname")

    @staticmethod
    def encode_categories(data, categories_data, include_zero=False):
        encoded_data = []
        for d_ in data:
            if d_ in categories_data:
                encoded_data.append([categories_data[d_]])
            elif include_zero:
                encoded_data.append([0])
        return encoded_data

    def get_feature_names(self) -> list[str]:
        feature_names = []
        if self.encoding_type == "one_hot":
            for _key in sorted(self.additional_info.items(), key=lambda x: x[1]):
                feature_names.append(_key[0])
        elif self.encoding_type == "hashing":
            feature_names = [str(x_) for x_ in range(self.max_features)]
        else:
            feature_names = self.encoder.get_feature_names_out().tolist()
        return feature_names

    def extract_data(self, logs):
        data_gathered = []
        for log in logs:
            if self.field_name in log["_source"]:
                data_gathered.append(log["_source"][self.field_name])
        return data_gathered

    def get_categories(self, data):
        data_frequency = {}
        for d_ in data:
            if d_ not in data_frequency:
                data_frequency[d_] = 0
            data_frequency[d_] += 1
        sorted_freq = sorted(data_frequency.items(), key=lambda x: x[1], reverse=True)
        idx = 1
        categories_labelling = {}
        for name, cnt in sorted_freq[:self.max_features]:
            categories_labelling[name] = idx
            idx += 1
        return categories_labelling

    def prepare_data_for_encoding(self, data, include_zero=False):
        if self.encoding_type == "one_hot":
            data = FeatureEncoder.encode_categories(data, self.additional_info, include_zero=include_zero)
        else:
            if self.field_name in self.prepare_text_functions:
                data = self.prepare_text_functions[self.field_name](data)
            else:
                logger.error("Prepare text function is not defined for the field '%s'" % self.field_name)
        return data

    def fit(self, texts):
        if self.encoding_type == "one_hot":
            self.encoder = OneHotEncoder(handle_unknown='ignore')
        elif self.encoding_type == "hashing":
            self.encoder = HashingVectorizer(
                n_features=self.max_features, ngram_range=(1, self.ngram_max), stop_words="english")
        elif self.encoding_type == "count_vector":
            self.encoder = CountVectorizer(
                max_features=self.max_features, ngram_range=(1, self.ngram_max),
                binary=True, stop_words="english")
        elif self.encoding_type == "tf_idf":
            self.encoder = TfidfVectorizer(
                max_features=self.max_features, ngram_range=(1, self.ngram_max), stop_words="english")
        else:
            logger.error("Encoding type '%s' is not found", self.encoding_type)
        if self.encoder:
            extracted_data = self.extract_data(texts)
            logger.debug("Extracted data %d", len(extracted_data))
            if self.encoding_type == "one_hot":
                self.additional_info = self.get_categories(extracted_data)
            prepared_data = self.prepare_data_for_encoding(extracted_data)
            logger.debug("Prepared data %d", len(prepared_data))
            self.encoder.fit(prepared_data)
            logger.debug("Fit data with encoding '%s'" % self.encoding_type)

    def transform(self, data):
        if self.encoder:
            prepared_data = self.prepare_data_for_encoding(data, include_zero=True)
            return self.encoder.transform(prepared_data)
        else:
            logger.error("Encoder was not fit")
            return []

    def load_from_feature_info(self, feature_info):
        self.field_name = feature_info["field_name"]
        self.encoding_type = feature_info["encoding_type"]
        self.max_features = feature_info["max_features"]
        self.additional_info = feature_info["additional_info"]
        self.ngram_max = feature_info["ngram_max"]
        self.encoder = feature_info["encoder"]

    def save_to_feature_info(self):
        feature_info = {"field_name": self.field_name, "encoding_type": self.encoding_type,
                        "max_features": self.max_features, "additional_info": self.additional_info,
                        "encoder": self.encoder, "ngram_max": self.ngram_max}
        return feature_info
