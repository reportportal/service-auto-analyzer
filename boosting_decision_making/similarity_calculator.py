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

from utils import utils
from scipy import spatial
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class SimilarityCalculator:

    def __init__(self, config, weighted_similarity_calculator=None):
        self.weighted_similarity_calculator = weighted_similarity_calculator
        self.config = config
        self.similarity_dict = {}

    def find_similarity(self, all_results, fields):
        for field in fields:
            if field in self.similarity_dict:
                continue
            self.similarity_dict[field] = {}
            log_field_ids = {}
            index_in_message_array = 0
            count_vector_matrix = None
            all_messages = []
            for log, res in all_results:
                for obj in [log] + res["hits"]["hits"]:
                    if obj["_id"] not in log_field_ids:
                        if self.weighted_similarity_calculator is None:
                            text = " ".join(utils.split_words(obj["_source"][field],
                                            min_word_length=self.config["min_word_length"]))
                            if text.strip() == "":
                                log_field_ids[obj["_id"]] = -1
                            else:
                                all_messages.append(text)
                                log_field_ids[obj["_id"]] = index_in_message_array
                                index_in_message_array += 1
                        else:
                            if obj["_source"][field].strip() == "":
                                log_field_ids[obj["_id"]] = -1
                            else:
                                text = []
                                if field == "message" and self.config["number_of_log_lines"] == -1:
                                    text = self.weighted_similarity_calculator.message_to_array(
                                        obj["_source"]["detected_message"],
                                        obj["_source"]["stacktrace"])
                                elif field == "stacktrace":
                                    text = self.weighted_similarity_calculator.message_to_array(
                                        "", obj["_source"]["stacktrace"])
                                else:
                                    text = utils.filter_empty_lines([" ".join(utils.split_words(
                                        obj["_source"][field],
                                        min_word_length=self.config["min_word_length"]))])
                                if len(text) == 0:
                                    log_field_ids[obj["_id"]] = -1
                                else:
                                    all_messages.extend(text)
                                    log_field_ids[obj["_id"]] = [index_in_message_array,
                                                                 len(all_messages) - 1]
                                    index_in_message_array += len(text)
            if len(all_messages) > 0:
                vectorizer = CountVectorizer(binary=True, analyzer="word", token_pattern="[^ ]+")
                count_vector_matrix = np.asarray(vectorizer.fit_transform(all_messages).toarray())
            for log, res in all_results:
                sim_dict = self._calculate_field_similarity(
                    log, res, log_field_ids, count_vector_matrix)
                for key in sim_dict:
                    self.similarity_dict[field][key] = sim_dict[key]

    def _calculate_field_similarity(self, log, res, log_field_ids, count_vector_matrix):
        all_results_similarity = {}
        for obj in res["hits"]["hits"]:
            group_id = (obj["_id"], log["_id"])
            index_query_message = log_field_ids[log["_id"]]
            index_log_message = log_field_ids[obj["_id"]]
            if (type(index_query_message) == int and index_query_message < 0) and\
                    (type(index_log_message) == int and index_log_message < 0):
                all_results_similarity[group_id] = {"similarity": 1.0, "both_empty": True}
            elif (type(index_query_message) == int and index_query_message < 0) or\
                    (type(index_log_message) == int and index_log_message < 0):
                all_results_similarity[group_id] = {"similarity": 0.0, "both_empty": False}
            else:
                if self.weighted_similarity_calculator is None:
                    similarity =\
                        round(1 - spatial.distance.cosine(
                            count_vector_matrix[index_query_message],
                            count_vector_matrix[index_log_message]), 3)
                else:
                    query_vector = self.weighted_similarity_calculator.weigh_data_rows(
                        count_vector_matrix[index_query_message[0]:index_query_message[1] + 1])
                    log_vector = self.weighted_similarity_calculator.weigh_data_rows(
                        count_vector_matrix[index_log_message[0]:index_log_message[1] + 1])
                    similarity =\
                        round(1 - spatial.distance.cosine(query_vector, log_vector), 3)
                all_results_similarity[group_id] = {"similarity": similarity, "both_empty": False}

        return all_results_similarity
