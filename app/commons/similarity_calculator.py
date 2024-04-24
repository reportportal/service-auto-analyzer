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

import numpy as np
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer

from app.machine_learning.models.weighted_similarity_calculator import WeightedSimilarityCalculator
from app.utils import text_processing


class SimilarityCalculator:
    similarity_model: WeightedSimilarityCalculator

    def __init__(self, config, similarity_model: WeightedSimilarityCalculator):
        self.similarity_model = similarity_model
        self.config = config
        self.similarity_dict = {}
        self.object_id_weights = {}
        self.fields_mapping_for_weighting = {
            "message_without_params_extended": [
                "detected_message_without_params_extended", "stacktrace_extended"],
            "message_extended": ["detected_message_extended", "stacktrace_extended"]
        }
        self.artificial_columns = ["namespaces_stacktrace"]

    def find_similarity(self, all_results, fields):
        for field in fields:
            if field in self.similarity_dict:
                continue
            self.similarity_dict[field] = {}
            log_field_ids = {}
            index_in_message_array = 0
            count_vector_matrix = None
            all_messages = []
            all_messages_needs_reweighting = []
            needs_reweighting_wc = False
            for log, res in all_results:
                for obj in [log] + res["hits"]["hits"]:
                    if obj["_id"] not in log_field_ids:
                        if field not in self.artificial_columns and not obj["_source"][field].strip():
                            log_field_ids[obj["_id"]] = -1
                        else:
                            text = []
                            needs_reweighting = 0
                            if self.config["number_of_log_lines"] == -1 and \
                                    field in self.fields_mapping_for_weighting:
                                fields_to_use = self.fields_mapping_for_weighting[field]
                                text = self.similarity_model.message_to_array(
                                    obj["_source"][fields_to_use[0]],
                                    obj["_source"][fields_to_use[1]])
                            elif field == "namespaces_stacktrace":
                                gathered_lines = []
                                weights = []
                                for line in obj["_source"]["stacktrace"].split("\n"):
                                    line_words = text_processing.split_words(
                                        line,
                                        min_word_length=self.config["min_word_length"])
                                    for word in line_words:
                                        part_of_namespace = ".".join(word.split(".")[:2])
                                        if part_of_namespace in self.config["chosen_namespaces"]:
                                            gathered_lines.append(" ".join(line_words))
                                            weights.append(
                                                self.config["chosen_namespaces"][part_of_namespace])
                                if len(gathered_lines):
                                    text = gathered_lines
                                    self.object_id_weights[obj["_id"]] = weights
                                else:
                                    text = []
                                    for line in obj["_source"]["stacktrace"].split("\n"):
                                        text.append(" ".join(text_processing.split_words(
                                            text_processing.clean_from_brackets(line),
                                            min_word_length=self.config["min_word_length"])))
                                    text = text_processing.filter_empty_lines(text)
                                    self.object_id_weights[obj["_id"]] = [1] * len(text)
                            elif field.startswith("stacktrace"):
                                if text_processing.does_stacktrace_need_words_reweighting(obj["_source"][field]):
                                    needs_reweighting = 1
                                text = self.similarity_model.message_to_array(
                                    "", obj["_source"][field])
                            else:
                                text = text_processing.filter_empty_lines([" ".join(
                                    text_processing.split_words(
                                        obj["_source"][field],
                                        min_word_length=self.config["min_word_length"]))])
                            if not text:
                                log_field_ids[obj["_id"]] = -1
                            else:
                                all_messages.extend(text)
                                all_messages_needs_reweighting.append(needs_reweighting)
                                log_field_ids[obj["_id"]] = [index_in_message_array,
                                                             len(all_messages) - 1]
                                index_in_message_array += len(text)
            if all_messages:
                needs_reweighting_wc = all_messages_needs_reweighting and \
                                       sum(all_messages_needs_reweighting) == len(all_messages_needs_reweighting)
                vectorizer = CountVectorizer(
                    binary=not needs_reweighting_wc,
                    analyzer="word", token_pattern="[^ ]+")
                count_vector_matrix = np.asarray(vectorizer.fit_transform(all_messages).toarray())
            for log, res in all_results:
                sim_dict = self._calculate_field_similarity(
                    log, res, log_field_ids, count_vector_matrix, needs_reweighting_wc, field)
                for key in sim_dict:
                    self.similarity_dict[field][key] = sim_dict[key]

    def reweight_words_weights_by_summing(self, count_vector_matrix):
        count_vector_matrix_weighted = np.zeros_like(count_vector_matrix, dtype=float)
        whole_sum_vector = np.sum(count_vector_matrix, axis=0)
        for i in range(len(count_vector_matrix)):
            for j in range(len(count_vector_matrix[i])):
                if whole_sum_vector[j] > 1 and count_vector_matrix[i][j] > 0:
                    count_vector_matrix_weighted[i][j] = max(0.1, 1 - whole_sum_vector[j] * 0.2)
                else:
                    count_vector_matrix_weighted[i][j] = count_vector_matrix[i][j]
        return count_vector_matrix_weighted

    def multiply_vectors_by_weight(self, rows, weights):
        return np.dot(np.reshape(weights, [-1]), rows)

    def normalize_weights(self, weights):
        normalized_weights = np.asarray(weights) / np.min(weights)
        return np.clip(normalized_weights, a_min=1.0, a_max=3.0)

    def _calculate_field_similarity(
            self, log, res, log_field_ids, count_vector_matrix, needs_reweighting_wc, field):
        all_results_similarity = {}
        for obj in res["hits"]["hits"]:
            group_id = (obj["_id"], log["_id"])
            index_query_message = log_field_ids[log["_id"]]
            index_log_message = log_field_ids[obj["_id"]]
            if (isinstance(index_query_message, int) and index_query_message < 0) and \
                    (isinstance(index_log_message, int) and index_log_message < 0):
                all_results_similarity[group_id] = {"similarity": 1.0, "both_empty": True}
            elif (isinstance(index_query_message, int) and index_query_message < 0) or \
                    (isinstance(index_log_message, int) and index_log_message < 0):
                all_results_similarity[group_id] = {"similarity": 0.0, "both_empty": False}
            else:
                query_vector = count_vector_matrix[index_query_message[0]:index_query_message[1] + 1]
                log_vector = count_vector_matrix[index_log_message[0]:index_log_message[1] + 1]
                if field == "namespaces_stacktrace":
                    query_vector = self.multiply_vectors_by_weight(
                        query_vector, self.normalize_weights(self.object_id_weights[log["_id"]]))
                    log_vector = self.multiply_vectors_by_weight(
                        log_vector, self.normalize_weights(self.object_id_weights[obj["_id"]]))
                else:
                    if needs_reweighting_wc:
                        query_vector = self.reweight_words_weights_by_summing(query_vector)
                        log_vector = self.reweight_words_weights_by_summing(log_vector)
                    query_vector = self.similarity_model.weigh_data_rows(query_vector)
                    log_vector = self.similarity_model.weigh_data_rows(log_vector)
                    if needs_reweighting_wc:
                        query_vector *= 2
                        log_vector *= 2
                similarity = round(1 - spatial.distance.cosine(query_vector, log_vector), 2)
                all_results_similarity[group_id] = {"similarity": similarity, "both_empty": False}

        return all_results_similarity
