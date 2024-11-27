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
from typing import Any, Optional

import numpy as np
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer

from app.machine_learning.models.weighted_similarity_calculator import WeightedSimilarityCalculator
from app.utils import text_processing

FIELDS_MAPPING_FOR_WEIGHTING = {
    'message_without_params_extended': ['detected_message_without_params_extended', 'stacktrace_extended'],
    'message_extended': ['detected_message_extended', 'stacktrace_extended']
}
ARTIFICIAL_COLUMNS = ['namespaces_stacktrace']


def multiply_vectors_by_weight(rows, weights):
    return np.dot(np.reshape(weights, [-1]), rows)


def normalize_weights(weights):
    normalized_weights = np.asarray(weights) / np.min(weights)
    return np.clip(normalized_weights, a_min=1.0, a_max=3.0)


class SimilarityCalculator:
    __similarity_dict: dict[str, dict]
    similarity_model: WeightedSimilarityCalculator
    config: dict[str, Any]
    object_id_weights: dict[str, list[float]]

    def __init__(self, config: dict[str, Any], similarity_model: WeightedSimilarityCalculator):
        self.similarity_model = similarity_model
        self.config = config
        self.__similarity_dict = {}
        self.object_id_weights = {}

    @staticmethod
    def reweight_words_weights_by_summing(count_vector_matrix):
        count_vector_matrix_weighted = np.zeros_like(count_vector_matrix, dtype=float)
        whole_sum_vector = np.sum(count_vector_matrix, axis=0)
        for i in range(len(count_vector_matrix)):
            for j in range(len(count_vector_matrix[i])):
                if whole_sum_vector[j] > 1 and count_vector_matrix[i][j] > 0:
                    count_vector_matrix_weighted[i][j] = max(0.1, 1 - whole_sum_vector[j] * 0.2)
                else:
                    count_vector_matrix_weighted[i][j] = count_vector_matrix[i][j]
        return count_vector_matrix_weighted

    def _calculate_field_similarity(
            self, log: dict, res: dict, log_field_ids: dict, count_vector_matrix: Optional[np.ndarray],
            needs_reweighting_wc: bool, field: str) -> dict[tuple[str, str], dict[str, Any]]:
        all_results_similarity = {}
        for obj in res["hits"]["hits"]:
            group_id = (str(obj["_id"]), str(log["_id"]))
            index_query_message = log_field_ids[log["_id"]]
            index_log_message = log_field_ids[obj["_id"]]
            if ((isinstance(index_query_message, int) and index_query_message < 0)
                    and (isinstance(index_log_message, int) and index_log_message < 0)):
                all_results_similarity[group_id] = {"similarity": 1.0, "both_empty": True}
            elif ((isinstance(index_query_message, int) and index_query_message < 0)
                  or (isinstance(index_log_message, int) and index_log_message < 0)):
                all_results_similarity[group_id] = {"similarity": 0.0, "both_empty": False}
            else:
                if count_vector_matrix is not None:
                    query_vector = count_vector_matrix[index_query_message[0]:index_query_message[1] + 1]
                    log_vector = count_vector_matrix[index_log_message[0]:index_log_message[1] + 1]
                    if field == "namespaces_stacktrace":
                        query_vector = multiply_vectors_by_weight(
                            query_vector, normalize_weights(self.object_id_weights[log["_id"]]))
                        log_vector = multiply_vectors_by_weight(
                            log_vector, normalize_weights(self.object_id_weights[obj["_id"]]))
                    else:
                        if needs_reweighting_wc:
                            query_vector = SimilarityCalculator.reweight_words_weights_by_summing(query_vector)
                            log_vector = SimilarityCalculator.reweight_words_weights_by_summing(log_vector)
                        query_vector = self.similarity_model.weigh_data_rows(query_vector)
                        log_vector = self.similarity_model.weigh_data_rows(log_vector)
                        if needs_reweighting_wc:
                            query_vector *= 2
                            log_vector *= 2
                    similarity = round(1 - spatial.distance.cosine(query_vector, log_vector), 2)
                    all_results_similarity[group_id] = {"similarity": similarity, "both_empty": False}
                else:
                    all_results_similarity[group_id] = {"similarity": 0.0, "both_empty": False}
        return all_results_similarity

    @staticmethod
    def _create_count_vector_matrix(all_messages: list[str],
                                    all_messages_needs_reweighting: list[int]) -> tuple[bool, Optional[np.ndarray]]:
        needs_reweighting_wc: bool = False
        count_vector_matrix: Optional[np.ndarray] = None
        if all_messages:
            needs_reweighting_wc = (all_messages_needs_reweighting
                                    and sum(all_messages_needs_reweighting) == len(all_messages_needs_reweighting))
            vectorizer = CountVectorizer(
                binary=not needs_reweighting_wc, analyzer="word", token_pattern="[^ ]+")
            try:
                count_vector_matrix = np.asarray(vectorizer.fit_transform(all_messages).toarray())
            except ValueError:
                # All messages are empty or contains only stop words
                pass
        return needs_reweighting_wc, count_vector_matrix

    def _calculate_similarity_for_all_results(
            self, all_results: list[tuple[dict[str, Any], dict[str, Any]]], log_field_ids: dict,
            count_vector_matrix: Optional[np.ndarray], needs_reweighting_wc: bool,
            field: str) -> dict[tuple[str, str], dict[str, Any]]:
        similarity = {}
        for log, res in all_results:
            sim_dict = self._calculate_field_similarity(
                log, res, log_field_ids, count_vector_matrix, needs_reweighting_wc, field)
            similarity.update(sim_dict)
        return similarity

    def _find_similarity_for_field(self, all_results: list[tuple[dict[str, Any], dict[str, Any]]],
                                   field: str) -> dict[tuple[str, str], dict[str, Any]]:
        log_field_ids: dict = {}
        index_in_message_array = 0
        all_messages: list[str] = []
        all_messages_needs_reweighting: list[int] = []
        for log, res in all_results:
            for obj in [log] + res["hits"]["hits"]:
                if obj["_id"] not in log_field_ids:
                    if field not in ARTIFICIAL_COLUMNS and (
                            field not in obj["_source"] or not obj["_source"][field].strip()):
                        log_field_ids[obj["_id"]] = -1
                    else:
                        needs_reweighting = 0
                        if (self.config["number_of_log_lines"] == -1
                                and field in FIELDS_MAPPING_FOR_WEIGHTING):
                            fields_to_use = FIELDS_MAPPING_FOR_WEIGHTING[field]
                            text = self.similarity_model.message_to_array(
                                obj["_source"][fields_to_use[0]], obj["_source"][fields_to_use[1]])
                        elif field == "namespaces_stacktrace":
                            gathered_lines = []
                            weights = []
                            for line in obj["_source"]["stacktrace"].split("\n"):
                                line_words = text_processing.split_words(
                                    line, min_word_length=self.config["min_word_length"])
                                for word in line_words:
                                    part_of_namespace = ".".join(word.split(".")[:2])
                                    if part_of_namespace in self.config["chosen_namespaces"]:
                                        gathered_lines.append(" ".join(line_words))
                                        weights.append(self.config["chosen_namespaces"][part_of_namespace])
                            if len(gathered_lines):
                                text = gathered_lines
                                self.object_id_weights[obj["_id"]] = weights
                            else:
                                text = []
                                for line in obj["_source"]["stacktrace"].split("\n"):
                                    text.append(" ".join(text_processing.split_words(
                                        line, min_word_length=self.config["min_word_length"])))
                                text = text_processing.filter_empty_lines(text)
                                self.object_id_weights[obj["_id"]] = [1] * len(text)
                        elif field.startswith("stacktrace"):
                            if text_processing.does_stacktrace_need_words_reweighting(obj["_source"][field]):
                                needs_reweighting = 1
                            text = self.similarity_model.message_to_array("", obj["_source"][field])
                        else:
                            text = [" ".join(
                                text_processing.split_words(
                                    obj["_source"][field], min_word_length=self.config["min_word_length"]))]
                        if not text:
                            log_field_ids[obj["_id"]] = -1
                        else:
                            all_messages.extend(text)
                            all_messages_needs_reweighting.append(needs_reweighting)
                            log_field_ids[obj["_id"]] = (index_in_message_array, len(all_messages) - 1)
                            index_in_message_array += len(text)

        needs_reweighting_wc, count_vector_matrix = self._create_count_vector_matrix(
            all_messages, all_messages_needs_reweighting)
        return self._calculate_similarity_for_all_results(
            all_results, log_field_ids, count_vector_matrix, needs_reweighting_wc, field)

    def find_similarity(self, all_results: list[tuple[dict[str, Any], dict[str, Any]]],
                        fields: list[str]) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
        for field in fields:
            if field in self.__similarity_dict:
                continue
            self.__similarity_dict[field] = self._find_similarity_for_field(all_results, field)
        return self.__similarity_dict
