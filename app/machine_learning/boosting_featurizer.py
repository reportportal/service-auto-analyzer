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

from collections import deque, defaultdict
from datetime import datetime
from typing import Optional, Any, Callable

import numpy as np

from app.commons import logging, similarity_calculator
from app.machine_learning.models import WeightedSimilarityCalculator
from app.machine_learning.models.defect_type_model import DATA_FIELD
from app.machine_learning.models.defect_type_model import DefectTypeModel
from app.utils import utils, text_processing

logger = logging.getLogger("analyzerApp.boosting_featurizer")


class BoostingFeaturizer:
    config: dict[str, Any]
    defect_type_predict_model: Optional[DefectTypeModel]
    scores_by_type: Optional[dict[str, dict[str, Any]]]
    feature_ids: list[int]
    feature_functions: dict[int, tuple[Callable, dict[str, Any], list[int]]]
    previously_gathered_features: dict[int, list[list[float]]]
    raw_results: list[tuple[dict[str, Any], dict[str, Any]]]
    all_results: list[tuple[dict[str, Any], list[dict[str, Any]]]]
    total_normalized_score: float

    def __init__(self, results: list[tuple[dict[str, Any], dict[str, Any]]], config: dict[str, Any],
                 feature_ids: str | list[int],
                 weighted_log_similarity_calculator: Optional[WeightedSimilarityCalculator] = None) -> None:
        self.config = config
        self.previously_gathered_features = {}
        self.similarity_calculator = similarity_calculator.SimilarityCalculator(
            self.config, similarity_model=weighted_log_similarity_calculator)
        if type(feature_ids) is str:
            self.feature_ids = text_processing.transform_string_feature_range_into_list(feature_ids)
        else:
            self.feature_ids = feature_ids
        self.fields_to_replace_with_merged_logs = [
            "message", "detected_message",
            "detected_message_without_params_extended",
            "message_without_params_extended",
            "message_extended",
            "detected_message_extended",
            "message_without_params_and_brackets",
            "detected_message_without_params_and_brackets"
        ]

        self.feature_functions = {
            0: (self._calculate_score, {}, []),
            1: (self._calculate_place, {}, []),
            3: (self._calculate_max_score_and_pos, {"return_val_name": "max_score_pos"}, []),
            5: (self._calculate_min_score_and_pos, {"return_val_name": "min_score_pos"}, []),
            7: (self._calculate_percent_count_items_and_mean, {"return_val_name": "cnt_items_percent"}, []),
            9: (self._calculate_percent_issue_types, {}, []),
            11: (self._calculate_similarity_percent, {"field_name": "message"}, []),
            12: (self.is_only_merged_small_logs, {}, []),
            13: (self._calculate_similarity_percent, {"field_name": "merged_small_logs"}, []),
            14: (self._has_test_item_several_logs, {}, []),
            15: (self._has_query_several_logs, {}, []),
            18: (self._calculate_similarity_percent, {"field_name": "detected_message"}, []),
            19: (self._calculate_similarity_percent, {"field_name": "detected_message_with_numbers"}, []),
            23: (self._calculate_similarity_percent, {"field_name": "stacktrace"}, []),
            25: (self._calculate_similarity_percent, {"field_name": "only_numbers"}, []),
            26: (self._calculate_max_score_and_pos, {"return_val_name": "max_score"}, []),
            27: (self._calculate_min_score_and_pos, {"return_val_name": "min_score"}, []),
            28: (self._calculate_percent_count_items_and_mean, {"return_val_name": "mean_score"}, []),
            29: (self._calculate_similarity_percent, {"field_name": "message_params"}, []),
            34: (self._calculate_similarity_percent, {"field_name": "found_exceptions"}, []),
            35: (self._is_all_log_lines, {}, []),
            36: (self._calculate_similarity_percent, {"field_name": "detected_message_extended"}, []),
            37: (self._calculate_similarity_percent, {"field_name": "detected_message_without_params_extended"}, []),
            38: (self._calculate_similarity_percent, {"field_name": "stacktrace_extended"}, []),
            40: (self._calculate_similarity_percent, {"field_name": "message_without_params_extended"}, []),
            41: (self._calculate_similarity_percent, {"field_name": "message_extended"}, []),
            42: (self.is_the_same_test_case, {}, []),
            43: (self.has_the_same_test_case_in_all_results, {}, []),
            48: (self.is_text_of_particular_defect_type, {"label_type": "ab"}, []),
            49: (self.is_text_of_particular_defect_type, {"label_type": "pb"}, []),
            50: (self.is_text_of_particular_defect_type, {"label_type": "si"}, []),
            51: (self.predict_particular_defect_type, {}, []),
            52: (self._calculate_similarity_percent, {"field_name": "namespaces_stacktrace"}, []),
            53: (self._calculate_similarity_percent, {"field_name": "detected_message_without_params_and_brackets"},
                 []),
            55: (self._calculate_similarity_percent, {"field_name": "potential_status_codes"}, []),
            56: (self.is_the_same_launch, {}, []),
            57: (self.is_the_same_launch_id, {}, []),
            59: (self._calculate_similarity_percent, {"field_name": "found_tests_and_methods"}, []),
            61: (self._calculate_similarity_percent, {"field_name": "test_item_name"}, []),
            64: (self._calculate_decay_function_score, {"field_name": "start_time"}, []),
            65: (self._calculate_test_item_logs_similar_percent, {}, []),
            66: (self._count_test_item_logs, {}, [])
        }

        fields_to_calc_similarity = self.find_columns_to_find_similarities_for()
        processed_results = self._perform_additional_text_processing(results)

        if "filter_min_should_match" in self.config and len(self.config["filter_min_should_match"]) > 0:
            self.similarity_calculator.find_similarity(
                processed_results, self.config["filter_min_should_match"] + ["merged_small_logs"])
            for field in self.config["filter_min_should_match"]:
                processed_results = self.filter_by_min_should_match(processed_results, field=field)
        if "filter_min_should_match_any" in self.config and len(self.config["filter_min_should_match_any"]) > 0:
            self.similarity_calculator.find_similarity(
                processed_results, self.config["filter_min_should_match_any"] + ["merged_small_logs"])
            processed_results = self.filter_by_min_should_match_any(
                processed_results, fields=self.config["filter_min_should_match_any"])
        self.test_item_log_stats = self._calculate_stats_by_test_item_ids(processed_results)
        if "filter_by_all_logs_should_be_similar" in self.config:
            if self.config["filter_by_all_logs_should_be_similar"]:
                processed_results = self.filter_by_all_logs_should_be_similar(processed_results)
        if "filter_by_test_case_hash" in self.config and self.config["filter_by_test_case_hash"]:
            processed_results = self.filter_by_test_case_hash(processed_results)
        if "calculate_similarities" not in self.config or self.config["calculate_similarities"]:
            self.similarity_calculator.find_similarity(processed_results, fields_to_calc_similarity)
        self.raw_results = processed_results
        self.total_normalized_score = 0.0
        self.all_results = self.normalize_results(processed_results)
        self.scores_by_type = None
        self.defect_type_predict_model = None
        self.used_model_info = set()
        self.features_to_recalculate_always = set([51, 58] + list(range(67, 74)))

    def find_most_relevant_by_type(self) -> dict[str, dict[str, Any]]:
        """Find most relevant log by issue type from OpenSearch query result.

        :return: dict with issue type as key and value as most relevant log and its metadata
        """
        if self.scores_by_type is not None:
            return self.scores_by_type

        scores_by_issue_type = defaultdict(lambda: {'mrHit': {'_score': -1}, 'score': 0})
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit['_source']['issue_type']
                hit['es_pos'] = idx

                issue_type_item = scores_by_issue_type[issue_type]
                if hit['_score'] > issue_type_item['mrHit']['_score']:
                    issue_type_item['mrHit'] = hit
                    issue_type_item['compared_log'] = log
                issue_type_item['score'] += (hit['normalized_score'] / self.total_normalized_score)
        self.scores_by_type = dict(scores_by_issue_type)
        return self.scores_by_type

    def _count_test_item_logs(self) -> dict[str, int]:
        """Count the number of requests (error logs) made to DB.

        Analyzer makes requests to DB to get the most relevant logs for the Test Item which it analyses. It takes into
        account each unique Error Log that is linked to the Item separately and makes a query to DB for each of them.
        This method counts the number of requests made to DB.
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        sim_logs_num_scores = {}
        for issue_type in scores_by_issue_type:
            sim_logs_num_scores[issue_type] = len(self.all_results)
        return sim_logs_num_scores

    @staticmethod
    def _calculate_stats_by_test_item_ids(
            all_results: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, float]:
        """Calculate relation between the number of logs found and queried for each Test Item.

        :param list[tuple[dict[str, Any], dict[str, Any]]] all_results: list of logs queried and their search results
        :return: dict with test item id as key and value as the relation between the number of logs found and queried
        """
        test_item_log_stats = defaultdict(lambda: 0)
        for _, res in all_results:
            for hit in res["hits"]["hits"]:
                test_item = hit["_source"]["test_item"]
                test_item_log_stats[test_item] += 1
        all_query_num = len(all_results)
        if all_query_num:
            for test_item_id in test_item_log_stats:
                test_item_log_stats[test_item_id] /= all_query_num
        return dict(test_item_log_stats)

    def _calculate_test_item_logs_similar_percent(self) -> dict[str, float]:
        """Calculate relation between the number of logs found and queried for each unique Issue Type.

        :return: dict with issue type as key and value as the relation between the number of logs found and queried
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        sim_logs_num_scores = {}
        for issue_type, search_rs in scores_by_issue_type.items():
            test_item_id = search_rs["mrHit"]["_source"]["test_item"]
            sim_logs_num_scores[issue_type] = 0.0
            if test_item_id in self.test_item_log_stats:
                sim_logs_num_scores[issue_type] = self.test_item_log_stats[test_item_id]
        return sim_logs_num_scores

    @staticmethod
    def _perform_additional_text_processing(all_results: list[tuple[dict[str, Any], dict[str, Any]]]):
        for log, res in all_results:
            for r in res["hits"]["hits"]:
                if "found_tests_and_methods" in r["_source"]:
                    r["_source"]["found_tests_and_methods"] = text_processing.preprocess_found_test_methods(
                        r["_source"]["found_tests_and_methods"])
        return all_results

    def _calculate_decay_function_score(self, field_name: str) -> dict[str, float]:
        """Calculate the decay function score.

        The function exponentially decrease float value from 1 to 0 depending on time passed from result log to log we
        analyse.

        :param str field_name: field name to compare, usually 'start_time'
        :return: dict with issue type as key and value as the decay function float score from 1 to 0.
        """
        decay_speed = np.log(self.config["time_weight_decay"])
        scores_by_issue_type = self.find_most_relevant_by_type()
        dates_by_issue_types = {}
        for issue_type, search_rs in scores_by_issue_type.items():
            field_date_str = search_rs["mrHit"]["_source"][field_name]
            field_date = datetime.strptime(field_date_str, '%Y-%m-%d %H:%M:%S')
            compared_field_date_str = search_rs["compared_log"]["_source"][field_name]
            compared_field_date = datetime.strptime(compared_field_date_str, '%Y-%m-%d %H:%M:%S')
            if compared_field_date < field_date:
                field_date, compared_field_date = compared_field_date, field_date
            dates_by_issue_types[issue_type] = np.exp(decay_speed * (compared_field_date - field_date).days / 7)
        return dates_by_issue_types

    def fill_previously_gathered_features(self, feature_list: list[list[float]], feature_ids: list[int]) -> None:
        self.previously_gathered_features = utils.fill_previously_gathered_features(feature_list, feature_ids)

    def set_defect_type_model(self, defect_type_model: DefectTypeModel):
        self.defect_type_predict_model = defect_type_model

    def get_used_model_info(self):
        return list(self.used_model_info)

    def predict_particular_defect_type(self) -> dict[str, float]:
        """Predict the probability of the most relevant log to be of a certain defect type.

        The feature uses Defect Type Model to predict the probability of the most relevant log to be of a certain
        defect type.

        :return: dict with issue type as key and value as the probability of the most relevant log to be of a certain
                 defect type
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        result = {}
        for issue_type, search_rs in scores_by_issue_type.items():
            compared_log = search_rs["compared_log"]
            det_message = compared_log["_source"][DATA_FIELD]
            mr_hit = search_rs["mrHit"]
            issue_type_to_compare: str = mr_hit["_source"]["issue_type"].lower()
            try:
                if issue_type_to_compare.startswith('nd') or issue_type_to_compare.startswith('ti'):
                    continue
                res, res_prob = self.defect_type_predict_model.predict([det_message], issue_type_to_compare)
                result[issue_type] = res_prob[0][1] if len(res_prob[0]) == 2 else 0.0
                self.used_model_info.update(self.defect_type_predict_model.get_model_info())
            except Exception as err:
                logger.exception(err)
        return result

    def is_text_of_particular_defect_type(self, label_type: str) -> dict[str, int]:
        """Check if the most relevant search results contain certain type of defect.

        :param str label_type: type of defect to check
        :return: dict with issue type as key and value as 1 if the most relevant search results contain certain type of
                 defect, 0 otherwise
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        issue_type_stats = {}
        for issue_type, search_rs in scores_by_issue_type.items():
            mr_hit = search_rs["mrHit"]
            rel_item_issue_type = mr_hit["_source"]["issue_type"]
            issue_type_stats[issue_type] = int(label_type == rel_item_issue_type.lower()[:2])
        return issue_type_stats

    def filter_by_all_logs_should_be_similar(
            self, all_results: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        new_results = []
        for log, res in all_results:
            new_elastic_res = []
            for r in res["hits"]["hits"]:
                if r["_source"]["test_item"] in self.test_item_log_stats:
                    if self.test_item_log_stats[r["_source"]["test_item"]] > 0.99:
                        new_elastic_res.append(r)
            new_results.append((log, {"hits": {"hits": new_elastic_res}}))
        return new_results

    @staticmethod
    def filter_by_test_case_hash(
            all_results: list[tuple[dict[str, Any], dict[str, Any]]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        new_results = []
        for log, res in all_results:
            test_case_hash_dict = {}
            for r in res["hits"]["hits"]:
                test_case_hash = r["_source"]["test_case_hash"]
                if test_case_hash not in test_case_hash_dict:
                    test_case_hash_dict[test_case_hash] = []
                test_case_hash_dict[test_case_hash].append(
                    (r["_id"], int(r["_score"]), datetime.strptime(
                        r["_source"]["start_time"], '%Y-%m-%d %H:%M:%S')))
            log_ids_to_take = set()
            for test_case_hash in test_case_hash_dict:
                test_case_hash_dict[test_case_hash] = sorted(
                    test_case_hash_dict[test_case_hash],
                    key=lambda x: (x[1], x[2]),
                    reverse=True)
                scores_used = set()
                for sorted_score in test_case_hash_dict[test_case_hash]:
                    if sorted_score[1] not in scores_used:
                        log_ids_to_take.add(sorted_score[0])
                        scores_used.add(sorted_score[1])
            new_elastic_res = []
            for elastic_res in res["hits"]["hits"]:
                if elastic_res["_id"] in log_ids_to_take:
                    new_elastic_res.append(elastic_res)
            new_results.append((log, {"hits": {"hits": new_elastic_res}}))
        return new_results

    def is_the_same_field(self, field_name: str) -> dict[str, int]:
        """Check if the query log and search results contain field and its values are equal.

        :param str field_name: field name to compare
        :return: dict with issue type as key and value as 1 if fields are equal, 0 otherwise
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        num_of_logs_issue_type = {}
        for issue_type, search_rs in scores_by_issue_type.items():
            rel_item_value = search_rs["mrHit"]["_source"][field_name]
            queried_item_value = search_rs["compared_log"]["_source"][field_name]

            if rel_item_value is None and queried_item_value is None:
                num_of_logs_issue_type[issue_type] = 0
                continue

            if type(queried_item_value) is str:
                queried_item_value = queried_item_value.strip().lower()
            if type(rel_item_value) is str:
                rel_item_value = rel_item_value.strip().lower()

            if rel_item_value == '' and queried_item_value == '':
                num_of_logs_issue_type[issue_type] = 0
                continue

            num_of_logs_issue_type[issue_type] = int(rel_item_value == queried_item_value)
        return num_of_logs_issue_type

    def is_the_same_test_case(self) -> dict[str, int]:
        """Check if the query log and search results contain the same Test Case Hash.

        :return: dict with issue type as key and value as 1 if Test Case Hashes are equal, 0 otherwise
        """
        return self.is_the_same_field('test_case_hash')

    def is_the_same_launch(self) -> dict[str, int]:
        """Check if the query log and search results contain the same Launch Name.

        :return: dict with issue type as key and value as 1 if Launch Names are equal, 0 otherwise
        """
        return self.is_the_same_field('launch_name')

    def is_the_same_launch_id(self) -> dict[str, int]:
        """Check if the query log and search results contain the same Launch ID.

        :return: dict with issue type as key and value as 1 if Launch IDs are equal, 0 otherwise
        """
        return self.is_the_same_field('launch_id')

    def has_the_same_test_case_in_all_results(self) -> dict[str, int]:
        """Check if the query log and search results contain the same Test Case Hash in any of all results.

        :return: dict with issue type as key and value as 1 if Test Case Hashes are equal in any of all results,
                 0 otherwise
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        num_of_logs_issue_type = {}
        has_the_same_test_case = 0
        for search_rs in scores_by_issue_type.values():
            rel_item_test_case_hash = search_rs["mrHit"]["_source"]["test_case_hash"]
            queried_item_test_case_hash = search_rs["compared_log"]["_source"]["test_case_hash"]
            if not rel_item_test_case_hash:
                continue
            if rel_item_test_case_hash == queried_item_test_case_hash:
                has_the_same_test_case = 1
                break
        for issue_type in scores_by_issue_type:
            num_of_logs_issue_type[issue_type] = has_the_same_test_case
        return num_of_logs_issue_type

    def find_columns_to_find_similarities_for(self) -> list[str]:
        fields_to_calc_similarity = set()
        for feature in self.feature_ids:
            method_params = self.feature_functions[feature]
            if 'field_name' in method_params[1]:
                fields_to_calc_similarity.add(method_params[1]['field_name'])
        return list(fields_to_calc_similarity)

    def _calculate_score(self) -> dict[str, float]:
        """Calculate Score for every unique Issue Type from OpenSearch query result, normalized by maximum score in the
        result.

        :return: dict with issue type as key and value as normalized score
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        return {item: search_rs['score'] for item, search_rs in scores_by_issue_type.items()}

    def _is_all_log_lines(self) -> dict[str, int]:
        """Return if all log lines were used to find the most relevant log.

        :return: dict with issue type as key and value as 1 if all log lines were used, 0 otherwise
        """
        scores_by_issue_type = self._calculate_score()
        num_of_logs_issue_type = {}
        for issue_type in scores_by_issue_type:
            num_of_logs_issue_type[issue_type] = int(self.config["number_of_log_lines"] == -1)
        return num_of_logs_issue_type

    def is_only_merged_small_logs(self) -> dict[str, int]:
        """Check if the query log and search results contain only merged small logs.

        :return: dict with issue type as key and value as 0 if both logs contain only merged small logs, 1 otherwise
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        similarity_percent_by_type = {}
        for issue_type, search_rs in scores_by_issue_type.items():
            group_id = (search_rs["mrHit"]["_id"], search_rs["compared_log"]["_id"])
            sim_obj = self.similarity_calculator.similarity_dict["message"][group_id]
            similarity_percent_by_type[issue_type] = int(sim_obj["both_empty"])
        return similarity_percent_by_type

    def filter_by_min_should_match(self, all_results: list[tuple[dict[str, Any], dict[str, Any]]], field="message"):
        new_results = []
        for log, res in all_results:
            new_elastic_res = []
            for elastic_res in res["hits"]["hits"]:
                group_id = (elastic_res["_id"], log["_id"])
                sim_obj = self.similarity_calculator.similarity_dict[field][group_id]
                similarity = sim_obj["similarity"]
                if sim_obj["both_empty"] and field in self.fields_to_replace_with_merged_logs:
                    sim_obj = self.similarity_calculator.similarity_dict["merged_small_logs"]
                    similarity = sim_obj[group_id]["similarity"]
                if similarity >= self.config["min_should_match"]:
                    new_elastic_res.append(elastic_res)
            new_results.append((log, {"hits": {"hits": new_elastic_res}}))
        return new_results

    def filter_by_min_should_match_any(
            self, all_results: list[tuple[dict[str, Any], dict[str, Any]]], fields: list[str]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        if not fields:
            return all_results
        new_results = []
        for log, res in all_results:
            new_elastic_res = []
            for elastic_res in res["hits"]["hits"]:
                group_id = (elastic_res["_id"], log["_id"])
                max_similarity = 0.0
                for field in fields:
                    sim_obj = self.similarity_calculator.similarity_dict[field][group_id]
                    similarity = sim_obj["similarity"]
                    if sim_obj["both_empty"] and field in self.fields_to_replace_with_merged_logs:
                        sim_obj = self.similarity_calculator.similarity_dict["merged_small_logs"]
                        similarity = sim_obj[group_id]["similarity"]
                    max_similarity = max(max_similarity, similarity)
                if max_similarity >= self.config["min_should_match"]:
                    new_elastic_res.append(elastic_res)
            new_results.append((log, {"hits": {"hits": new_elastic_res}}))
        return new_results

    def _calculate_percent_issue_types(self) -> dict[str, float]:
        """Calculate weight of every unique Issue Type among all unique Issue Types.

        :return: dict with issue type as key and float value as weight
        """
        scores_by_issue_type = self._calculate_score()
        percent_by_issue_type = {}
        for issue_type in scores_by_issue_type:
            percent_by_issue_type[issue_type] = 1 / len(scores_by_issue_type) if len(scores_by_issue_type) else 0
        return percent_by_issue_type

    def _has_test_item_several_logs(self) -> dict[str, int]:
        """Calculate if each of the most relevant results' Test Item has several small logs which were merged.

        :return: dict with issue type as key and value as 0 if Test Item has several small logs which were merged,
                 1 otherwise.
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        has_several_logs_by_type = {}
        for issue_type, search_rs in scores_by_issue_type.items():
            merged_small_logs = search_rs["mrHit"]["_source"]["merged_small_logs"]
            has_several_logs_by_type[issue_type] = int(merged_small_logs.strip() != "")
        return has_several_logs_by_type

    def _has_query_several_logs(self) -> dict[str, int]:
        """Calculate if request Test Item has several small logs which were merged.

        :return: dict with issue type as key and value as 0 if request Test Item has several small logs which were
                 merged, 1 otherwise.
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        has_several_logs_by_type = {}
        for issue_type, search_rs in scores_by_issue_type.items():
            merged_small_logs = search_rs["compared_log"]["_source"]["merged_small_logs"]
            has_several_logs_by_type[issue_type] = int(merged_small_logs.strip() != "")
        return has_several_logs_by_type

    def _calculate_place(self) -> dict[str, float]:
        """
        Calculate Inverse order for every unique Issue Type as it returned in OpenSearch result.

        :return: dict with issue type as key and value as inverse order
        """
        scores_by_issue_type = self._calculate_score()
        place_by_issue_type = {}
        for idx, issue_type_item in enumerate(sorted(scores_by_issue_type.items(), key=lambda x: x[1], reverse=True)):
            place_by_issue_type[issue_type_item[0]] = 1 / (1 + idx)
        return place_by_issue_type

    def _calculate_max_score_and_pos(self, return_val_name: str = 'max_score') -> dict[str, float]:
        """Calculate maximum Entry score and Inverse order for every issue type in query result.

        :param str return_val_name: name of return value, can be 'max_score' or 'max_score_pos'
        :return: dict with issue type as key and value as maximum score or inverse order of this score
        """
        max_scores_by_issue_type = {}
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if issue_type not in max_scores_by_issue_type \
                        or hit["normalized_score"] > max_scores_by_issue_type[issue_type]["max_score"]:
                    max_scores_by_issue_type[issue_type] = {"max_score": hit["normalized_score"],
                                                            "max_score_pos": 1 / (1 + idx), }
        return {item: results[return_val_name] for item, results in max_scores_by_issue_type.items()}

    def _calculate_min_score_and_pos(self, return_val_name: str = 'min_score') -> dict[str, float]:
        """Calculate minimum Entry score and Inverse order for every issue type in query result.

        :param str return_val_name: name of return value, can be 'min_score' or 'min_score_pos'
        :return: dict with issue type as key and value as minimum score or inverse order of this score
        """
        min_scores_by_issue_type = {}
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if issue_type not in min_scores_by_issue_type \
                        or hit["normalized_score"] < min_scores_by_issue_type[issue_type]["min_score"]:
                    min_scores_by_issue_type[issue_type] = {"min_score": hit["normalized_score"],
                                                            "min_score_pos": 1 / (1 + idx), }
        return {item: results[return_val_name] for item, results in min_scores_by_issue_type.items()}

    def _calculate_percent_count_items_and_mean(self, return_val_name: str = 'mean_score') -> dict[str, float]:
        """Calculate percent of items by issue type and mean score of this issue type.

        :param str return_val_name: name of return value, can be 'mean_score' or 'cnt_items_percent'
        :return: dict with issue type as key and value as mean score or percent of items
        """
        cnt_items_by_issue_type: dict[str, dict[str: int]] = defaultdict(lambda: defaultdict(lambda: 0))
        cnt_items_glob = 0
        for log, es_results in self.all_results:
            cnt_items_glob += len(es_results)

            for idx, hit in enumerate(es_results):
                issue_type = hit['_source']['issue_type']
                cnt_items_by_issue_type[issue_type]['cnt_items_percent'] += 1
                cnt_items_by_issue_type[issue_type]['mean_score'] += hit['normalized_score']

        for issue_scores in cnt_items_by_issue_type.values():
            issue_scores['mean_score'] /= issue_scores['cnt_items_percent']
            issue_scores['cnt_items_percent'] /= cnt_items_glob
        return {item: results[return_val_name] for item, results in cnt_items_by_issue_type.items()}

    def normalize_results(
            self, all_elastic_results: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        all_results = []
        max_score = 0
        self.total_normalized_score = 0.0
        for query_log, es_results in all_elastic_results:
            for hit in es_results["hits"]["hits"]:
                max_score = max(max_score, hit["_score"])
        for query_log, es_results in all_elastic_results:
            for hit in es_results["hits"]["hits"]:
                hit["normalized_score"] = hit["_score"] / max_score
                self.total_normalized_score += hit["normalized_score"]
            all_results.append((query_log, es_results["hits"]["hits"]))
        return all_results

    def _calculate_similarity_percent(self, field_name="message") -> dict[str, float]:
        """Calculate similarity percent by specified filed for every unique Issue Type from OpenSearch query result.

        This method calculates cosine similarity by specified filed by vectors from CountVectorizer of sklearn library
        under the hood. Text lines are reweighed by the WeightedSimilarityCalculator model.

        :param str field_name: name of field to calculate similarity
        :return: dict with issue type as key and float value as similarity percent
        """
        scores_by_issue_type = self.find_most_relevant_by_type()
        self.similarity_calculator.find_similarity(self.raw_results, [field_name])
        similarity_percent_by_type = {}
        for issue_type, search_rs in scores_by_issue_type.items():
            group_id = (search_rs["mrHit"]["_id"], search_rs["compared_log"]["_id"])
            sim_obj = self.similarity_calculator.similarity_dict[field_name][group_id]
            similarity_percent_by_type[issue_type] = sim_obj["similarity"]
        return similarity_percent_by_type

    def get_ordered_features_to_process(self) -> list[int]:
        feature_graph: dict[int, list[int]] = {}
        features_queue = deque(self.feature_ids.copy())
        while features_queue:
            cur_feature = features_queue.popleft()
            if cur_feature in feature_graph:
                continue
            _, _, dependants = self.feature_functions[cur_feature]
            feature_graph[cur_feature] = dependants
            features_queue.extend(dependants)
        ordered_features = utils.topological_sort(feature_graph)
        return ordered_features

    def gather_features_info(self) -> tuple[list[list[float]], list[str]]:
        """Gather all features from feature_ids for a test item"""
        gathered_data = []
        gathered_data_dict: dict[int, list[list[float]]] = {}
        issue_type_names: list[str] = []
        issue_type_by_index: dict[int, str] = {}
        try:
            scores_by_types = self.find_most_relevant_by_type()
            for idx, issue_type in enumerate(scores_by_types):
                issue_type_by_index[idx] = issue_type
                issue_type_names.append(issue_type)

            for feature in self.get_ordered_features_to_process():
                if feature in self.previously_gathered_features and feature not in self.features_to_recalculate_always:
                    gathered_data_dict[feature] = self.previously_gathered_features[feature]
                else:
                    func, args, _ = self.feature_functions[feature]
                    result = func(**args)
                    if type(result) is list:
                        gathered_data_dict[feature] = result
                    else:
                        gathered_data_dict[feature] = []
                        for idx in sorted(issue_type_by_index.keys()):
                            issue_type = issue_type_by_index[idx]
                            try:
                                _ = result[issue_type][0]
                                gathered_data_dict[feature].append(result[issue_type])
                            except:  # noqa
                                gathered_data_dict[feature].append([round(result[issue_type], 2)])
                    self.previously_gathered_features[feature] = gathered_data_dict[feature]
            gathered_data = utils.gather_feature_list(gathered_data_dict, self.feature_ids)
        except Exception as err:
            logger.error("Errors in boosting features calculation")
            logger.exception(err)
        return gathered_data, issue_type_names
