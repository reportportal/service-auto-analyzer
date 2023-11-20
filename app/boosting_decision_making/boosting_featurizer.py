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
from collections import deque
from datetime import datetime

import numpy as np

from app.boosting_decision_making.boosting_decision_maker import BoostingDecisionMaker
from app.commons import similarity_calculator
from app.utils import utils, text_processing

logger = logging.getLogger("analyzerApp.boosting_featurizer")


class BoostingFeaturizer:

    def __init__(self, all_results, config, feature_ids, weighted_log_similarity_calculator=None,
                 features_dict_with_saved_objects=None):
        self.config = config
        self.previously_gathered_features = {}
        self.models = {}
        self.features_dict_with_saved_objects = {}
        if features_dict_with_saved_objects is not None:
            self.features_dict_with_saved_objects = features_dict_with_saved_objects
        self.similarity_calculator = similarity_calculator.SimilarityCalculator(
            self.config,
            weighted_similarity_calculator=weighted_log_similarity_calculator)
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
            "detected_message_without_params_and_brackets"]

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
            28: (self._calculate_percent_count_items_and_mean,
                 {"return_val_name": "mean_score"}, []),
            29: (self._calculate_similarity_percent, {"field_name": "message_params"}, []),
            34: (self._calculate_similarity_percent, {"field_name": "found_exceptions"}, []),
            35: (self._is_all_log_lines, {}, []),
            36: (self._calculate_similarity_percent, {"field_name": "detected_message_extended"}, []),
            37: (self._calculate_similarity_percent,
                 {"field_name": "detected_message_without_params_extended"}, []),
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
            53: (self._calculate_similarity_percent,
                 {"field_name": "detected_message_without_params_and_brackets"}, []),
            55: (self._calculate_similarity_percent,
                 {"field_name": "potential_status_codes"}, []),
            56: (self.is_the_same_launch, {}, []),
            57: (self.is_the_same_launch_id, {}, []),
            58: (self._calculate_model_probability,
                 {"model_folder": self.config["boosting_model"]},
                 self.get_necessary_features(self.config["boosting_model"])),
            59: (self._calculate_similarity_percent, {"field_name": "found_tests_and_methods"}, []),
            61: (self._calculate_similarity_percent, {"field_name": "test_item_name"}, []),
            64: (self._calculate_decay_function_score, {"field_name": "start_time"}, []),
            65: (self._calculate_test_item_logs_similar_percent, {}, []),
            66: (self._count_test_item_logs, {}, []),
            67: (self._encode_into_vector,
                 {"field_name": "launch_name", "feature_name": 67, "only_query": True}, []),
            68: (self._encode_into_vector,
                 {"field_name": "detected_message", "feature_name": 68, "only_query": False}, []),
            69: (self._encode_into_vector,
                 {"field_name": "stacktrace", "feature_name": 69, "only_query": False}, []),
            70: (self._encode_into_vector,
                 {"field_name": "launch_name", "feature_name": 70, "only_query": True}, []),
            71: (self._encode_into_vector,
                 {"field_name": "test_item_name", "feature_name": 71, "only_query": False}, []),
            72: (self._encode_into_vector,
                 {"field_name": "found_exceptions", "feature_name": 72, "only_query": True}, [])
        }

        fields_to_calc_similarity = self.find_columns_to_find_similarities_for()
        all_results = self._perform_additional_text_processing(all_results)

        if "filter_min_should_match" in self.config and len(self.config["filter_min_should_match"]) > 0:
            self.similarity_calculator.find_similarity(
                all_results,
                self.config["filter_min_should_match"] + ["merged_small_logs"])
            for field in self.config["filter_min_should_match"]:
                all_results = self.filter_by_min_should_match(all_results, field=field)
        if "filter_min_should_match_any" in self.config and \
                len(self.config["filter_min_should_match_any"]) > 0:
            self.similarity_calculator.find_similarity(
                all_results,
                self.config["filter_min_should_match_any"] + ["merged_small_logs"])
            all_results = self.filter_by_min_should_match_any(
                all_results,
                fields=self.config["filter_min_should_match_any"])
        self.test_item_log_stats = self._calculate_stats_by_test_item_ids(all_results)
        if "filter_by_all_logs_should_be_similar" in self.config:
            if self.config["filter_by_all_logs_should_be_similar"]:
                all_results = self.filter_by_all_logs_should_be_similar(all_results)
        if "filter_by_test_case_hash" in self.config \
                and self.config["filter_by_test_case_hash"]:
            all_results = self.filter_by_test_case_hash(all_results)
        if "calculate_similarities" not in self.config or self.config["calculate_similarities"]:
            self.similarity_calculator.find_similarity(
                all_results,
                fields_to_calc_similarity)
        self.raw_results = all_results
        self.all_results = self.normalize_results(all_results)
        self.scores_by_issue_type = None
        self.defect_type_predict_model = None
        self.used_model_info = set()
        self.features_to_recalculate_always = set([51, 58] + list(range(67, 74)))

    def _count_test_item_logs(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        sim_logs_num_scores = {}
        for issue_type in scores_by_issue_type:
            sim_logs_num_scores[issue_type] = len(self.all_results)
        return sim_logs_num_scores

    def _calculate_test_item_logs_similar_percent(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        sim_logs_num_scores = {}
        for issue_type in scores_by_issue_type:
            test_item_id = scores_by_issue_type[issue_type]["mrHit"]["_source"]["test_item"]
            sim_logs_num_scores[issue_type] = 0.0
            if test_item_id in self.test_item_log_stats:
                sim_logs_num_scores[issue_type] = self.test_item_log_stats[test_item_id]
        return sim_logs_num_scores

    def _calculate_stats_by_test_item_ids(self, all_results):
        test_item_log_stats = {}
        for log, res in all_results:
            for r in res["hits"]["hits"]:
                if r["_source"]["test_item"] not in test_item_log_stats:
                    test_item_log_stats[r["_source"]["test_item"]] = 0
                test_item_log_stats[r["_source"]["test_item"]] += 1
        all_logs = len(all_results)
        if all_logs:
            for test_item_id in test_item_log_stats:
                test_item_log_stats[test_item_id] /= all_logs
        return test_item_log_stats

    def _perform_additional_text_processing(self, all_results):
        for log, res in all_results:
            for r in res["hits"]["hits"]:
                if "found_tests_and_methods" in r["_source"]:
                    r["_source"]["found_tests_and_methods"] = text_processing.preprocess_found_test_methods(
                        r["_source"]["found_tests_and_methods"])
        return all_results

    def _calculate_decay_function_score(self, field_name):
        scores_by_issue_type = self.find_most_relevant_by_type()
        dates_by_issue_types = {}
        for issue_type in scores_by_issue_type:
            field_date = scores_by_issue_type[issue_type]["mrHit"]["_source"][field_name]
            field_date = datetime.strptime(field_date, '%Y-%m-%d %H:%M:%S')
            compared_field_date = scores_by_issue_type[issue_type]["compared_log"]["_source"][field_name]
            compared_field_date = datetime.strptime(compared_field_date, '%Y-%m-%d %H:%M:%S')
            if compared_field_date < field_date:
                field_date, compared_field_date = compared_field_date, field_date
            dates_by_issue_types[issue_type] = np.exp(
                np.log(self.config["time_weight_decay"]) * (compared_field_date - field_date).days / 7)
        return dates_by_issue_types

    def _encode_into_vector(self, field_name, feature_name, only_query):
        if feature_name not in self.features_dict_with_saved_objects:
            logger.error(self.features_dict_with_saved_objects)
            logger.error("Feature '%s' has no encoder" % feature_name)
            return []
        if field_name != self.features_dict_with_saved_objects[feature_name].field_name:
            logger.error(field_name)
            logger.error("Field name '%s' is not the same as in the settings '%s'" % (
                field_name, self.features_dict_with_saved_objects[feature_name].field_name))
            return []
        scores_by_issue_type = self.find_most_relevant_by_type()
        encodings_by_issue_type = {}
        issue_types, gathered_data = [], []
        for issue_type in scores_by_issue_type:
            field_data = scores_by_issue_type[issue_type]["compared_log"]["_source"][field_name]
            issue_types.append(issue_type)
            gathered_data.append(field_data)
            if not only_query:
                gathered_data.append(
                    scores_by_issue_type[issue_type]["mrHit"]["_source"][field_name])
        if gathered_data:
            encoded_data = self.features_dict_with_saved_objects[feature_name].transform(
                gathered_data).toarray()
            encoded_data[encoded_data != 0.0] = 1.0
            for idx in range(len(issue_types)):
                if only_query:
                    encodings_by_issue_type[issue_types[idx]] = list(encoded_data[idx])
                else:
                    encodings_by_issue_type[issue_types[idx]] = list(
                        (encoded_data[2 * idx] + encoded_data[2 * idx + 1]) / 2)
        return encodings_by_issue_type

    def _calculate_model_probability(self, model_folder=""):
        if not model_folder.strip():
            return []
        if model_folder not in self.models:
            logger.error("Model folder is not found: '%s'", model_folder)
            return []
        feature_ids = self.models[model_folder].get_feature_ids()
        feature_data = utils.gather_feature_list(self.previously_gathered_features, feature_ids, to_list=True)
        predicted_labels, predicted_labels_probability = self.models[model_folder].predict(
            feature_data)
        predicted_probability = []
        for res in predicted_labels_probability:
            predicted_probability.append(float(res[1]))
        return [[round(r, 2)] for r in predicted_probability]

    def get_necessary_features(self, model_folder):
        if not model_folder.strip():
            return []
        if model_folder not in self.models:
            try:
                self.models[model_folder] = BoostingDecisionMaker(folder=model_folder)
                return self.models[model_folder].get_feature_ids()
            except Exception as err:
                logger.debug(err)
                return []
        return self.models[model_folder].get_feature_ids()

    def fill_prevously_gathered_features(self, feature_list, feature_ids):
        self.previously_gathered_features = utils.fill_prevously_gathered_features(
            feature_list, feature_ids)

    def get_used_model_info(self):
        return list(self.used_model_info)

    def set_defect_type_model(self, defect_type_model):
        self.defect_type_predict_model = defect_type_model

    def predict_particular_defect_type(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        result = {}
        for issue_type in scores_by_issue_type:
            compared_log = scores_by_issue_type[issue_type]["compared_log"]
            det_message = compared_log["_source"]["detected_message_without_params_extended"]
            mr_hit = scores_by_issue_type[issue_type]["mrHit"]
            issue_type_to_compare = mr_hit["_source"]["issue_type"]
            det_message = text_processing.clean_from_brackets(det_message)
            result[issue_type] = 0.0
            try:
                model_to_use = issue_type_to_compare.lower()[:2]
                if model_to_use in ["nd", "ti"]:
                    continue
                if issue_type_to_compare in self.defect_type_predict_model.models:
                    model_to_use = issue_type_to_compare
                res, res_prob = self.defect_type_predict_model.predict(
                    [det_message], model_to_use)
                result[issue_type] = res_prob[0][1] if len(res_prob[0]) == 2 else 0.0
                self.used_model_info.update(self.defect_type_predict_model.get_model_info())
            except Exception as err:
                logger.error(err)
        return result

    def is_text_of_particular_defect_type(self, label_type):
        scores_by_issue_type = self.find_most_relevant_by_type()
        issue_type_stats = {}
        for issue_type in scores_by_issue_type:
            mr_hit = scores_by_issue_type[issue_type]["mrHit"]
            rel_item_issue_type = mr_hit["_source"]["issue_type"]
            issue_type_stats[issue_type] = int(label_type == rel_item_issue_type.lower()[:2])
        return issue_type_stats

    def filter_by_all_logs_should_be_similar(self, all_results):
        new_results = []
        for log, res in all_results:
            new_elastic_res = []
            for r in res["hits"]["hits"]:
                if r["_source"]["test_item"] in self.test_item_log_stats:
                    if self.test_item_log_stats[r["_source"]["test_item"]] > 0.99:
                        new_elastic_res.append(r)
            new_results.append((log, {"hits": {"hits": new_elastic_res}}))
        return new_results

    def filter_by_test_case_hash(self, all_results):
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
        scores_by_issue_type = self.find_most_relevant_by_type()
        num_of_logs_issue_type = {}
        for issue_type in scores_by_issue_type:
            rel_item_value = scores_by_issue_type[issue_type]["mrHit"]["_source"][field_name]
            queried_item_value = scores_by_issue_type[issue_type]["compared_log"]["_source"][field_name]

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
        return self.is_the_same_field('test_case_hash')

    def is_the_same_launch(self) -> dict[str, int]:
        return self.is_the_same_field('launch_name')

    def is_the_same_launch_id(self) -> dict[str, int]:
        return self.is_the_same_field('launch_id')

    def has_the_same_test_case_in_all_results(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        num_of_logs_issue_type = {}
        has_the_same_test_case = 0
        for issue_type in scores_by_issue_type:
            rel_item_test_case_hash = scores_by_issue_type[issue_type]["mrHit"]["_source"]["test_case_hash"]
            queried_item_test_case_hash = \
                scores_by_issue_type[issue_type]["compared_log"]["_source"]["test_case_hash"]
            if not rel_item_test_case_hash:
                continue
            if rel_item_test_case_hash == queried_item_test_case_hash:
                has_the_same_test_case = 1
                break
        for issue_type in scores_by_issue_type:
            num_of_logs_issue_type[issue_type] = has_the_same_test_case
        return num_of_logs_issue_type

    def find_columns_to_find_similarities_for(self):
        fields_to_calc_similarity = set()
        for feature in self.feature_ids:
            method_params = self.feature_functions[feature]
            if "field_name" in method_params[1]:
                fields_to_calc_similarity.add(method_params[1]["field_name"])
        return list(fields_to_calc_similarity)

    def _is_all_log_lines(self):
        scores_by_issue_type = self._calculate_score()
        num_of_logs_issue_type = {}
        for issue_type in scores_by_issue_type:
            num_of_logs_issue_type[issue_type] = int(self.config["number_of_log_lines"] == -1)
        return num_of_logs_issue_type

    def is_only_merged_small_logs(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        similarity_percent_by_type = {}
        for issue_type in scores_by_issue_type:
            group_id = (scores_by_issue_type[issue_type]["mrHit"]["_id"],
                        scores_by_issue_type[issue_type]["compared_log"]["_id"])
            sim_obj = self.similarity_calculator.similarity_dict["message"][group_id]
            similarity_percent_by_type[issue_type] = int(sim_obj["both_empty"])
        return similarity_percent_by_type

    def filter_by_min_should_match(self, all_results, field="message"):
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

    def filter_by_min_should_match_any(self, all_results, fields=["detected_message"]):
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

    def _calculate_percent_issue_types(self):
        scores_by_issue_type = self._calculate_score()
        percent_by_issue_type = {}
        for issue_type in scores_by_issue_type:
            percent_by_issue_type[issue_type] = 1 / len(scores_by_issue_type) \
                if len(scores_by_issue_type) else 0
        return percent_by_issue_type

    def _has_test_item_several_logs(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        has_several_logs_by_type = {}
        for issue_type in scores_by_issue_type:
            merged_small_logs = \
                scores_by_issue_type[issue_type]["mrHit"]["_source"]["merged_small_logs"]
            has_several_logs_by_type[issue_type] = int(merged_small_logs.strip() != "")
        return has_several_logs_by_type

    def _has_query_several_logs(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        has_several_logs_by_type = {}
        for issue_type in scores_by_issue_type:
            merged_small_logs = \
                scores_by_issue_type[issue_type]["compared_log"]["_source"]["merged_small_logs"]
            has_several_logs_by_type[issue_type] = int(merged_small_logs.strip() != "")
        return has_several_logs_by_type

    def find_most_relevant_by_type(self):
        if self.scores_by_issue_type is not None:
            return self.scores_by_issue_type
        self.scores_by_issue_type = {}
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]
                hit["es_pos"] = idx

                if issue_type not in self.scores_by_issue_type:
                    self.scores_by_issue_type[issue_type] = {
                        "mrHit": hit,
                        "compared_log": log,
                        "score": 0}

                issue_type_item = self.scores_by_issue_type[issue_type]
                if hit["_score"] > issue_type_item["mrHit"]["_score"]:
                    self.scores_by_issue_type[issue_type]["mrHit"] = hit
                    self.scores_by_issue_type[issue_type]["compared_log"] = log

            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]
                self.scores_by_issue_type[issue_type]["score"] += \
                    (hit["normalized_score"] / self.total_normalized)
        return self.scores_by_issue_type

    def _calculate_score(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        return {item: scores_by_issue_type[item]["score"] for item in scores_by_issue_type}

    def _calculate_place(self):
        scores_by_issue_type = self._calculate_score()
        place_by_issue_type = {}
        for idx, issue_type_item in enumerate(sorted(scores_by_issue_type.items(),
                                                     key=lambda x: x[1],
                                                     reverse=True)):
            place_by_issue_type[issue_type_item[0]] = 1 / (1 + idx)
        return place_by_issue_type

    def _calculate_max_score_and_pos(self, return_val_name="max_score"):
        max_scores_by_issue_type = {}
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if issue_type not in max_scores_by_issue_type or \
                        hit["normalized_score"] > max_scores_by_issue_type[issue_type]["max_score"]:
                    max_scores_by_issue_type[issue_type] = {"max_score": hit["normalized_score"],
                                                            "max_score_pos": 1 / (1 + idx), }

        return {item: max_scores_by_issue_type[item][return_val_name]
                for item in max_scores_by_issue_type}

    def _calculate_min_score_and_pos(self, return_val_name="min_score"):
        min_scores_by_issue_type = {}
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if issue_type not in min_scores_by_issue_type or \
                        hit["normalized_score"] < min_scores_by_issue_type[issue_type]["min_score"]:
                    min_scores_by_issue_type[issue_type] = {"min_score": hit["normalized_score"],
                                                            "min_score_pos": 1 / (1 + idx), }

        return {item: min_scores_by_issue_type[item][return_val_name]
                for item in min_scores_by_issue_type}

    def _calculate_percent_count_items_and_mean(self, return_val_name="mean_score", scaled=False):
        cnt_items_by_issue_type = {}
        cnt_items_glob = 0
        for log, es_results in self.all_results:
            cnt_items_glob += len(es_results)

            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if issue_type not in cnt_items_by_issue_type:
                    cnt_items_by_issue_type[issue_type] = {"mean_score": 0,
                                                           "cnt_items_percent": 0, }

                cnt_items_by_issue_type[issue_type]["cnt_items_percent"] += 1
                cnt_items_by_issue_type[issue_type]["mean_score"] += hit["normalized_score"]

        for issue_type in cnt_items_by_issue_type:
            cnt_items_by_issue_type[issue_type]["mean_score"] /= \
                cnt_items_by_issue_type[issue_type]["cnt_items_percent"]
            cnt_items_by_issue_type[issue_type]["cnt_items_percent"] /= cnt_items_glob
        return {item: cnt_items_by_issue_type[item][return_val_name]
                for item in cnt_items_by_issue_type}

    def normalize_results(self, all_elastic_results):
        all_results = []
        max_score = 0
        self.total_normalized = 0
        for log, es_results in all_elastic_results:
            for hit in es_results["hits"]["hits"]:
                max_score = max(max_score, hit["_score"])
        for log, es_results in all_elastic_results:
            for hit in es_results["hits"]["hits"]:
                hit["normalized_score"] = hit["_score"] / max_score
                self.total_normalized += hit["normalized_score"]

            all_results.append((log, es_results["hits"]["hits"]))
        return all_results

    def _calculate_similarity_percent(self, field_name="message"):
        scores_by_issue_type = self.find_most_relevant_by_type()
        if field_name not in self.similarity_calculator.similarity_dict:
            self.similarity_calculator.find_similarity(
                self.raw_results,
                [field_name])
        similarity_percent_by_type = {}
        for issue_type in scores_by_issue_type:
            group_id = (scores_by_issue_type[issue_type]["mrHit"]["_id"],
                        scores_by_issue_type[issue_type]["compared_log"]["_id"])
            sim_obj = self.similarity_calculator.similarity_dict[field_name][group_id]
            similarity_percent_by_type[issue_type] = sim_obj["similarity"]
        return similarity_percent_by_type

    def get_ordered_features_to_process(self):
        feature_graph = {}
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

    @utils.ignore_warnings
    def gather_features_info(self):
        """Gather all features from feature_ids for a test item"""
        gathered_data = []
        gathered_data_dict = {}
        issue_type_names = []
        issue_type_by_index = {}
        try:
            issue_types = self.find_most_relevant_by_type()
            for idx, issue_type in enumerate(issue_types):
                issue_type_by_index[idx] = issue_type
                issue_type_names.append(issue_type)

            for feature in self.get_ordered_features_to_process():
                if feature in self.previously_gathered_features and \
                        feature not in self.features_to_recalculate_always:
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
            gathered_data = utils.gather_feature_list(gathered_data_dict, self.feature_ids, to_list=True)
        except Exception as err:
            logger.error("Errors in boosting features calculation")
            logger.error(err)
        return gathered_data, issue_type_names
