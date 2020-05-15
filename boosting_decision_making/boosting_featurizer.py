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
from boosting_decision_making import similarity_calculator
import logging

logger = logging.getLogger("analyzerApp.boosting_featurizer")


class BoostingFeaturizer:

    def __init__(self, all_results, config, feature_ids,
                 weighted_log_similarity_calculator=None):
        self.config = config
        self.similarity_calculator = similarity_calculator.SimilarityCalculator(
            self.config,
            weighted_similarity_calculator=weighted_log_similarity_calculator)
        if type(feature_ids) == str:
            self.feature_ids = utils.transform_string_feature_range_into_list(feature_ids)
        else:
            self.feature_ids = feature_ids
        self.fields_to_replace_with_merged_logs = [
            "message", "detected_message",
            "detected_message_without_params_extended",
            "message_without_params_extended",
            "message_extended",
            "detected_message_extended"]
        self.feature_functions = {
            0: (self._calculate_score, {}),
            1: (self._calculate_place, {}),
            3: (self._calculate_max_score_and_pos, {"return_val_name": "max_score_pos"}),
            5: (self._calculate_min_score_and_pos, {"return_val_name": "min_score_pos"}),
            7: (self._calculate_percent_count_items_and_mean, {"return_val_name": "cnt_items_percent"}),
            9: (self._calculate_percent_issue_types, {}),
            11: (self._calculate_similarity_percent, {"field_name": "message"}),
            12: (self.is_only_merged_small_logs, {}),
            13: (self._calculate_similarity_percent, {"field_name": "merged_small_logs"}),
            14: (self._has_test_item_several_logs, {}),
            15: (self._has_query_several_logs, {}),
            18: (self._calculate_similarity_percent, {"field_name": "detected_message"}),
            19: (self._calculate_similarity_percent, {"field_name": "detected_message_with_numbers"}),
            23: (self._calculate_similarity_percent, {"field_name": "stacktrace"}),
            25: (self._calculate_similarity_percent, {"field_name": "only_numbers"}),
            26: (self._calculate_max_score_and_pos, {"return_val_name": "max_score"}),
            27: (self._calculate_min_score_and_pos, {"return_val_name": "min_score"}),
            28: (self._calculate_percent_count_items_and_mean,
                 {"return_val_name": "mean_score"}),
            29: (self._calculate_similarity_percent, {"field_name": "message_params"}),
            34: (self._calculate_similarity_percent, {"field_name": "found_exceptions"}),
            35: (self._is_all_log_lines, {}),
            36: (self._calculate_similarity_percent, {"field_name": "detected_message_extended"}),
            37: (self._calculate_similarity_percent,
                 {"field_name": "detected_message_without_params_extended"}),
            38: (self._calculate_similarity_percent, {"field_name": "stacktrace_extended"}),
            40: (self._calculate_similarity_percent, {"field_name": "message_without_params_extended"}),
            41: (self._calculate_similarity_percent, {"field_name": "message_extended"})
        }

        fields_to_calc_similarity = self.find_columns_to_find_similarities_for()

        if "filter_min_should_match" in self.config and len(self.config["filter_min_should_match"]) > 0:
            self.similarity_calculator.find_similarity(
                all_results,
                self.config["filter_min_should_match"] + ["merged_small_logs"])
            for field in self.config["filter_min_should_match"]:
                all_results = self.filter_by_min_should_match(all_results, field=field)
        if "filter_min_should_match_any" in self.config and\
                len(self.config["filter_min_should_match_any"]) > 0:
            self.similarity_calculator.find_similarity(
                all_results,
                self.config["filter_min_should_match_any"] + ["merged_small_logs"])
            all_results = self.filter_by_min_should_match_any(
                all_results,
                fields=self.config["filter_min_should_match_any"])
        self.similarity_calculator.find_similarity(
            all_results,
            fields_to_calc_similarity)
        self.all_results = self.normalize_results(all_results)
        self.scores_by_issue_type = None

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
            if new_elastic_res:
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
                similarity_to_compare = self.config["min_should_match"]
                for field in fields:
                    sim_obj = self.similarity_calculator.similarity_dict[field][group_id]
                    similarity = sim_obj["similarity"]
                    if sim_obj["both_empty"] and field in self.fields_to_replace_with_merged_logs:
                        sim_obj = self.similarity_calculator.similarity_dict["merged_small_logs"]
                        similarity = sim_obj[group_id]["similarity"]
                        similarity_to_compare = max(0.8, self.config["min_should_match"])
                    max_similarity = max(max_similarity, similarity)
                if max_similarity >= similarity_to_compare:
                    new_elastic_res.append(elastic_res)
            if len(new_elastic_res) > 0:
                new_results.append((log, {"hits": {"hits": new_elastic_res}}))
        return new_results

    def _calculate_percent_issue_types(self):
        scores_by_issue_type = self._calculate_score()
        percent_by_issue_type = {}
        for issue_type in scores_by_issue_type:
            percent_by_issue_type[issue_type] = 1 / len(scores_by_issue_type)\
                if len(scores_by_issue_type) else 0
        return percent_by_issue_type

    def _has_test_item_several_logs(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        has_several_logs_by_type = {}
        for issue_type in scores_by_issue_type:
            merged_small_logs =\
                scores_by_issue_type[issue_type]["mrHit"]["_source"]["merged_small_logs"]
            has_several_logs_by_type[issue_type] = int(merged_small_logs.strip() != "")
        return has_several_logs_by_type

    def _has_query_several_logs(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        has_several_logs_by_type = {}
        for issue_type in scores_by_issue_type:
            merged_small_logs =\
                scores_by_issue_type[issue_type]["compared_log"]["_source"]["merged_small_logs"]
            has_several_logs_by_type[issue_type] = int(merged_small_logs.strip() != "")
        return has_several_logs_by_type

    def find_most_relevant_by_type(self):
        if self.scores_by_issue_type is not None:
            return self.scores_by_issue_type
        self.scores_by_issue_type = {}
        for log, es_results in self.all_results:
            for hit in es_results:
                issue_type = hit["_source"]["issue_type"]

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
                self.scores_by_issue_type[issue_type]["score"] +=\
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

                if issue_type not in max_scores_by_issue_type or\
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

                if issue_type not in min_scores_by_issue_type or\
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
                                                           "cnt_items_percent":  0, }

                cnt_items_by_issue_type[issue_type]["cnt_items_percent"] += 1
                cnt_items_by_issue_type[issue_type]["mean_score"] += hit["normalized_score"]

        for issue_type in cnt_items_by_issue_type:
            cnt_items_by_issue_type[issue_type]["mean_score"] /=\
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
        similarity_percent_by_type = {}
        for issue_type in scores_by_issue_type:
            group_id = (scores_by_issue_type[issue_type]["mrHit"]["_id"],
                        scores_by_issue_type[issue_type]["compared_log"]["_id"])
            sim_obj = self.similarity_calculator.similarity_dict[field_name][group_id]
            similarity_percent_by_type[issue_type] = sim_obj["similarity"]
        return similarity_percent_by_type

    @utils.ignore_warnings
    def gather_features_info(self):
        """Gather all features from feature_ids for a test item"""
        gathered_data = []
        issue_type_names = []
        issue_type_by_index = {}
        try:
            issue_types = self.find_most_relevant_by_type()
            for idx, issue_type in enumerate(issue_types):
                gathered_data.append([])
                issue_type_by_index[issue_type] = idx
                issue_type_names.append(issue_type)
            for feature in self.feature_ids:
                func, args = self.feature_functions[feature]
                result = func(**args)
                for issue_type in result:
                    gathered_data[issue_type_by_index[issue_type]].append(round(result[issue_type], 2))
        except Exception as err:
            logger.error("Errors in boosting features calculation")
            logger.error(err)

        return gathered_data, issue_type_names
