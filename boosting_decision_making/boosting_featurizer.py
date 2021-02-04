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
from commons import similarity_calculator
from boosting_decision_making.boosting_decision_maker import BoostingDecisionMaker
import logging
import numpy as np

logger = logging.getLogger("analyzerApp.boosting_featurizer")


class BoostingFeaturizer:

    def __init__(self, all_results, config, feature_ids,
                 weighted_log_similarity_calculator=None):
        self.config = config
        self.previously_gathered_features = {}
        self.models = {}
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
            56: (self._calculate_model_probability,
                 {"model_folder": self.config["boosting_model"]},
                 self.get_necessary_features(self.config["boosting_model"]))
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
        if "filter_by_unique_id" in self.config and self.config["filter_by_unique_id"]:
            all_results = self.filter_by_unique_id(all_results)
        if "calculate_similarities" not in self.config or self.config["calculate_similarities"]:
            self.similarity_calculator.find_similarity(
                all_results,
                fields_to_calc_similarity)
        self.all_results = self.normalize_results(all_results)
        self.scores_by_issue_type = None
        self.defect_type_predict_model = None
        self.used_model_info = set()
        self.features_to_recalculate_always = set([56])

    def _calculate_model_probability(self, model_folder=""):
        if not model_folder.strip():
            return []
        if model_folder not in self.models:
            logger.error("Model folder is not found: '%s'", model_folder)
            return []
        feature_ids = self.models[model_folder].get_feature_ids()
        feature_data = self.gather_feature_list(self.previously_gathered_features, feature_ids)
        predicted_labels, predicted_labels_probability = self.models[model_folder].predict(
            feature_data)
        predicted_probability = []
        for res in predicted_labels_probability:
            predicted_probability.append(float(res[1]))
        return predicted_probability

    def get_necessary_features(self, model_folder):
        if not model_folder.strip():
            return[]
        if model_folder not in self.models:
            try:
                self.models[model_folder] = BoostingDecisionMaker(folder=model_folder)
                return self.models[model_folder].get_feature_ids()
            except Exception as err:
                logger.debug(err)
                return []
        return self.models[model_folder].get_feature_ids()

    def fill_prevously_gathered_features(self, feature_list, feature_ids):
        self.previously_gathered_features = {}
        if type(feature_ids) == str:
            feature_ids = utils.transform_string_feature_range_into_list(feature_ids)
        else:
            feature_ids = feature_ids
        for i in range(len(feature_list)):
            for idx, feature in enumerate(feature_ids):
                if feature not in self.previously_gathered_features:
                    self.previously_gathered_features[feature] = []
                self.previously_gathered_features[feature].append(feature_list[i][idx])

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
            det_message = utils.clean_from_brackets(det_message)
            result[issue_type] = 0.0
            try:
                model_to_use = issue_type_to_compare.lower()[:2]
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

    def filter_by_unique_id(self, all_results):
        new_results = []
        for log, res in all_results:
            new_elastic_res = []
            unique_ids = set()
            for elastic_res in res["hits"]["hits"]:
                if elastic_res["_source"]["unique_id"] not in unique_ids:
                    unique_ids.add(elastic_res["_source"]["unique_id"])
                    new_elastic_res.append(elastic_res)
            if new_elastic_res:
                new_results.append((log, {"hits": {"hits": new_elastic_res}}))
        return new_results

    def is_the_same_test_case(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        num_of_logs_issue_type = {}
        for issue_type in scores_by_issue_type:
            rel_item_unique_id = scores_by_issue_type[issue_type]["mrHit"]["_source"]["unique_id"]
            queiried_item_unique_id = scores_by_issue_type[issue_type]["compared_log"]["_source"]["unique_id"]
            if not rel_item_unique_id.strip() and not queiried_item_unique_id.strip():
                num_of_logs_issue_type[issue_type] = 0
            else:
                num_of_logs_issue_type[issue_type] = int(rel_item_unique_id == queiried_item_unique_id)
        return num_of_logs_issue_type

    def has_the_same_test_case_in_all_results(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        num_of_logs_issue_type = {}
        has_the_same_test_case = 0
        for issue_type in scores_by_issue_type:
            rel_item_unique_id = scores_by_issue_type[issue_type]["mrHit"]["_source"]["unique_id"]
            queiried_item_unique_id = scores_by_issue_type[issue_type]["compared_log"]["_source"]["unique_id"]
            if not rel_item_unique_id.strip():
                continue
            if rel_item_unique_id == queiried_item_unique_id:
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

    def is_only_additional_info(self):
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

    def gather_feature_list(self, gathered_data_dict, feature_ids):
        len_data = 0
        for feature in feature_ids:
            len_data = len(gathered_data_dict[feature])
            break
        gathered_data = np.zeros((len_data, len(feature_ids)))
        for idx, feature in enumerate(feature_ids):
            for j in range(len(gathered_data_dict[feature])):
                gathered_data[j][idx] = round(gathered_data_dict[feature][j], 2)
        gathered_data = gathered_data.tolist()
        return gathered_data

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

            feature_graph = {}
            for feature in self.feature_ids:
                _, _, dependants = self.feature_functions[feature]
                feature_graph[feature] = dependants
            ordered_features = utils.topological_sort(feature_graph)

            for feature in ordered_features:
                if feature in self.previously_gathered_features and\
                        feature not in self.features_to_recalculate_always:
                    gathered_data_dict[feature] = self.previously_gathered_features[feature]
                else:
                    func, args, _ = self.feature_functions[feature]
                    result = func(**args)
                    if type(result) == list:
                        gathered_data_dict[feature] = [round(r, 2) for r in result]
                    else:
                        gathered_data_dict[feature] = []
                        for idx in sorted(issue_type_by_index.keys()):
                            issue_type = issue_type_by_index[idx]
                            gathered_data_dict[feature].append(round(result[issue_type], 2))
                    self.previously_gathered_features[feature] = gathered_data_dict[feature]

            gathered_data = self.gather_feature_list(gathered_data_dict, self.feature_ids)
        except Exception as err:
            logger.error("Errors in boosting features calculation")
            logger.error(err)
        return gathered_data, issue_type_names
