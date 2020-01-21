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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import logging

logger = logging.getLogger("analyzerApp.boosting_featurizer")


class BoostingFeaturizer:

    def __init__(self, all_results, config, feature_ids):
        self.config = config
        if "filter_min_should_match" in self.config:
            for field in self.config["filter_min_should_match"]:
                all_results = self.filter_by_min_should_match(all_results, field=field)
        self.all_results = self.normalize_results(all_results)
        self.scores_by_issue_type = None

        self.feature_functions = {
            0: (self._calculate_score, {}),
            1: (self._calculate_place, {}),
            2: (self._calculate_max_score_and_pos, {"return_val_name": "max_score"}),
            3: (self._calculate_max_score_and_pos, {"return_val_name": "max_score_pos"}),
            4: (self._calculate_min_score_and_pos, {"return_val_name": "min_score"}),
            5: (self._calculate_min_score_and_pos, {"return_val_name": "min_score_pos"}),
            6: (self._calculate_percent_count_items_and_mean, {"return_val_name": "mean_score"}),
            7: (self._calculate_percent_count_items_and_mean, {"return_val_name": "cnt_items_percent"}),
            8: (self._calculate_max_score_and_pos, {"return_val_name": "max_score_global_percent"}),
            9: (self._calculate_percent_issue_types, {}),
            10: (self._calculate_query_terms_percent, {"field_name": "message"}),
            11: (self._calculate_similarity_percent, {"field_name": "message"}),
            13: (self._calculate_similarity_percent, {"field_name": "merged_small_logs"}),
            14: (self._has_test_item_several_logs, {}),
            15: (self._has_query_several_logs, {}),
            16: (self._calculate_query_terms_percent, {"field_name": "message"}),
            17: (self._calculate_query_terms_percent, {"field_name": "merged_small_logs"}),
            18: (self._calculate_similarity_percent, {"field_name": "detected_message"}),
            19: (self._calculate_similarity_percent, {"field_name": "detected_message_with_numbers"}),
            20: (self._calculate_query_terms_percent, {"field_name": "detected_message"}),
            21: (self._calculate_query_terms_percent, {"field_name": "detected_message_with_numbers"}),
            22: (self._calculate_query_terms_percent, {"field_name": "stacktrace"}),
            23: (self._calculate_similarity_percent, {"field_name": "stacktrace"}),
            24: (self._calculate_query_terms_percent, {"field_name": "only_numbers"}),
            25: (self._calculate_similarity_percent, {"field_name": "only_numbers"}),
            26: (self._calculate_max_score_and_pos, {"return_val_name": "max_score", "scaled": True}),
            27: (self._calculate_min_score_and_pos, {"return_val_name": "min_score", "scaled": True}),
            28: (self._calculate_percent_count_items_and_mean,
                 {"return_val_name": "mean_score", "scaled": True}),
        }

        if type(feature_ids) == str:
            self.feature_ids = utils.transform_string_feature_range_into_list(feature_ids)
        else:
            self.feature_ids = feature_ids

    def filter_by_min_should_match(self, all_results, field="message"):
        all_results = self.calculate_sim_percent_logs(all_results, field=field)
        new_results = []
        for log, res in all_results:
            new_elastic_res = []
            for elastic_res in res["hits"]["hits"]:
                sim_field = "similarity_%s" % field\
                    if "similarity_%s" % field in elastic_res else "similarity_merged_small_logs"
                if elastic_res[sim_field] >= self.config["min_should_match"]:
                    new_elastic_res.append(elastic_res)
            if len(new_elastic_res) > 0:
                new_results.append((log, {"hits": {"hits": new_elastic_res}}))
        return new_results

    def calculate_sim_percent_logs(self, all_results, field="message"):
        all_results_similarity = {}
        rearranged_items = []
        for log, res in all_results:
            for elastic_res in res["hits"]["hits"]:
                rearranged_items.append((elastic_res["_id"], log, elastic_res))

        all_messages, messages_to_check, all_results_similarity, sim_field_dict =\
            self._prepare_message_for_similarity_check(rearranged_items,
                                                       field,
                                                       for_filter=True)

        calculated_similarity = self._calculate_similarity(all_messages, messages_to_check)
        for key, val in calculated_similarity.items():
            all_results_similarity[key] = val

        for log, elastic_res in all_results:
            for res in elastic_res["hits"]["hits"]:
                res[sim_field_dict[res["_id"]]] = all_results_similarity[res["_id"]]
        return all_results

    def _calculate_similarity(self, all_messages, messages_to_check):
        if len(all_messages) > 0:
            all_results_similarity = {}
            vectorizer = CountVectorizer(binary=True, analyzer="word", token_pattern="[^ ]+")
            count_vector_matrix = vectorizer.fit_transform(all_messages)
            for res_id in messages_to_check:
                indices_to_check = messages_to_check[res_id]
                all_results_similarity[res_id] =\
                    round(float(cosine_similarity(count_vector_matrix[indices_to_check[0]],
                          count_vector_matrix[indices_to_check[1]])), 3)
            return all_results_similarity
        return {}

    def _prepare_message_for_similarity_check(self, items, field_name, for_filter=False):
        all_results_similarity = {}
        messages_to_check = {}
        all_messages = []
        message_index = 0
        log_message_index = {}
        sim_field_dict = {}
        for group_id, log, elastic_res in items:
            sim_field = "similarity_%s" % field_name

            if sim_field in elastic_res:
                all_results_similarity[group_id] = elastic_res[sim_field]
                continue
            min_word_length = self.config["min_word_length"] if "min_word_length" in self.config else 0

            all_message_words = " ".join(utils.split_words(elastic_res["_source"][field_name],
                                         min_word_length=min_word_length))
            all_log_query_words = " ".join(utils.split_words(log["_source"][field_name],
                                           min_word_length=min_word_length))

            if all_message_words.strip() == "" and all_log_query_words.strip() == "":
                if for_filter:
                    all_message_words = " ".join(utils.split_words(
                        elastic_res["_source"]["merged_small_logs"],
                        min_word_length=min_word_length))
                    all_log_query_words = " ".join(utils.split_words(
                        log["_source"]["merged_small_logs"],
                        min_word_length=min_word_length))
                    sim_field = "similarity_merged_small_logs"
                else:
                    all_results_similarity[group_id] = 1.0
            sim_field_dict[group_id] = sim_field

            if all_message_words.strip() == "" or all_log_query_words.strip() == "":
                if group_id not in all_results_similarity:
                    all_results_similarity[group_id] = 0.0
            else:
                new_message_ind = message_index
                all_messages.append(all_message_words)
                message_index += 1

                log_message_index_in_array = message_index
                if log["_id"] not in log_message_index:
                    all_messages.append(all_log_query_words)
                    log_message_index[log["_id"]] = log_message_index_in_array
                    message_index += 1
                else:
                    log_message_index_in_array =\
                        log_message_index[log["_id"]]
                messages_to_check[group_id] = [new_message_ind, log_message_index_in_array]
        return all_messages, messages_to_check, all_results_similarity, sim_field_dict

    def _calculate_percent_issue_types(self):
        scores_by_issue_type = self._calculate_score()
        percent_by_issue_type = {}
        for issue_type in scores_by_issue_type:
            percent_by_issue_type[issue_type] = 1 / len(scores_by_issue_type)\
                if len(scores_by_issue_type) > 0 else 0
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
                    (hit["normalized_score"] / len(self.all_results))
        return self.scores_by_issue_type

    def _calculate_score(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        return dict([(item, scores_by_issue_type[item]["score"]) for item in scores_by_issue_type])

    def _calculate_place(self):
        scores_by_issue_type = self._calculate_score()
        place_by_issue_type = {}
        for idx, issue_type_item in enumerate(sorted(scores_by_issue_type.items(),
                                                     key=lambda x: x[1],
                                                     reverse=True)):
            place_by_issue_type[issue_type_item[0]] = 1 / (1 + idx)
        return place_by_issue_type

    def _calculate_max_score_and_pos(self, return_val_name="max_score", scaled=False):
        max_scores_by_issue_type = {}
        max_score_global = 0
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if hit["normalized_score"] > max_score_global:
                    max_score_global = hit["normalized_score"]

                if issue_type not in max_scores_by_issue_type or\
                        hit["normalized_score"] > max_scores_by_issue_type[issue_type]["max_score"]:
                    max_scores_by_issue_type[issue_type] = {"max_score": hit["normalized_score"],
                                                            "max_score_pos": 1 / (1 + idx), }
        for issue_type in max_scores_by_issue_type:
            max_scores_by_issue_type[issue_type]["max_score_global_percent"] =\
                max_scores_by_issue_type[issue_type]["max_score"] / max_score_global
            if scaled:
                max_scores_by_issue_type[issue_type]["max_score"] /= max_score_global

        return dict([(item, max_scores_by_issue_type[item][return_val_name])
                    for item in max_scores_by_issue_type])

    def _calculate_min_score_and_pos(self, return_val_name="min_score", scaled=False):
        min_scores_by_issue_type = {}
        max_score_global = 0
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if hit["normalized_score"] > max_score_global:
                    max_score_global = hit["normalized_score"]

                if issue_type not in min_scores_by_issue_type or\
                        hit["normalized_score"] < min_scores_by_issue_type[issue_type]["min_score"]:
                    min_scores_by_issue_type[issue_type] = {"min_score": hit["normalized_score"],
                                                            "min_score_pos": 1 / (1 + idx), }
        for issue_type in min_scores_by_issue_type:
            if scaled:
                min_scores_by_issue_type[issue_type]["min_score"] /= max_score_global
        return dict([(item, min_scores_by_issue_type[item][return_val_name])
                    for item in min_scores_by_issue_type])

    def _calculate_percent_count_items_and_mean(self, return_val_name="mean_score", scaled=False):
        cnt_items_by_issue_type = {}
        cnt_items_glob = 0
        max_score_global = 0
        for log, es_results in self.all_results:
            cnt_items_glob += len(es_results)

            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if hit["normalized_score"] > max_score_global:
                    max_score_global = hit["normalized_score"]

                if issue_type not in cnt_items_by_issue_type:
                    cnt_items_by_issue_type[issue_type] = {"mean_score": 0,
                                                           "cnt_items_percent":  0, }

                cnt_items_by_issue_type[issue_type]["cnt_items_percent"] += 1
                cnt_items_by_issue_type[issue_type]["mean_score"] += hit["normalized_score"]

        for issue_type in cnt_items_by_issue_type:
            cnt_items_by_issue_type[issue_type]["mean_score"] /=\
                cnt_items_by_issue_type[issue_type]["cnt_items_percent"]
            cnt_items_by_issue_type[issue_type]["cnt_items_percent"] /= cnt_items_glob
            if scaled:
                cnt_items_by_issue_type[issue_type]["mean_score"] /= max_score_global
        return dict([(item, cnt_items_by_issue_type[item][return_val_name])
                    for item in cnt_items_by_issue_type])

    def normalize_results(self, all_elastic_results):
        all_results = []
        for log, es_results in all_elastic_results:
            total_score = 0
            for hit in es_results["hits"]["hits"]:
                total_score += hit["_score"]

            for hit in es_results["hits"]["hits"]:
                hit["normalized_score"] = hit["_score"] / total_score

            all_results.append((log, es_results["hits"]["hits"]))
        return all_results

    def _calculate_query_terms_percent(self, field_name="message"):
        scores_by_issue_type = self.find_most_relevant_by_type()
        query_terms_percent_by_type = {}
        for issue_type in scores_by_issue_type:
            all_query_words = utils.find_query_words_count_from_explanation(
                scores_by_issue_type[issue_type]["mrHit"], field_name=field_name)

            all_log_words = utils.split_words(
                scores_by_issue_type[issue_type]["compared_log"]["_source"][field_name])

            if len(all_log_words) == 0:
                query_terms_percent_by_type[issue_type] = 1.0
            else:
                terms_percent = min(len(all_query_words) / len(all_log_words), 1.0)
                query_terms_percent_by_type[issue_type] = (
                    terms_percent if len(all_log_words) <= self.config["max_query_terms"]
                    else len(all_query_words) / self.config["max_query_terms"])
        return query_terms_percent_by_type

    def _calculate_similarity_percent(self, field_name="message"):
        scores_by_issue_type = self.find_most_relevant_by_type()
        rearranged_items = []
        for issue_type in scores_by_issue_type:
            rearranged_items.append((issue_type,
                                     scores_by_issue_type[issue_type]["compared_log"],
                                     scores_by_issue_type[issue_type]["mrHit"]))

        all_messages, messages_to_check, similarity_percent_by_type, sim_field_dict =\
            self._prepare_message_for_similarity_check(rearranged_items,
                                                       field_name,
                                                       for_filter=False)

        calculated_similarity = self._calculate_similarity(all_messages, messages_to_check)
        for key, val in calculated_similarity.items():
            similarity_percent_by_type[key] = val

        return similarity_percent_by_type

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
                    gathered_data[issue_type_by_index[issue_type]].append(round(result[issue_type], 3))
        except Exception as err:
            logger.error("Errors in boosting features calculation")
            logger.error(err)

        return gathered_data, issue_type_names
