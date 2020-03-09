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
import logging
from threading import Thread

logger = logging.getLogger("analyzerApp.boosting_featurizer")


class BoostingFeaturizer:

    def __init__(self, all_results, config, feature_ids,
                 weighted_log_similarity_calculator=None):
        self.config = config
        self.weighted_log_similarity_calculator = weighted_log_similarity_calculator
        self.prepare_word_vectors(all_results)
        if "filter_min_should_match" in self.config:
            for field in self.config["filter_min_should_match"]:
                all_results = self.filter_by_min_should_match(all_results, field=field)
        self.all_results = self.normalize_results(all_results)
        self.scores_by_issue_type = None

        self.feature_functions = {
            0: (self._calculate_score, {}),
            1: (self._calculate_place, {}),
            3: (self._calculate_max_score_and_pos, {"return_val_name": "max_score_pos"}),
            5: (self._calculate_min_score_and_pos, {"return_val_name": "min_score_pos"}),
            7: (self._calculate_percent_count_items_and_mean, {"return_val_name": "cnt_items_percent"}),
            9: (self._calculate_percent_issue_types, {}),
            11: (self._calculate_similarity_percent, {"field_name": "message"}),
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
        }

        if type(feature_ids) == str:
            self.feature_ids = utils.transform_string_feature_range_into_list(feature_ids)
        else:
            self.feature_ids = feature_ids

    class CountVectorizerThread(Thread):
        def __init__(self, fields, all_results, config, weight_calculator):
            Thread.__init__(self)
            self.fields = fields
            self.config = config
            self.weight_calculator = weight_calculator
            self.all_results = all_results
            self.dict_count_vectorizer = {}
            self.all_text_field_ids = {}

        def run(self):
            for field in self.fields:
                log_field_ids = {}
                index_in_message_array = 0
                count_vector_matrix = None
                all_messages = []
                for log, res in self.all_results:
                    for obj in [log] + res["hits"]["hits"]:
                        if obj["_id"] not in log_field_ids:
                            if self.weight_calculator is None:
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
                                        text = self.weight_calculator.message_to_array(
                                            obj["_source"]["detected_message"],
                                            obj["_source"]["stacktrace"])
                                    elif field == "stacktrace":
                                        text = self.weight_calculator.message_to_array(
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
                self.all_text_field_ids[field] = log_field_ids
                self.dict_count_vectorizer[field] = count_vector_matrix

    @utils.ignore_warnings
    def prepare_word_vectors(self, all_results):
        self.all_text_field_ids = {}
        self.dict_count_vectorizer = {}
        if "min_word_length" not in self.config:
            self.config["min_word_length"] = 0
        threads = []
        for fields in [["message", "detected_message", "detected_message_with_numbers"],
                       ["merged_small_logs", "stacktrace", "only_numbers"]]:
            thread = BoostingFeaturizer.CountVectorizerThread(fields, all_results, self.config,
                                                              self.weighted_log_similarity_calculator)
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            for field in thread.fields:
                self.all_text_field_ids[field] = thread.all_text_field_ids[field]
                self.dict_count_vectorizer[field] = thread.dict_count_vectorizer[field]

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
        rearranged_items = []
        for log, res in all_results:
            for elastic_res in res["hits"]["hits"]:
                rearranged_items.append(((elastic_res["_id"], log["_id"]), log, elastic_res))

        all_results_similarity, sim_field_dict =\
            self._calculate_field_similarity(rearranged_items,
                                             field,
                                             for_filter=True)

        for log, elastic_res in all_results:
            for res in elastic_res["hits"]["hits"]:
                group_id = (res["_id"], log["_id"])
                res[sim_field_dict[group_id]] = all_results_similarity[group_id]
        return all_results

    def _calculate_field_similarity(self, items, field_name, for_filter=False):
        all_results_similarity = {}
        sim_field_dict = {}
        for group_id, log, elastic_res in items:
            sim_field = "similarity_%s" % field_name
            field_to_check = field_name

            if sim_field in elastic_res:
                all_results_similarity[group_id] = elastic_res[sim_field]
                continue
            index_query_message = self.all_text_field_ids[field_to_check][log["_id"]]
            index_log_message = self.all_text_field_ids[field_to_check][elastic_res["_id"]]
            if (type(index_query_message) == int and index_query_message < 0) and\
                    (type(index_log_message) == int and index_log_message < 0):
                if for_filter and field_to_check in ["message", "detected_message",
                                                     "detected_message_with_numbers"]:
                    index_query_message = self.all_text_field_ids["merged_small_logs"][log["_id"]]
                    index_log_message = self.all_text_field_ids["merged_small_logs"][elastic_res["_id"]]
                    sim_field = "similarity_merged_small_logs"
                    field_to_check = "merged_small_logs"
                else:
                    all_results_similarity[group_id] = 1.0
            sim_field_dict[group_id] = sim_field

            if (type(index_query_message) == int and index_query_message < 0) or\
                    (type(index_log_message) == int and index_log_message < 0):
                if group_id not in all_results_similarity:
                    all_results_similarity[group_id] = 0.0
            else:
                if self.weighted_log_similarity_calculator is None:
                    all_results_similarity[group_id] =\
                        round(1 - spatial.distance.cosine(
                            self.dict_count_vectorizer[field_to_check][index_query_message],
                            self.dict_count_vectorizer[field_to_check][index_log_message]), 3)
                else:
                    field_lines_array = self.dict_count_vectorizer[field_to_check]
                    query_vector = self.weighted_log_similarity_calculator.weigh_data_rows(
                        field_lines_array[index_query_message[0]:index_query_message[1] + 1])
                    log_vector = self.weighted_log_similarity_calculator.weigh_data_rows(
                        field_lines_array[index_log_message[0]:index_log_message[1] + 1])
                    all_results_similarity[group_id] =\
                        round(1 - spatial.distance.cosine(query_vector, log_vector), 3)

        return all_results_similarity, sim_field_dict

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
                    (hit["normalized_score"] / self.total_normalized)
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

    def _calculate_max_score_and_pos(self, return_val_name="max_score"):
        max_scores_by_issue_type = {}
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if issue_type not in max_scores_by_issue_type or\
                        hit["normalized_score"] > max_scores_by_issue_type[issue_type]["max_score"]:
                    max_scores_by_issue_type[issue_type] = {"max_score": hit["normalized_score"],
                                                            "max_score_pos": 1 / (1 + idx), }

        return dict([(item, max_scores_by_issue_type[item][return_val_name])
                    for item in max_scores_by_issue_type])

    def _calculate_min_score_and_pos(self, return_val_name="min_score"):
        min_scores_by_issue_type = {}
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if issue_type not in min_scores_by_issue_type or\
                        hit["normalized_score"] < min_scores_by_issue_type[issue_type]["min_score"]:
                    min_scores_by_issue_type[issue_type] = {"min_score": hit["normalized_score"],
                                                            "min_score_pos": 1 / (1 + idx), }

        return dict([(item, min_scores_by_issue_type[item][return_val_name])
                    for item in min_scores_by_issue_type])

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
        return dict([(item, cnt_items_by_issue_type[item][return_val_name])
                    for item in cnt_items_by_issue_type])

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
                for config_field in self.config:
                    hit[config_field] = self.config[config_field]

            all_results.append((log, es_results["hits"]["hits"]))
        return all_results

    def _calculate_similarity_percent(self, field_name="message"):
        scores_by_issue_type = self.find_most_relevant_by_type()
        rearranged_items = []
        for issue_type in scores_by_issue_type:
            rearranged_items.append((issue_type,
                                     scores_by_issue_type[issue_type]["compared_log"],
                                     scores_by_issue_type[issue_type]["mrHit"]))

        similarity_percent_by_type, sim_field_dict =\
            self._calculate_field_similarity(rearranged_items,
                                             field_name,
                                             for_filter=False)

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
                    gathered_data[issue_type_by_index[issue_type]].append(round(result[issue_type], 3))
        except Exception as err:
            logger.error("Errors in boosting features calculation")
            logger.error(err)

        return gathered_data, issue_type_names
