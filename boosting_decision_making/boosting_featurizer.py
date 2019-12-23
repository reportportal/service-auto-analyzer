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
            7: (self._calculate_percent_count_items_and_mean,
                {"return_val_name": "cnt_items_percent"}),
            8: (self._calculate_max_score_and_pos, {"return_val_name": "max_score_global_percent"}),
            9: (self._calculate_percent_issue_types, {}),
            10: (self._calculate_query_terms_percent, {}),
            11: (self._calculate_similarity_percent, {}),
        }

        if type(feature_ids) == str:
            self.feature_ids = utils.transform_string_feature_range_into_list(feature_ids)
        else:
            self.feature_ids = feature_ids

    def _calculate_percent_issue_types(self):
        scores_by_issue_type = self._calculate_score()
        percent_by_issue_type = {}
        for issue_type in scores_by_issue_type:
            percent_by_issue_type[issue_type] = 1 / len(scores_by_issue_type)\
                if len(scores_by_issue_type) > 0 else 0
        return percent_by_issue_type

    def find_most_relevant_by_type(self):
        if self.scores_by_issue_type is not None:
            return self.scores_by_issue_type
        self.scores_by_issue_type = {}
        for log_message, es_results in self.all_results:
            for hit in es_results:
                issue_type = hit["_source"]["issue_type"]

                if issue_type not in self.scores_by_issue_type:
                    self.scores_by_issue_type[issue_type] = {"mrHit": hit,
                                                             "log_message": log_message,
                                                             "score": 0}

                issue_type_item = self.scores_by_issue_type[issue_type]
                if hit["_score"] > issue_type_item["mrHit"]["_score"]:
                    self.scores_by_issue_type[issue_type]["mrHit"] = hit
                    self.scores_by_issue_type[issue_type]["log_message"] = log_message

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

    def _calculate_max_score_and_pos(self, return_val_name="max_score"):
        max_scores_by_issue_type = {}
        max_score_global = 0
        for log_message, es_results in self.all_results:
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

        return dict([(item, max_scores_by_issue_type[item][return_val_name])
                    for item in max_scores_by_issue_type])

    def _calculate_min_score_and_pos(self, return_val_name="min_score"):
        min_scores_by_issue_type = {}
        for log_message, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                issue_type = hit["_source"]["issue_type"]

                if issue_type not in min_scores_by_issue_type or\
                        hit["normalized_score"] < min_scores_by_issue_type[issue_type]["min_score"]:
                    min_scores_by_issue_type[issue_type] = {"min_score": hit["normalized_score"],
                                                            "min_score_pos": 1 / (1 + idx), }
        return dict([(item, min_scores_by_issue_type[item][return_val_name])
                    for item in min_scores_by_issue_type])

    def _calculate_percent_count_items_and_mean(self, return_val_name="mean_score"):
        cnt_items_by_issue_type = {}
        cnt_items_glob = 0
        for log_message, es_results in self.all_results:
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
        for log_message, es_results in all_elastic_results:
            total_score = 0
            for hit in es_results["hits"]["hits"]:
                total_score += hit["_score"]

            for hit in es_results["hits"]["hits"]:
                hit["normalized_score"] = hit["_score"] / total_score

            all_results.append((log_message, es_results["hits"]["hits"]))
        return all_results

    def _calculate_query_terms_percent(self):
        scores_by_issue_type = self.find_most_relevant_by_type()
        query_terms_percent_by_type = {}
        for issue_type in scores_by_issue_type:
            all_query_words = utils.find_query_words_count_from_explanation(
                scores_by_issue_type[issue_type]["mrHit"])

            all_log_words = utils.split_words(scores_by_issue_type[issue_type]["log_message"])

            query_terms_percent_by_type[issue_type] =\
                (self.config["min_should_match"]
                    if len(all_log_words) >= self.config["max_query_terms"]
                    else len(all_query_words) / self.config["max_query_terms"])
        return query_terms_percent_by_type

    def _calculate_similarity_percent(self, field_name="message", log_field="log_message"):
        scores_by_issue_type = self.find_most_relevant_by_type()
        similarity_percent_by_type = {}
        messages_to_check = {}
        all_messages = []
        message_index = 0
        for issue_type in scores_by_issue_type:
            min_word_length = self.config["min_word_length"]\
                if "min_word_length" in self.config else 0
            similar_message = scores_by_issue_type[issue_type]["mrHit"]["_source"]["message"]
            all_message_words = " ".join(utils.split_words(similar_message,
                                                           min_word_length=min_word_length))
            query_message = scores_by_issue_type[issue_type]["log_message"]
            all_log_query_words = " ".join(utils.split_words(query_message,
                                                             min_word_length=min_word_length))
            if all_message_words.strip() == "" and all_log_query_words.strip() == "":
                similarity_percent_by_type[issue_type] = 1.0
            elif all_message_words.strip() == "" or all_log_query_words.strip() == "":
                similarity_percent_by_type[issue_type] = 0.0
            else:
                all_messages.append(all_message_words)
                all_messages.append(all_log_query_words)
                messages_to_check[issue_type] = [message_index, message_index + 1]
                message_index += 2

        if len(all_messages) > 0:
            vectorizer = CountVectorizer(binary=True, analyzer="word", token_pattern="[^ ]+")
            count_vector_matrix = vectorizer.fit_transform(all_messages)
            for issue_type in messages_to_check:
                indices_to_check = messages_to_check[issue_type]
                similarity_percent_by_type[issue_type] =\
                    float(cosine_similarity(count_vector_matrix[indices_to_check[0]],
                                            count_vector_matrix[indices_to_check[1]]))
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
                    gathered_data[issue_type_by_index[issue_type]].append(result[issue_type])
        except Exception as err:
            logger.error("Errors in boosting features calculation")
            logger.error(err)

        return gathered_data, issue_type_names
