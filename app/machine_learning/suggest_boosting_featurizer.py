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

from collections import defaultdict
from typing import Any

from app.machine_learning import boosting_featurizer
from app.machine_learning.models import WeightedSimilarityCalculator


class SuggestBoostingFeaturizer(boosting_featurizer.BoostingFeaturizer):

    def __init__(self, all_results, config, feature_ids: str | list[int],
                 weighted_log_similarity_calculator: WeightedSimilarityCalculator = None) -> None:
        super().__init__(
            all_results, config, feature_ids, weighted_log_similarity_calculator=weighted_log_similarity_calculator)

    def _calculate_percent_issue_types(self) -> dict[str, float]:
        scores_by_issue_type = self.find_most_relevant_by_type()
        percent_by_issue_type = {}
        issue_types = set()
        for search_rs in scores_by_issue_type.values():
            issue_type = search_rs["mrHit"]["_source"]["issue_type"]
            issue_types.add(issue_type)
        for test_item in scores_by_issue_type:
            percent_by_issue_type[test_item] = 1 / len(issue_types) if len(issue_types) > 0 else 0
        return percent_by_issue_type

    def find_most_relevant_by_type(self) -> dict[str, dict[str, Any]]:
        if self.scores_by_type is not None:
            return self.scores_by_type
        scores_by_type = defaultdict(lambda: {'mrHit': {'_score': -1}, 'score': 0})
        for log, es_results in self.all_results:
            for idx, hit in enumerate(es_results):
                test_item = str(hit['_source']['test_item'])
                hit['es_pos'] = idx

                issue_type_item = scores_by_type[test_item]
                if hit['_score'] > issue_type_item['mrHit']['_score']:
                    issue_type_item['mrHit'] = hit
                    issue_type_item['compared_log'] = log

                issue_type_item['score'] = max(issue_type_item['score'], hit['normalized_score'])
        self.scores_by_type = dict(scores_by_type)
        return self.scores_by_type

    def _calculate_max_score_and_pos(self, return_val_name="max_score"):
        max_scores_results = super()._calculate_max_score_and_pos(return_val_name=return_val_name)
        max_scores_by_test_item = {}
        test_items = self.find_most_relevant_by_type()
        for test_item, search_rs in test_items.items():
            issue_type = search_rs["mrHit"]["_source"]["issue_type"]
            max_scores_by_test_item[test_item] = max_scores_results[issue_type]
        return max_scores_by_test_item

    def _calculate_min_score_and_pos(self, return_val_name="min_score"):
        min_scores_results = super()._calculate_min_score_and_pos(return_val_name=return_val_name)
        min_scores_by_test_item = {}
        test_items = self.find_most_relevant_by_type()
        for test_item, search_rs in test_items.items():
            issue_type = search_rs["mrHit"]["_source"]["issue_type"]
            min_scores_by_test_item[test_item] = min_scores_results[issue_type]
        return min_scores_by_test_item

    def _calculate_percent_count_items_and_mean(self, return_val_name="mean_score") -> dict[str, float]:
        mean_scores_results = super()._calculate_percent_count_items_and_mean(return_val_name=return_val_name)
        mean_scores_by_test_item = {}
        test_items = self.find_most_relevant_by_type()
        for test_item, search_rs in test_items.items():
            issue_type = search_rs["mrHit"]["_source"]["issue_type"]
            mean_scores_by_test_item[test_item] = mean_scores_results[issue_type]
        return mean_scores_by_test_item
