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

import logging
import utils.utils as utils
import elasticsearch
import elasticsearch.helpers
from time import time
from commons.esclient import EsClient
from commons.launch_objects import SuggestPattern, SuggestPatternLabel

logger = logging.getLogger("analyzerApp.suggestPatternsService")


class SuggestPatternsService:

    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.es_client = EsClient(app_config=app_config, search_cfg=search_cfg)

    def query_data(self, project, label):
        data = []
        for d in elasticsearch.helpers.scan(
                self.es_client.es_client,
                index=project,
                query={
                    "_source": ["detected_message", "issue_type"],
                    "sort": {"start_time": "desc"},
                    "size": self.app_config["esChunkNumber"],
                    "query": {
                        "bool": {
                            "must": [
                                {
                                    "bool": {
                                        "should": [
                                            {"wildcard": {"issue_type": "{}*".format(label.upper())}},
                                            {"wildcard": {"issue_type": "{}*".format(label.lower())}},
                                            {"wildcard": {"issue_type": "{}*".format(label)}},
                                        ]
                                    }
                                }
                            ],
                            "should": [
                                {"term": {"is_auto_analyzed": {"value": "false", "boost": 1.0}}},
                            ]
                        }
                    }
                }):
            data.append((d["_source"]["detected_message"], d["_source"]["issue_type"]))
        return data

    def get_patterns_with_labels(self, exceptions_with_labels):
        min_count = self.search_cfg["PatternLabelMinCountToSuggest"]
        min_percent = self.search_cfg["PatternLabelMinPercentToSuggest"]
        suggestedPatternsWithLabels = []
        for exception in exceptions_with_labels:
            sum_all = sum(exceptions_with_labels[exception].values())
            for issue_type in exceptions_with_labels[exception]:
                percent_for_label = round(exceptions_with_labels[exception][issue_type] / sum_all, 2)
                count_for_exception_with_label = exceptions_with_labels[exception][issue_type]
                if percent_for_label >= min_percent and count_for_exception_with_label >= min_count:
                    suggestedPatternsWithLabels.append(SuggestPatternLabel(
                        pattern=exception,
                        totalCount=sum_all,
                        percentTestItemsWithLabel=percent_for_label,
                        label=issue_type))
        return suggestedPatternsWithLabels

    def get_patterns_without_labels(self, all_exceptions):
        suggestedPatternsWithoutLabels = []
        for exception in all_exceptions:
            if all_exceptions[exception] >= self.search_cfg["PatternMinCountToSuggest"]:
                suggestedPatternsWithoutLabels.append(SuggestPatternLabel(
                    pattern=exception,
                    totalCount=all_exceptions[exception]))
        return suggestedPatternsWithoutLabels

    @utils.ignore_warnings
    def suggest_patterns(self, project_id):
        index_name = utils.unite_project_name(
            str(project_id), self.app_config["esProjectIndexPrefix"])
        logger.info("Started suggesting patterns for project '%s'", index_name)
        t_start = time()
        found_data = []
        exceptions_with_labels = {}
        all_exceptions = {}
        if not self.es_client.index_exists(index_name):
            return SuggestPattern(
                suggestionsWithLabels=[],
                suggestionsWithoutLabels=[])
        for label in ["ab", "pb", "si", "ti"]:
            found_data.extend(self.query_data(index_name, label))
        for log, label in found_data:
            for exception in utils.get_found_exceptions(log).split(" "):
                if exception.strip():
                    if exception not in all_exceptions:
                        all_exceptions[exception] = 0
                    all_exceptions[exception] += 1

                    if label[:2].lower() != "ti":
                        if exception not in exceptions_with_labels:
                            exceptions_with_labels[exception] = {}
                        if label not in exceptions_with_labels[exception]:
                            exceptions_with_labels[exception][label] = 0
                        exceptions_with_labels[exception][label] += 1
        suggestedPatternsWithLabels = self.get_patterns_with_labels(exceptions_with_labels)
        suggestedPatternsWithoutLabels = self.get_patterns_without_labels(all_exceptions)
        logger.info("Finished suggesting patterns %.2f s", time() - t_start)
        return SuggestPattern(
            suggestionsWithLabels=suggestedPatternsWithLabels,
            suggestionsWithoutLabels=suggestedPatternsWithoutLabels)
