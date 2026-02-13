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

from time import time
from typing import Any, Optional

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig, SuggestPattern, SuggestPatternLabel
from app.commons.os_client import OsClient
from app.utils import text_processing

LOGGER = logging.getLogger("analyzerApp.suggestPatternsService")


class SuggestPatternsService:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    os_client: OsClient

    def __init__(self, app_config: ApplicationConfig, search_cfg: SearchConfig, os_client: Optional[OsClient] = None):
        """Initialize SuggestPatternsService

        :param app_config: Application configuration object
        :param search_cfg: Search configuration object
        :param os_client: Optional OsClient instance. If not provided, a new one will be created.
        """
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.os_client = os_client or OsClient(app_config=self.app_config)

    def _build_query(self, labels: list[str]) -> dict[str, Any]:
        label_filters = [
            {
                "wildcard": {
                    "issue_type": {"value": f"{label}*", "case_insensitive": True},
                }
            }
            for label in labels
        ]
        query = {
            "sort": {"start_time": "desc"},
            "size": self.app_config.esChunkNumber,
            "query": {"bool": {"filter": [{"bool": {"should": label_filters}}]}},
            "_source": ["launch_id", "test_item_id", "logs", "issue_type"],
        }
        return query

    def _get_patterns_with_labels(
        self, exceptions_with_labels: dict[str, dict[str, int]]
    ) -> list[SuggestPatternLabel]:
        min_count = self.search_cfg.PatternLabelMinCountToSuggest
        min_percent = self.search_cfg.PatternLabelMinPercentToSuggest
        suggested_patterns_with_labels = []
        for exception in exceptions_with_labels:
            sum_all = sum(exceptions_with_labels[exception].values())
            for issue_type in exceptions_with_labels[exception]:
                percent_for_label = round(exceptions_with_labels[exception][issue_type] / sum_all, 2)
                count_for_exception_with_label = exceptions_with_labels[exception][issue_type]
                if percent_for_label >= min_percent and count_for_exception_with_label >= min_count:
                    suggested_patterns_with_labels.append(
                        SuggestPatternLabel(
                            pattern=exception,
                            totalCount=sum_all,
                            percentTestItemsWithLabel=percent_for_label,
                            label=issue_type,
                        )
                    )
        return suggested_patterns_with_labels

    def _get_patterns_without_labels(self, all_exceptions: dict) -> list[SuggestPatternLabel]:
        suggested_patterns_without_labels = []
        for exception in all_exceptions:
            if all_exceptions[exception] >= self.search_cfg.PatternMinCountToSuggest:
                suggested_patterns_without_labels.append(
                    SuggestPatternLabel(pattern=exception, totalCount=all_exceptions[exception])
                )
        return suggested_patterns_without_labels

    def suggest_patterns(self, project_id: int) -> SuggestPattern:
        LOGGER.info(f"Started suggesting patterns for project '{project_id}'")
        t_start = time()
        exceptions_with_labels: dict[str, dict[str, int]] = {}
        all_exceptions = {}
        query = self._build_query(["ab", "pb", "si", "ti"])
        for hit in self.os_client.search(project_id, query):
            issue_type = (hit.source.issue_type or "").strip()
            if not issue_type:
                continue
            for log in hit.source.logs or []:
                detected_message = log.detected_message
                for exception in text_processing.get_found_exceptions(detected_message):
                    if exception.strip():
                        if exception not in all_exceptions:
                            all_exceptions[exception] = 0
                        all_exceptions[exception] += 1

                        if issue_type[:2].lower() != "ti":
                            if exception not in exceptions_with_labels:
                                exceptions_with_labels[exception] = {}
                            if issue_type not in exceptions_with_labels[exception]:
                                exceptions_with_labels[exception][issue_type] = 0
                            exceptions_with_labels[exception][issue_type] += 1
        suggested_patterns_with_labels = self._get_patterns_with_labels(exceptions_with_labels)
        suggested_patterns_without_labels = self._get_patterns_without_labels(all_exceptions)
        LOGGER.info("Finished suggesting patterns %.2f s", time() - t_start)
        return SuggestPattern(
            suggestionsWithLabels=suggested_patterns_with_labels,
            suggestionsWithoutLabels=suggested_patterns_without_labels,
        )
