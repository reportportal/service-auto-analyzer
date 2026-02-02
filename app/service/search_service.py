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
from app.commons.model.launch_objects import (
    ApplicationConfig,
    SearchConfig,
    SearchLogInfo,
    SearchLogs,
)
from app.commons.os_client import OsClient
from app.utils import text_processing, utils

LOGGER = logging.getLogger("analyzerApp.searchService")


class SearchService:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    os_client: OsClient

    def __init__(
        self, app_config: ApplicationConfig, search_cfg: SearchConfig, *, os_client: Optional[OsClient] = None
    ):
        """Initialize SearchService

        :param app_config: Application configuration object
        :param search_cfg: Search configuration object
        :param os_client: Optional OsClient instance. If not provided, a new one will be created.
        """
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.os_client = os_client or OsClient(app_config=self.app_config)

    def _prepare_log_messages_for_search(self, search_req: SearchLogs) -> list[str]:
        """Prepare and deduplicate log messages for search."""
        log_messages = [message for message in search_req.logMessages if message.strip()]
        if not log_messages:
            return []

        try:
            logs_to_take, _ = text_processing.find_last_unique_texts(
                search_req.analyzerConfig.similarityThresholdToDrop, log_messages
            )
        except ValueError:
            return []

        number_of_logs_to_index = search_req.analyzerConfig.numberOfLogsToIndex
        if number_of_logs_to_index > 0:
            logs_to_take = logs_to_take[-number_of_logs_to_index:]

        return [log_messages[idx] for idx in logs_to_take]

    def _build_search_query(self, search_req: SearchLogs, log_messages: list[str]) -> dict[str, Any]:
        """Build search query for Test Item-centric index."""
        min_log_level = search_req.analyzerConfig.minimumLogLevel
        min_should_match = text_processing.prepare_es_min_should_match(
            search_req.analyzerConfig.searchLogsMinShouldMatch / 100
        )
        nested_should = [
            utils.build_more_like_this_query(
                min_should_match,
                message,
                field_name="logs.message",
                boost=1.0,
                max_query_terms=self.search_cfg.MaxQueryTerms,
            )
            for message in log_messages
        ]

        query: dict[str, Any] = {
            "size": self.app_config.esChunkNumber,
            "query": {
                "bool": {
                    "filter": [
                        {"exists": {"field": "issue_type"}},
                        {"terms": {"launch_id": search_req.filteredLaunchIds}},
                    ],
                    "must_not": [{"term": {"test_item_id": str(search_req.itemId)}}],
                    "must": [
                        {
                            "bool": {
                                "should": [
                                    {"wildcard": {"issue_type": {"value": "ti*", "case_insensitive": True}}},
                                ]
                            }
                        },
                        {
                            "nested": {
                                "path": "logs",
                                "score_mode": search_req.analyzerConfig.searchScoreMode,
                                "query": {
                                    "bool": {
                                        "filter": [{"range": {"logs.log_level": {"gte": min_log_level}}}],
                                        "should": nested_should,
                                    }
                                },
                            }
                        },
                    ],
                }
            },
        }

        utils.append_aa_ma_boosts(query, self.search_cfg)
        return query

    def search_logs(self, search_req: SearchLogs) -> list[SearchLogInfo]:
        """Get all logs similar to given logs"""
        LOGGER.info(f"Started searching for test item with id: {search_req.itemId}")
        LOGGER.debug(f"Started searching by request: {search_req.model_dump_json()}")
        t_start = time()
        log_messages = self._prepare_log_messages_for_search(search_req)
        if not log_messages:
            return []

        query = self._build_search_query(search_req, log_messages)
        search_results = list(self.os_client.search(search_req.projectId, query) or [])
        if not search_results:
            return []

        joined_request_messages = "\n".join(log_messages)
        min_similarity = search_req.analyzerConfig.searchLogsMinShouldMatch / 100.0
        filtered_results: list[tuple[SearchLogInfo, float]] = []
        request_status_codes = [
            " ".join(sorted(text_processing.get_unique_potential_status_codes(message))) for message in log_messages
        ]

        candidates: list[tuple[Any, list[tuple[Any, str]], list[str]]] = []
        joined_item_messages_list: list[str] = []
        for hit in search_results:
            source = hit.source
            if not source.logs:
                continue
            logs_sorted = sorted(source.logs, key=lambda log: log.log_order if log.log_order else int(log.log_id))
            logs_with_messages = [(log, log.message) for log in logs_sorted if log.message and log.message.strip()]
            if not logs_with_messages:
                continue
            log_messages_sorted: list[str] = [message for _, message in logs_with_messages if message]
            joined_item_messages = "\n".join(log_messages_sorted)
            candidates.append((source, logs_with_messages, log_messages_sorted))
            joined_item_messages_list.append(joined_item_messages)

        if not candidates:
            return []

        cumulative_similarities, _ = text_processing.calculate_text_similarity(
            joined_request_messages, joined_item_messages_list
        )

        for (source, logs_with_messages, log_messages_sorted), cumulative_similarity in zip(
            candidates, cumulative_similarities
        ):
            if cumulative_similarity.similarity < min_similarity:
                continue

            per_log_similarity, _ = text_processing.calculate_text_similarity(
                joined_request_messages, log_messages_sorted
            )
            best_log_idx = max(
                range(len(per_log_similarity)),
                key=lambda idx: per_log_similarity[idx].similarity,
            )
            best_log = logs_with_messages[best_log_idx][0]

            request_similarity, _ = text_processing.calculate_text_similarity(best_log.message, log_messages)
            best_request_similarity = 0.0
            best_request_index = 0
            if request_similarity:
                best_request_index = max(
                    range(len(request_similarity)),
                    key=lambda idx: request_similarity[idx].similarity,
                )
                best_request_similarity = request_similarity[best_request_index].similarity

            request_codes = request_status_codes[best_request_index] if request_status_codes else ""
            log_codes = " ".join(sorted((best_log.potential_status_codes or "").split()))
            if log_codes != request_codes:
                continue

            if search_req.analyzerConfig.allMessagesShouldMatch:
                all_messages_match = True
                for request_message in log_messages:
                    per_request_similarity, _ = text_processing.calculate_text_similarity(
                        request_message, log_messages_sorted
                    )
                    if not per_request_similarity:
                        all_messages_match = False
                        break
                    if max(result.similarity for result in per_request_similarity) < min_similarity:
                        all_messages_match = False
                        break
                if not all_messages_match:
                    continue

            search_info = SearchLogInfo(
                logId=utils.extract_real_id(best_log.log_id),
                testItemId=int(source.test_item_id),
                matchScore=round(best_request_similarity, 2) * 100,
            )
            filtered_results.append((search_info, cumulative_similarity.similarity))

        filtered_results.sort(key=lambda entry: entry[1], reverse=True)
        final_results = [entry[0] for entry in filtered_results]
        LOGGER.info(
            "Finished searching by request %s with %d results. It took %.2f sec.",
            search_req.model_dump_json(),
            len(final_results),
            time() - t_start,
        )
        return final_results
