#  Copyright 2026 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Test Item-centric OpenSearch query builders shared by analysis, suggestions and model training.

One query is built per request Test Item. Every request log gets its own `nested` clause with named inner hits
(`log_{i}`, where `i` is the position of the log in `TestItemIndexData.logs`), all the clauses are combined under one
`bool.should`, so a found Test Item is scored by the combination of its logs, and the matched logs are aligned with
the request logs by the inner hits name.
"""

from abc import ABCMeta, abstractmethod
from typing import Any, Optional

from app.commons.model.db import Hit
from app.commons.model.launch_objects import SearchConfig
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.utils import utils

LOG_INNER_HITS_PREFIX = "log_"
INNER_HITS_SIZE = 3
DEFAULT_ITEM_QUERY_SIZE = 20
MIN_TERMS_PER_LOG = 10

TEST_ITEM_FIELDS_BOOST_SCORES = [
    ("test_item_name", utils.BOOST_SUPPORTING),
]

ITEM_LOG_SOURCE_FIELDS = [
    "logs.log_id",
    "logs.log_order",
    "logs.log_level",
    "logs.message",
    "logs.message_extended",
    "logs.message_without_params_extended",
    "logs.message_without_params_and_brackets",
    "logs.detected_message",
    "logs.detected_message_with_numbers",
    "logs.detected_message_extended",
    "logs.detected_message_without_params_extended",
    "logs.detected_message_without_params_and_brackets",
    "logs.stacktrace",
    "logs.stacktrace_extended",
    "logs.only_numbers",
    "logs.potential_status_codes",
    "logs.found_exceptions",
    "logs.found_tests_and_methods",
    "logs.urls",
    "logs.message_params",
    "logs.whole_message",
]

TEST_ITEM_SOURCE_FIELDS = [
    "test_item_id",
    "test_item_name",
    "unique_id",
    "test_case_hash",
    "launch_id",
    "launch_name",
    "launch_number",
    "issue_type",
    "is_auto_analyzed",
    "start_time",
    "log_count",
    "issue_history",
    *ITEM_LOG_SOURCE_FIELDS,
]

INNER_HITS_SOURCE = [
    "logs.log_id",
    "logs.log_order",
    "logs.log_time",
    "logs.log_level",
    "logs.cluster_id",
    "logs.cluster_message",
    "logs.cluster_with_numbers",
    "logs.original_message",
    "logs.message",
    "logs.message_extended",
    "logs.message_without_params_extended",
    "logs.message_without_params_and_brackets",
    "logs.detected_message",
    "logs.detected_message_with_numbers",
    "logs.detected_message_extended",
    "logs.detected_message_without_params_extended",
    "logs.detected_message_without_params_and_brackets",
    "logs.stacktrace",
    "logs.stacktrace_extended",
    "logs.only_numbers",
    "logs.potential_status_codes",
    "logs.found_exceptions",
    "logs.found_exceptions_extended",
    "logs.found_tests_and_methods",
    "logs.urls",
    "logs.paths",
    "logs.message_params",
    "logs.whole_message",
]


def get_log_inner_hits_name(log_index: int) -> str:
    """Get the name of inner hits which hold found logs matched to the request log.

    :param log_index: Position of the request log in `TestItemIndexData.logs`
    :return: Inner hits name
    """
    return f"{LOG_INNER_HITS_PREFIX}{log_index}"


def extract_log_matches(hit: Hit[TestItemIndexData]) -> dict[int, list[Hit[LogData]]]:
    """Extract found logs matched to each request log from the named inner hits of a found Test Item.

    :param hit: Found Test Item hit
    :return: Request log position mapped to the found logs matched to it, best first; only matched request logs
    """
    matches: dict[int, list[Hit[LogData]]] = {}
    for name, group in (hit.inner_hits or {}).items():
        if not name.startswith(LOG_INNER_HITS_PREFIX):
            continue
        log_index = name[len(LOG_INNER_HITS_PREFIX) :]
        if not log_index.isdigit():
            continue
        raw_hits = (group or {}).get("hits", {}).get("hits", [])
        log_hits = [Hit[LogData].from_dict(raw_hit) for raw_hit in raw_hits]
        if log_hits:
            matches[int(log_index)] = log_hits
    return dict(sorted(matches.items()))


def best_log_match(hit: Hit[TestItemIndexData]) -> Optional[tuple[int, Hit[LogData]]]:
    """Find the best scored pair of a request log and a found log in a found Test Item.

    :param hit: Found Test Item hit
    :return: Request log position and the found log, or None if nothing matched
    """
    best: Optional[tuple[int, Hit[LogData]]] = None
    for log_index, log_hits in extract_log_matches(hit).items():
        top_hit = log_hits[0]
        if best is None or (top_hit.score or 0.0) > (best[1].score or 0.0):
            best = (log_index, top_hit)
    return best


def add_start_time_decay(main_query: dict[str, Any], start_time: Optional[str], decay: float) -> dict[str, Any]:
    """Wrap the query into a function score which decreases scores of Test Items started far from the given time.

    :param main_query: Query to wrap
    :param start_time: Start time of the request Test Item, the query is returned as is if empty
    :param decay: Decay rate per 7 days
    :return: Wrapped query
    """
    if not start_time:
        return main_query
    result: dict[str, Any] = {
        "size": main_query["size"],
        "sort": main_query["sort"],
        "track_total_hits": False,
        "query": {
            "function_score": {
                "query": main_query["query"],
                "functions": [
                    {
                        "exp": {
                            "start_time": {
                                "origin": start_time,
                                "scale": "7d",
                                "offset": "1d",
                                "decay": decay,
                            }
                        }
                    },
                    {"script_score": {"script": {"source": "0.6"}}},
                ],
                "score_mode": "max",
                "boost_mode": "multiply",
            }
        },
    }
    if "_source" in main_query:
        result["_source"] = main_query["_source"]
    return result


class ItemQueryBuilder(metaclass=ABCMeta):
    """Base builder of queries which search for Test Items similar to the request one by all its logs."""

    search_cfg: SearchConfig

    def __init__(self, search_cfg: SearchConfig) -> None:
        self.search_cfg = search_cfg

    @abstractmethod
    def choose_message_field(self, number_of_log_lines: int) -> str:
        """Choose the primary message field, which every found log must be similar by.

        :param number_of_log_lines: Number of log lines to use, -1 means all lines
        :return: Log field name
        """
        ...

    def _select_logs(self, request_item: TestItemIndexData, message_field: str) -> list[tuple[int, LogData]]:
        logs = [
            (log_index, log)
            for log_index, log in enumerate(request_item.logs or [])
            if (getattr(log, message_field, None) or "").strip()
        ]
        max_logs = max(1, self.search_cfg.ItemQueryTermsBudget // MIN_TERMS_PER_LOG)
        return logs[-max_logs:]

    def _get_terms_per_log(self, logs_number: int) -> int:
        terms_share = self.search_cfg.ItemQueryTermsBudget // max(logs_number, 1)
        return max(MIN_TERMS_PER_LOG, min(self.search_cfg.MaxQueryTerms, terms_share))

    def _build_log_clause(
        self,
        log_index: int,
        log: LogData,
        message_field: str,
        min_should_match: str,
        max_query_terms: int,
    ) -> dict[str, Any]:
        nested_must = [
            utils.build_more_like_this_query(
                min_should_match,
                (getattr(log, message_field, None) or "").strip(),
                field_name=f"logs.{message_field}",
                boost=utils.BOOST_MESSAGE,
                max_query_terms=max_query_terms,
            )
        ]
        nested_should: list[dict[str, Any]] = []
        found_exceptions = (log.found_exceptions or "").strip()
        if found_exceptions:
            nested_should.append(
                utils.build_more_like_this_query(
                    "1",
                    found_exceptions,
                    field_name="logs.found_exceptions",
                    boost=utils.BOOST_ERROR_IDENTITY,
                    override_min_should_match="1",
                    max_query_terms=max_query_terms,
                )
            )
        nested_should.extend(
            utils.build_status_codes_queries(
                log.potential_status_codes or "",
                field_name="logs.potential_status_codes",
                boost=utils.BOOST_ERROR_IDENTITY,
            )
        )
        nested_bool: dict[str, Any] = {"must": nested_must}
        if nested_should:
            nested_bool["should"] = nested_should
        return {
            "nested": {
                "path": "logs",
                "score_mode": "max",
                "query": {"bool": nested_bool},
                "inner_hits": {
                    "name": get_log_inner_hits_name(log_index),
                    "size": INNER_HITS_SIZE,
                    "_source": INNER_HITS_SOURCE,
                },
            }
        }

    def _build_item_should(self, request_item: TestItemIndexData) -> list[dict[str, Any]]:
        should: list[dict[str, Any]] = []
        if request_item.test_case_hash:
            should.append(
                {
                    "term": {
                        "test_case_hash": {
                            "value": request_item.test_case_hash,
                            "boost": abs(self.search_cfg.BoostTestCaseHash),
                        }
                    }
                }
            )
        for field, boost_score in TEST_ITEM_FIELDS_BOOST_SCORES:
            field_value = (getattr(request_item, field, None) or "").strip()
            if field_value:
                should.append(
                    utils.build_more_like_this_query(
                        "1",
                        field_value,
                        field_name=field,
                        boost=boost_score,
                        override_min_should_match="1",
                        max_query_terms=self.search_cfg.MaxQueryTerms,
                    )
                )
        return should

    def build(
        self,
        request_item: TestItemIndexData,
        *,
        number_of_log_lines: int,
        min_should_match: str,
        min_logs_to_match: str,
        filter_no_defect: bool,
        size: int = DEFAULT_ITEM_QUERY_SIZE,
        exclude_issue_type: str = "",
    ) -> dict[str, Any]:
        """Build a query to search for Test Items similar to the request one by all its logs.

        The query is returned without start time decay, so callers can add their constraints to the `bool` query
        first and wrap it with `add_start_time_decay` after.

        :param request_item: Request Test Item with logs
        :param number_of_log_lines: Number of log lines to use, -1 means all lines
        :param min_should_match: Minimum should match for the primary message field of every log, e.g. "80%"
        :param min_logs_to_match: Minimum number of request logs a found Test Item must match, e.g. "1" or "100%"
        :param filter_no_defect: Whether to exclude "No Defect" Test Items along with "To Investigate" ones
        :param size: Number of Test Items to return
        :param exclude_issue_type: Issue type to exclude from results
        :return: OpenSearch query, or an empty dict if the request Test Item has no logs to search by
        """
        message_field = self.choose_message_field(number_of_log_lines)
        logs = self._select_logs(request_item, message_field)
        if not logs:
            return {}
        max_query_terms = self._get_terms_per_log(len(logs))
        log_clauses = [
            self._build_log_clause(log_index, log, message_field, min_should_match, max_query_terms)
            for log_index, log in logs
        ]

        must_not = utils.prepare_restrictions_by_issue_type(filter_no_defect=filter_no_defect)
        must_not.append({"term": {"test_item_id": str(request_item.test_item_id)}})
        if exclude_issue_type:
            must_not.append({"term": {"issue_type": exclude_issue_type}})

        query: dict[str, Any] = {
            "size": size,
            "sort": ["_score", {"start_time": "desc"}],
            "_source": TEST_ITEM_SOURCE_FIELDS,
            "query": {
                "bool": {
                    "filter": [{"exists": {"field": "issue_type"}}],
                    "must_not": must_not,
                    "must": [{"bool": {"should": log_clauses, "minimum_should_match": min_logs_to_match}}],
                    "should": self._build_item_should(request_item),
                }
            },
        }
        utils.append_aa_ma_boosts(query, self.search_cfg)
        return query


class AutoAnalysisQueryBuilder(ItemQueryBuilder):
    """Query builder for Auto-analysis and Auto-analysis model training."""

    def choose_message_field(self, number_of_log_lines: int) -> str:
        return "detected_message_without_params_extended" if number_of_log_lines == -1 else "message"


class SuggestQueryBuilder(ItemQueryBuilder):
    """Query builder for suggestions and suggestion model training."""

    def choose_message_field(self, number_of_log_lines: int) -> str:
        return "detected_message_extended" if number_of_log_lines == -1 else "message_extended"
