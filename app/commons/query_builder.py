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
from app.commons.model.launch_objects import Launch, SearchConfig, TestItemInfo
from app.commons.model.test_item_index import LogData, TestItemIndexData

LOGS_FOUND_EXCEPTIONS_FIELD = "logs.found_exceptions"

POTENTIAL_STATUS_CODES_FIELD = "logs.potential_status_codes"

ERROR_HASH_FIELD = "error_hash"

LOG_INNER_HITS_PREFIX = "log_"
INNER_HITS_SIZE = 3
DEFAULT_ITEM_QUERY_SIZE = 20
MIN_TERMS_PER_LOG = 10

# Boost ladder for search clauses. Values are relative to each other only: a clause boosted
# BOOST_ERROR_IDENTITY contributes roughly twice what BOOST_MESSAGE does, and so on.
BOOST_NEUTRAL = 1.0
"""No emphasis. For a clause that carries the query on its own, so its weight cannot rank anything."""

BOOST_SUPPORTING = 2.0
"""A field that corroborates a match without identifying it: numbers, params, URLs, paths."""

BOOST_MESSAGE = 4.0
"""The primary message field a candidate is matched on."""

BOOST_ERROR_IDENTITY = 8.0
"""A field that nearly identifies the failure: exception names, status codes."""

CONTAINED_SEQUENCE_BOOST_FACTOR = 0.5
"""Share of the boost given to a sequence found inside a longer one rather than matched whole."""


TEST_ITEM_FIELDS_BOOST_SCORES = [
    ("test_item_name", BOOST_SUPPORTING),
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
    POTENTIAL_STATUS_CODES_FIELD,
    LOGS_FOUND_EXCEPTIONS_FIELD,
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

# Fields of a Test Item found by its error hash: what an analysis result is made of, plus the required model fields
ERROR_HASH_SOURCE_FIELDS = ["test_item_id", "launch_id", "issue_type", "test_case_hash", "start_time"]

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
    POTENTIAL_STATUS_CODES_FIELD,
    LOGS_FOUND_EXCEPTIONS_FIELD,
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


def parse_log_inner_hits_name(name: str) -> Optional[int]:
    """Get the position of the request log from the name of inner hits which hold found logs matched to it.

    :param name: Inner hits name
    :return: Position of the request log in `TestItemIndexData.logs`, or None if these are not log inner hits
    """
    if not name.startswith(LOG_INNER_HITS_PREFIX):
        return None
    log_index = name[len(LOG_INNER_HITS_PREFIX) :]
    if not log_index.isdigit():
        return None
    return int(log_index)


def extract_log_matches(hit: Hit[TestItemIndexData]) -> dict[int, list[Hit[LogData]]]:
    """Extract found logs matched to each request log from the named inner hits of a found Test Item.

    :param hit: Found Test Item hit
    :return: Request log position mapped to the found logs matched to it, best first; only matched request logs
    """
    matches: dict[int, list[Hit[LogData]]] = {}
    for name, group in (hit.inner_hits or {}).items():
        log_index = parse_log_inner_hits_name(name)
        if log_index is None:
            continue
        raw_hits = (group or {}).get("hits", {}).get("hits", [])
        log_hits = [Hit[LogData].from_dict(raw_hit) for raw_hit in raw_hits]
        if log_hits:
            matches[log_index] = log_hits
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


def create_path(query: dict, path: tuple[str, ...], value: Any) -> Any:
    """Create path in a dictionary and assign passed value on the last element in path."""
    path_length = len(path)
    last_element = path[path_length - 1]
    current_node = query
    for i in range(path_length - 1):
        element = path[i]
        if element not in current_node:
            current_node[element] = {}
        current_node = current_node[element]
    if last_element not in current_node:
        current_node[last_element] = value
    return current_node[last_element]


def append_aa_ma_boosts(query: dict[str, Any], search_cfg: SearchConfig) -> None:
    """Append boosts for auto-analyzed and manually-analyzed fields to ES/OS query.

    :param query: ES/OS query
    :param search_cfg: Search configuration
    """
    should = create_path(query, ("query", "bool", "should"), [])
    boost_aa = search_cfg.BoostAA
    boost_ma = search_cfg.BoostMA
    if boost_aa > boost_ma:
        should.append(
            {
                "term": {
                    "is_auto_analyzed": {
                        "value": True,
                        "boost": boost_aa - boost_ma,
                    }
                }
            }
        )
    else:
        should.append(
            {
                "term": {
                    "is_auto_analyzed": {
                        "value": False,
                        "boost": boost_ma - boost_aa,
                    }
                }
            }
        )


def build_more_like_this_query(
    min_should_match: str,
    log_message,
    field_name: str = "message",
    boost: float = BOOST_NEUTRAL,
    override_min_should_match: Optional[str] = None,
    max_query_terms: int = 50,
):
    return {
        "more_like_this": {
            "fields": [field_name],
            "like": log_message,
            "min_doc_freq": 1,
            "min_term_freq": 1,
            "minimum_should_match": override_min_should_match or "5<" + min_should_match,
            "max_query_terms": max_query_terms,
            "boost": boost,
        }
    }


def build_status_codes_exact_query(
    status_codes: str, field_name: str = "potential_status_codes", boost: float = BOOST_NEUTRAL
) -> Optional[dict[str, Any]]:
    """Build a clause matching the exact status code sequence.

    `potential_status_codes` stores the codes as they appeared in the message, in order and
    without de-duplication, so "expected status code 400, but was 401" is stored as `400 401`
    while the inverse failure is stored as `401 400`. Those are different failures and must not
    match each other, which rules out both `more_like_this` and `terms`: they test term overlap
    and score the two identically.

    :param status_codes: Status codes as stored in the field, whitespace separated
    :param field_name: Field holding the codes; its `.exact` sub-field is matched
    :param boost: Boost for the clause
    :return: A term query on the `.exact` sub-field, or None when there are no codes
    """
    codes = status_codes.strip()
    if not codes:
        return None
    return {"term": {f"{field_name}.exact": {"value": codes, "boost": boost}}}


def build_status_codes_queries(
    status_codes: str, field_name: str = "potential_status_codes", boost: float = BOOST_NEUTRAL
) -> list[dict[str, Any]]:
    """Build clauses that rank status code sequences by how exactly they match.

    The exact sequence scores highest; the same sequence occurring inside a longer one still
    matches, at half the boost. Order is preserved in both, unlike `more_like_this`.

    :param status_codes: Status codes as stored in the field, whitespace separated
    :param field_name: Field holding the codes
    :param boost: Boost for an exact match; a contained match gets CONTAINED_SEQUENCE_BOOST_FACTOR of it
    :return: Clauses to add to a `should`, empty when there are no codes
    """
    exact = build_status_codes_exact_query(status_codes, field_name=field_name, boost=boost)
    if not exact:
        return []
    codes = status_codes.strip()
    contained_boost = boost * CONTAINED_SEQUENCE_BOOST_FACTOR
    return [exact, {"match_phrase": {field_name: {"query": codes, "boost": contained_boost}}}]


def build_status_codes_filter(status_codes: str, field_name: str = "potential_status_codes") -> dict[str, Any]:
    """Build a filter clause which leaves only logs with exactly the same status code sequence.

    The sequence is compared as a whole, see `build_status_codes_exact_query`. A log without codes matches only logs
    without codes, so a cut log which lost its "expected 409, but was 404" line does not match the full one.

    :param status_codes: Status codes as stored in the field, whitespace separated
    :param field_name: Field holding the codes; its `.exact` sub-field is matched
    :return: A term query on the `.exact` sub-field, or a query for logs with no codes when there are no codes
    """
    exact = build_status_codes_exact_query(status_codes, field_name=field_name)
    if exact:
        return exact
    # Logs without codes may store an empty string as well as no value at all, "?*" matches only non-empty values.
    return {"bool": {"must_not": [{"wildcard": {f"{field_name}.exact": {"value": "?*"}}}]}}


def prepare_restrictions_by_issue_type(filter_no_defect: bool = True) -> list[dict[str, Any]]:
    issue_type_filter = [{"wildcard": {"issue_type": {"value": "ti*", "case_insensitive": True}}}]
    if filter_no_defect:
        issue_type_filter.append({"wildcard": {"issue_type": {"value": "nd*", "case_insensitive": True}}})
    return issue_type_filter


def add_launch_name_boost(query: dict[str, Any], launch_name: str, launch_boost: float) -> None:
    should = create_path(query, ("query", "bool", "should"), [])
    should.append({"term": {"launch_name": {"value": launch_name, "boost": launch_boost}}})


def add_launch_id_boost(query: dict[str, Any], launch_id: int | str, launch_boost: float) -> None:
    should = create_path(query, ("query", "bool", "should"), [])
    should.append({"term": {"launch_id": {"value": launch_id, "boost": launch_boost}}})


def add_launch_name_and_id_boost(
    query: dict[str, Any], launch_name: str, launch_id: int | str, launch_boost: float
) -> None:
    add_launch_id_boost(query, launch_id, launch_boost)
    add_launch_name_boost(query, launch_name, launch_boost)


def add_constraints_for_launches_into_query(query: dict, launch: Launch, launch_boost: float) -> dict:
    previous_launch_id = getattr(launch, "previousLaunchId", 0) or 0
    previous_launch_id = int(previous_launch_id)
    analyzer_mode = launch.analyzerConfig.analyzerMode
    launch_name = launch.launchName
    launch_id = launch.launchId
    if analyzer_mode == "LAUNCH_NAME":
        # Previous launches with the same name
        must = create_path(query, ("query", "bool", "must"), [])
        must_not = create_path(query, ("query", "bool", "must_not"), [])
        must.append({"term": {"launch_name": launch_name}})
        must_not.append({"term": {"launch_id": launch_id}})
    elif analyzer_mode == "CURRENT_AND_THE_SAME_NAME":
        # All launches with the same name
        must = create_path(query, ("query", "bool", "must"), [])
        must.append({"term": {"launch_name": launch_name}})
        add_launch_id_boost(query, launch_id, launch_boost)
    elif analyzer_mode == "CURRENT_LAUNCH":
        # Just current launch
        must = create_path(query, ("query", "bool", "must"), [])
        must.append({"term": {"launch_id": launch_id}})
    elif analyzer_mode == "PREVIOUS_LAUNCH":
        # Just previous launch
        must = create_path(query, ("query", "bool", "must"), [])
        must.append({"term": {"launch_id": previous_launch_id}})
    elif analyzer_mode == "ALL":
        # All previous launches
        must_not = create_path(query, ("query", "bool", "must_not"), [])
        must_not.append({"term": {"launch_id": launch_id}})
    else:
        # Boost launches with the same name and ID, but do not ignore any
        add_launch_name_and_id_boost(query, launch_name, launch_id, launch_boost)
    return query


def add_constraints_for_launches_into_query_suggest(
    query: dict, test_item_info: TestItemInfo, launch_boost: float
) -> dict:
    previous_launch_id = getattr(test_item_info, "previousLaunchId", 0) or 0
    previous_launch_id = int(previous_launch_id)
    analyzer_mode = test_item_info.analyzerConfig.analyzerMode
    launch_name = test_item_info.launchName
    launch_id = test_item_info.launchId
    if analyzer_mode in {"LAUNCH_NAME", "ALL"}:
        # Previous launches with the same name
        add_launch_name_boost(query, launch_name, launch_boost)
        should = create_path(query, ("query", "bool", "should"), [])
        should.append({"term": {"launch_id": {"value": launch_id, "boost": 1 / launch_boost}}})
    elif analyzer_mode == "PREVIOUS_LAUNCH":
        # Just previous launch
        if previous_launch_id:
            add_launch_id_boost(query, previous_launch_id, launch_boost)
    else:
        # For:
        # * CURRENT_LAUNCH
        # * CURRENT_AND_THE_SAME_NAME
        # Boost launches with the same name, but do not ignore any
        add_launch_name_and_id_boost(query, launch_name, launch_id, launch_boost)
    return query


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
    filter_by_status_codes: bool = False
    """Whether found logs must have exactly the same status codes as the request log, not only be boosted by them."""

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

    def select_log_indices(self, request_item: TestItemIndexData, number_of_log_lines: int) -> list[int]:
        """Get positions of the request logs the query searches by.

        :param request_item: Request Test Item with logs
        :param number_of_log_lines: Number of log lines to use, -1 means all lines
        :return: Positions of the logs in `TestItemIndexData.logs`
        """
        message_field = self.choose_message_field(number_of_log_lines)
        return [log_index for log_index, _ in self._select_logs(request_item, message_field)]

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
            build_more_like_this_query(
                min_should_match,
                (getattr(log, message_field, None) or "").strip(),
                field_name=f"logs.{message_field}",
                boost=BOOST_MESSAGE,
                max_query_terms=max_query_terms,
            )
        ]
        nested_should: list[dict[str, Any]] = []
        found_exceptions = (log.found_exceptions or "").strip()
        if found_exceptions:
            nested_should.append(
                build_more_like_this_query(
                    "1",
                    found_exceptions,
                    field_name=LOGS_FOUND_EXCEPTIONS_FIELD,
                    boost=BOOST_ERROR_IDENTITY,
                    override_min_should_match="1",
                    max_query_terms=max_query_terms,
                )
            )
        nested_should.extend(
            build_status_codes_queries(
                log.potential_status_codes or "",
                field_name=POTENTIAL_STATUS_CODES_FIELD,
                boost=BOOST_ERROR_IDENTITY,
            )
        )
        nested_bool: dict[str, Any] = {"must": nested_must}
        if self.filter_by_status_codes:
            # Logs with different status codes are different failures, however similar their texts are. A found Test
            # Item which has no logs left after this filter does not match the clause.
            nested_bool["filter"] = [
                build_status_codes_filter(log.potential_status_codes or "", field_name=POTENTIAL_STATUS_CODES_FIELD)
            ]
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
                    build_more_like_this_query(
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

        must_not = prepare_restrictions_by_issue_type(filter_no_defect=filter_no_defect)
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
        append_aa_ma_boosts(query, self.search_cfg)
        return query


class AutoAnalysisQueryBuilder(ItemQueryBuilder):
    """Query builder for Auto-analysis and Auto-analysis model training."""

    filter_by_status_codes = True

    def choose_message_field(self, number_of_log_lines: int) -> str:
        return "detected_message_without_params_extended" if number_of_log_lines == -1 else "message"


class SuggestQueryBuilder(ItemQueryBuilder):
    """Query builder for suggestions and suggestion model training."""

    def choose_message_field(self, number_of_log_lines: int) -> str:
        return "detected_message_extended" if number_of_log_lines == -1 else "message_extended"


class ErrorHashQueryBuilder:
    """Builder of queries which find the latest analyzed Test Items failed with the same error.

    Test Items are matched by their error hash exactly, see `app.commons.error_hash`, and sorted by start time, newest
    first. Only analyzed Test Items are found: "To Investigate" ones are skipped, the analyzer mode restricts launches
    the same way as for the "more like this" query.
    """

    search_cfg: SearchConfig
    launch_boost: float

    def __init__(self, search_cfg: SearchConfig) -> None:
        """Initialize the builder.

        :param search_cfg: Search configuration
        """
        self.search_cfg = search_cfg
        self.launch_boost = abs(search_cfg.BoostLaunch)

    def build(
        self, launch: Launch, request_item: TestItemIndexData, size: int, same_test_case: bool = True
    ) -> dict[str, Any]:
        """Build a query to find Test Items failed with the same error as the request one.

        :param launch: Launch being analyzed
        :param request_item: Prepared request Test Item
        :param size: Number of the latest Test Items to find
        :param same_test_case: Whether to find only runs of the same test case
        :return: OpenSearch query, or an empty dict if the request Test Item has no error hash, or no test case hash
                 when runs of the same test case are requested
        """
        if not request_item.error_hash:
            return {}
        filters: list[dict[str, Any]] = [{"term": {ERROR_HASH_FIELD: request_item.error_hash}}]
        if same_test_case:
            if not request_item.test_case_hash:
                return {}
            filters.append({"term": {"test_case_hash": request_item.test_case_hash}})
        query: dict[str, Any] = {
            "size": size,
            "_source": ERROR_HASH_SOURCE_FIELDS,
            "sort": [{"start_time": {"order": "desc"}}, {"test_item_id": {"order": "desc"}}],
            "query": {
                "bool": {
                    "filter": filters,
                    "must_not": prepare_restrictions_by_issue_type(filter_no_defect=False),
                }
            },
        }
        return add_constraints_for_launches_into_query(query, launch, self.launch_boost)
