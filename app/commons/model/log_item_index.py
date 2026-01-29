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
"""Log entry-centric index data model for OpenSearch storage."""
from typing import Any

from pydantic import BaseModel, Field

from app.commons.model.db import Hit


class LogItemIndexData(BaseModel):
    """
    Log entry-centric document used during migration to Test Item-centric indexing.

    Contains both log-level fields and Test Item metadata required by ML featurizers.
    """

    log_id: str = Field(default="", description="Unique identifier for the log entry")
    test_item: int = Field(default=0, description="Test Item identifier")
    test_item_name: str = Field(default="", description="Preprocessed test item name")
    test_case_hash: int = Field(default=0, description="Test case hash")
    unique_id: str = Field(default="", description="Unique ID from ReportPortal")
    launch_id: int = Field(default=0, description="Launch identifier")
    launch_name: str = Field(default="", description="Launch name")
    issue_type: str = Field(default="", description="Issue type label")
    is_auto_analyzed: bool = Field(default=False, description="Whether assigned by auto-analysis")
    start_time: str = Field(default="", description="Test item start time")
    log_time: str = Field(default="", description="Log time")
    log_level: int = Field(default=0, description="Log level")
    is_merged: bool = Field(default=False, description="Whether log is a merged entry")
    merged_small_logs: str = Field(default="", description="Merged small logs text")
    message: str = Field(default="", description="Prepared log message")
    detected_message: str = Field(default="", description="Detected message without numbers")
    detected_message_with_numbers: str = Field(default="", description="Detected message with numbers")
    detected_message_extended: str = Field(default="", description="Detected message with enriched data")
    detected_message_without_params_extended: str = Field(
        default="", description="Detected message without params (enriched)"
    )
    detected_message_without_params_and_brackets: str = Field(
        default="", description="Detected message without params and brackets"
    )
    stacktrace: str = Field(default="", description="Stacktrace")
    stacktrace_extended: str = Field(default="", description="Enriched stacktrace")
    message_extended: str = Field(default="", description="Enriched message")
    message_without_params_extended: str = Field(default="", description="Message without params (enriched)")
    message_without_params_and_brackets: str = Field(default="", description="Message without params and brackets")
    message_params: str = Field(default="", description="Message parameters")
    only_numbers: str = Field(default="", description="Only numbers extracted from message")
    found_exceptions: str = Field(default="", description="Found exceptions")
    found_tests_and_methods: str = Field(default="", description="Found tests and methods")
    potential_status_codes: str = Field(default="", description="Potential status codes")
    urls: str = Field(default="", description="URLs extracted from message")
    original_message: str = Field(default="", description="Original raw message")
    whole_message: str = Field(default="", description="Combined exception message and stacktrace")
    cluster_id: str = Field(default="", description="Cluster identifier")
    cluster_message: str = Field(default="", description="Cluster message")
    cluster_with_numbers: bool = Field(default=False, description="Cluster message with numbers flag")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogItemIndexData":
        """
        Create LogItemIndexData instance from dictionary.

        :param data: Dictionary containing log item data
        :return: LogItemIndexData instance
        """
        return cls.model_validate(data)


def _extract_log_id(raw: dict[str, Any]) -> str:
    log_id = raw.get("_id")
    if log_id is None:
        log_id = raw.get("log_id")
    return "" if log_id is None else str(log_id)


def deserialize_log_item_source(raw: dict[str, Any]) -> LogItemIndexData:
    """
    Deserialize raw log document or _source payload into LogItemIndexData.

    :param raw: Raw log document or source payload
    :return: Parsed LogItemIndexData instance
    """
    source = raw.get("_source", raw)
    data = dict(source)
    log_id = _extract_log_id(raw)
    if log_id:
        data["log_id"] = log_id
    return LogItemIndexData.model_validate(data)


def deserialize_log_item_hit(raw_hit: dict[str, Any]) -> Hit[LogItemIndexData]:
    """
    Deserialize raw OpenSearch hit into typed Hit[LogItemIndexData].

    :param raw_hit: Raw hit dictionary
    :return: Parsed hit with LogItemIndexData source
    """
    if isinstance(raw_hit, Hit):
        return raw_hit
    source = raw_hit.get("_source", {})
    data = dict(source)
    log_id = _extract_log_id(raw_hit)
    if log_id:
        data["log_id"] = log_id
    return Hit[LogItemIndexData].model_validate({**raw_hit, "_source": data})


def deserialize_log_item_search_results(
    raw_results: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[tuple[LogItemIndexData, list[Hit[LogItemIndexData]]]]:
    """
    Deserialize raw search results into typed log item tuples.

    :param raw_results: List of (log_info, search_results) tuples
    :return: List of (LogItemIndexData, list[Hit[LogItemIndexData]]) tuples
    """
    parsed_results: list[tuple[LogItemIndexData, list[Hit[LogItemIndexData]]]] = []
    for log_info, search_result in raw_results:
        if isinstance(log_info, LogItemIndexData):
            log_item = log_info
        else:
            log_item = deserialize_log_item_source(log_info)
        if isinstance(search_result, list):
            raw_hits = search_result
        else:
            raw_hits = search_result.get("hits", {}).get("hits", [])
        parsed_hits = [deserialize_log_item_hit(hit) for hit in raw_hits]
        parsed_results.append((log_item, parsed_hits))
    return parsed_results
