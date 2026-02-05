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
from typing import Any

from commons.model import LogData, LogItemIndexData, TestItemIndexData
from commons.model.db import Hit
from ml.training import normalize_issue_type
from utils import text_processing


def _safe_int(value: Any) -> int:
    """
    Safely cast a value to integer.

    :param value: Value to cast
    :return: Integer value or 0 on failure
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def convert_test_item_log(
    test_item: TestItemIndexData,
    log_data: LogData,
    issue_type: str = "",
) -> LogItemIndexData:
    """
    Convert Test Item-centric log data to log-centric model for ML featurizers.

    :param test_item: Source Test Item document
    :param log_data: Nested log entry data
    :param issue_type: Optional issue type override for the log
    :return: LogItemIndexData instance
    """
    resolved_issue_type = issue_type or normalize_issue_type(test_item.issue_type)
    return LogItemIndexData(
        log_id=str(log_data.log_id or ""),
        test_item=_safe_int(test_item.test_item_id),
        test_item_name=test_item.test_item_name or "",
        test_case_hash=_safe_int(test_item.test_case_hash),
        unique_id=test_item.unique_id or "",
        launch_id=_safe_int(test_item.launch_id),
        launch_name=test_item.launch_name or "",
        issue_type=resolved_issue_type,
        is_auto_analyzed=bool(test_item.is_auto_analyzed),
        start_time=test_item.start_time or "",
        log_time=log_data.log_time or "",
        log_level=log_data.log_level or 0,
        is_merged=False,
        merged_small_logs="",
        message=log_data.message or "",
        detected_message=log_data.detected_message or "",
        detected_message_with_numbers=log_data.detected_message_with_numbers or "",
        detected_message_extended=log_data.detected_message_extended or "",
        detected_message_without_params_extended=log_data.detected_message_without_params_extended or "",
        detected_message_without_params_and_brackets=log_data.detected_message_without_params_and_brackets or "",
        stacktrace=log_data.stacktrace or "",
        stacktrace_extended=log_data.stacktrace_extended or "",
        message_extended=log_data.message_extended or "",
        message_without_params_extended=log_data.message_without_params_extended or "",
        message_without_params_and_brackets=log_data.message_without_params_and_brackets or "",
        message_params=log_data.message_params or "",
        only_numbers=log_data.only_numbers or "",
        found_exceptions=log_data.found_exceptions or "",
        found_tests_and_methods=log_data.found_tests_and_methods or "",
        potential_status_codes=log_data.potential_status_codes or "",
        urls=log_data.urls or "",
        original_message=log_data.original_message or "",
        whole_message=log_data.whole_message or "",
        cluster_id=log_data.cluster_id or "",
        cluster_message=log_data.cluster_message or "",
        cluster_with_numbers=bool(log_data.cluster_with_numbers),
    )


def get_request_logs(test_item: TestItemIndexData, issue_type: str) -> list[LogItemIndexData]:
    logs = list(test_item.logs or [])
    if not logs:
        return []
    logs_sorted = sorted(
        logs,
        key=lambda log: log.log_order if log.log_order is not None else _safe_int(log.log_id),
    )
    return [convert_test_item_log(test_item, log_data, issue_type=issue_type) for log_data in logs_sorted]


def _get_log_text(log_item: LogItemIndexData) -> str:
    """
    Build a text payload for similarity comparison.

    :param log_item: Log item to extract text from
    :return: Combined text for similarity matching
    """
    parts = [
        log_item.whole_message,
        log_item.message,
        log_item.detected_message,
        log_item.stacktrace,
    ]
    return " ".join([part.strip() for part in parts if part and part.strip()])


def bucket_sort_logs_by_similarity(
    request_logs: list[LogItemIndexData],
    found_hits: list[Hit[LogItemIndexData]],
) -> list[list[Hit[LogItemIndexData]]]:
    """
    Align found logs to the most similar request logs using bucket sorting.

    :param request_logs: Log items used as search requests
    :param found_hits: Log hits retrieved from OpenSearch
    :return: Buckets aligned with request logs
    """
    buckets: list[list[Hit[LogItemIndexData]]] = [[] for _ in request_logs]
    request_texts = [_get_log_text(log_item) for log_item in request_logs]
    if not request_texts:
        return buckets
    my_vectorizer = None
    for hit in found_hits:
        hit_text = _get_log_text(hit.source)
        if not hit_text.strip():
            continue
        similarities, my_vectorizer = text_processing.calculate_text_similarity(
            hit_text, request_texts, vectorizer=my_vectorizer
        )
        if not similarities:
            continue
        best_idx = max(range(len(similarities)), key=lambda idx: similarities[idx].similarity)
        buckets[best_idx].append(hit)
    return buckets
