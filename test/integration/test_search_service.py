#  Copyright 2025 EPAM Systems
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

from unittest import mock

import pytest

from app.commons.model.launch_objects import AnalyzerConf, SearchLogs
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.os_client import Hit, OsClient
from app.service.search_service import SearchService
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG


def _make_log_data(log_id: str, log_order: int, message: str) -> LogData:
    return LogData(
        log_id=log_id,
        log_order=log_order,
        log_time="2025-01-01 00:00:00",
        log_level=40000,
        cluster_id="",
        cluster_message="",
        cluster_with_numbers=False,
        original_message=message,
        message=message,
        message_lines=1,
        message_words_number=2,
        message_extended=message,
        message_without_params_extended=message,
        message_without_params_and_brackets=message,
        detected_message=message,
        detected_message_with_numbers=message,
        detected_message_extended=message,
        detected_message_without_params_extended=message,
        detected_message_without_params_and_brackets=message,
        stacktrace="",
        stacktrace_extended="",
        only_numbers="",
        potential_status_codes="",
        found_exceptions="",
        found_exceptions_extended="",
        found_tests_and_methods="",
        urls="",
        paths="",
        message_params="",
        whole_message=message,
    )


@pytest.fixture
def mocked_os_client() -> OsClient:
    return mock.Mock(spec=OsClient)


@pytest.fixture
def search_service(mocked_os_client: OsClient) -> SearchService:
    return SearchService(APP_CONFIG, DEFAULT_SEARCH_CONFIG, os_client=mocked_os_client)


def test_search_logs_builds_nested_query(search_service: SearchService, mocked_os_client: OsClient) -> None:
    search_request = SearchLogs(
        launchId=1001,
        launchName="Test Launch",
        itemId=2001,
        projectId=123,
        filteredLaunchIds=[1001, 1002],
        logMessages=["login failed error", "database timeout error"],
        analyzerConfig=AnalyzerConf(searchScoreMode="sum"),
        logLines=5,
    )

    mocked_os_client.search.return_value = iter([])

    result = search_service.search_logs(search_request)

    mocked_os_client.search.assert_called_once()
    _, query = mocked_os_client.search.call_args[0]
    nested_query = query["query"]["bool"]["must"][1]["nested"]

    assert nested_query["path"] == "logs"
    assert nested_query["score_mode"] == "sum"

    should_queries = nested_query["query"]["bool"]["should"]
    assert should_queries, "nested query should include more_like_this clauses"
    assert "logs.message" in should_queries[0]["more_like_this"]["fields"]
    assert result == []


def test_search_logs_filters_and_returns_best_log(search_service: SearchService, mocked_os_client: OsClient) -> None:
    analyzer_config = AnalyzerConf(searchLogsMinShouldMatch=30)
    search_request = SearchLogs(
        launchId=1001,
        launchName="Test Launch",
        itemId=2001,
        projectId=123,
        filteredLaunchIds=[1001],
        logMessages=["login failed error", "user authentication failure"],
        analyzerConfig=analyzer_config,
        logLines=5,
    )

    item_one_logs = [
        _make_log_data("501", 0, "login failed error"),
        _make_log_data("502", 1, "database timeout error"),
    ]
    item_two_logs = [_make_log_data("601", 0, "unrelated content")]

    item_one = TestItemIndexData(
        test_item_id="3001",
        launch_id="1001",
        logs=item_one_logs,
        issue_type="ti001",
    )
    item_two = TestItemIndexData(
        test_item_id="3002",
        launch_id="1001",
        logs=item_two_logs,
        issue_type="ti001",
    )

    mocked_os_client.search.return_value = iter(
        [
            Hit[TestItemIndexData].from_dict({"_index": "rp_123", "_id": "3001", "_source": item_one.model_dump()}),
            Hit[TestItemIndexData].from_dict({"_index": "rp_123", "_id": "3002", "_source": item_two.model_dump()}),
        ]
    )

    result = search_service.search_logs(search_request)

    assert len(result) == 1
    assert result[0].testItemId == 3001
    assert result[0].logId == 501
    assert result[0].matchScore == 100
