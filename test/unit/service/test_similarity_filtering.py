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
from unittest import mock

import pytest

from app.commons.model.db import Hit
from app.commons.model.launch_objects import AnalyzerConf, Launch, Log, TestItem, TestItemInfo
from app.commons.model.test_item_index import TestItemIndexData
from app.commons.similarity_filter import SHORTENED_MESSAGE_FIELDS, UNSHORTENED_MESSAGE_FIELDS
from app.service.auto_analyzer_service import (
    AutoAnalyzerService,
    choose_fields_to_filter_strict,
    prepare_request_items_for_launch,
)
from app.service.suggest_service import SuggestService
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG

REQUEST_MESSAGE = (
    "@pytest.mark.issue(issue_id='ABC-1234', reason='some_bug', issue_type='PB')\n    def test_issue_id():\n"
    ">       assert False\nE       assert False\n\ntests/test_issue_id.py:19: AssertionError"
)
UNRELATED_MESSAGE = (
    "('/Users/user/work/git/examples-python/pytest/tests/test_skipped.py', 27, "
    "'Skipped: no way of currently testing this')"
)
DATABASE_MESSAGE = "java.sql.SQLException: Connection refused by the database host db01"


def build_launch(test_item_id: int, messages: list[str], analyzer_config: AnalyzerConf) -> Launch:
    logs = [
        Log(logId=test_item_id * 10 + idx, logLevel=40000, message=message) for idx, message in enumerate(messages)
    ]
    return Launch(
        launchId=test_item_id,
        project=1,
        launchName="PyTest",
        analyzerConfig=analyzer_config,
        testItems=[
            TestItem(testItemId=test_item_id, uniqueId=f"auto:{test_item_id}", isAutoAnalyzed=False, logs=logs)
        ],
    )


def prepare_item(test_item_id: int, messages: list[str], analyzer_config: AnalyzerConf) -> TestItemIndexData:
    return prepare_request_items_for_launch(build_launch(test_item_id, messages, analyzer_config))[0]


def build_found_hit(
    test_item_id: int, messages: list[str], analyzer_config: AnalyzerConf, matched_logs: dict[int, int]
) -> Hit[TestItemIndexData]:
    """Build a found Test Item hit, `matched_logs` maps a request log position to a found log position."""
    found_item = prepare_item(test_item_id, messages, analyzer_config)
    found_logs = found_item.logs or []
    source: dict[str, Any] = found_item.model_dump()
    source["issue_type"] = "ab001"
    return Hit[TestItemIndexData].from_dict(
        {
            "_id": str(test_item_id),
            "_score": 1.0,
            "_source": source,
            "inner_hits": {
                f"log_{request_idx}": {
                    "hits": {
                        "hits": [{"_id": found_logs[found_idx].log_id, "_source": found_logs[found_idx].model_dump()}]
                    }
                }
                for request_idx, found_idx in matched_logs.items()
            },
        }
    )


def get_ids(hits: list[Hit[TestItemIndexData]]) -> list[str]:
    return [hit.source.test_item_id for hit in hits]


@pytest.mark.parametrize(
    "number_of_log_lines, min_similarity, expected_fields",
    [
        (-1, 0.8, UNSHORTENED_MESSAGE_FIELDS),
        (2, 0.8, SHORTENED_MESSAGE_FIELDS),
        (2, 1.0, SHORTENED_MESSAGE_FIELDS + ["found_tests_and_methods"]),
    ],
)
def test_choose_fields_to_filter_strict(number_of_log_lines: int, min_similarity: float, expected_fields: list[str]):
    assert choose_fields_to_filter_strict(number_of_log_lines, min_similarity) == expected_fields


@pytest.mark.parametrize("number_of_log_lines", [-1, 2])
def test_auto_analysis_removes_not_similar_found_items(number_of_log_lines: int):
    analyzer_config = AnalyzerConf(numberOfLogLines=number_of_log_lines, minShouldMatch=80)
    launch = build_launch(1, [REQUEST_MESSAGE], analyzer_config)
    request_items = prepare_request_items_for_launch(launch)
    hits = [
        build_found_hit(2, [UNRELATED_MESSAGE], analyzer_config, {0: 0}),
        build_found_hit(3, [REQUEST_MESSAGE], analyzer_config, {0: 0}),
    ]
    os_client = mock.Mock()
    os_client.msearch_grouped.return_value = iter([hits])
    service = AutoAnalyzerService(mock.Mock(), APP_CONFIG, DEFAULT_SEARCH_CONFIG, os_client=os_client)

    results = service._query_candidates_for_launch(launch, request_items)

    os_client.msearch_grouped.assert_called_once()
    assert len(results) == 1
    assert get_ids(results[0][1]) == ["3"]


@pytest.mark.parametrize("all_messages_should_match, expected_ids", [(False, ["2", "3"]), (True, ["3"])])
def test_auto_analysis_all_messages_should_match(all_messages_should_match: bool, expected_ids: list[str]):
    analyzer_config = AnalyzerConf(numberOfLogLines=-1, allMessagesShouldMatch=all_messages_should_match)
    messages = [REQUEST_MESSAGE, DATABASE_MESSAGE]
    launch = build_launch(1, messages, analyzer_config)
    request_items = prepare_request_items_for_launch(launch)
    hits = [
        build_found_hit(2, [REQUEST_MESSAGE, UNRELATED_MESSAGE], analyzer_config, {0: 0, 1: 1}),
        build_found_hit(3, messages, analyzer_config, {0: 0, 1: 1}),
    ]
    os_client = mock.Mock()
    os_client.msearch_grouped.return_value = iter([hits])
    service = AutoAnalyzerService(mock.Mock(), APP_CONFIG, DEFAULT_SEARCH_CONFIG, os_client=os_client)

    results = service._query_candidates_for_launch(launch, request_items)

    assert get_ids(results[0][1]) == expected_ids


def build_suggest_service(hits: list[Hit[TestItemIndexData]]) -> tuple[SuggestService, mock.Mock]:
    boosting_model = mock.Mock()
    boosting_model.feature_ids = [2, 3]
    boosting_model.is_custom = False
    boosting_model.get_model_info.return_value = ["global boosting model"]
    boosting_model.predict.side_effect = lambda data: ([1] * len(data), [[0.1, 0.9]] * len(data))
    defect_type_model = mock.Mock()
    defect_type_model.get_model_info.return_value = ["global defect type model"]
    model_chooser = mock.Mock()
    model_chooser.choose_model.side_effect = [boosting_model, defect_type_model]
    os_client = mock.Mock()
    os_client.msearch_grouped.return_value = iter([hits])
    return SuggestService(model_chooser, APP_CONFIG, DEFAULT_SEARCH_CONFIG, os_client=os_client), boosting_model


def build_test_item_info(analyzer_config: AnalyzerConf) -> TestItemInfo:
    return TestItemInfo(
        testItemId=1,
        uniqueId="auto:1",
        launchId=1,
        launchName="PyTest",
        project=1,
        analyzerConfig=analyzer_config,
        logs=[Log(logId=10, logLevel=40000, message=REQUEST_MESSAGE)],
    )


@pytest.mark.parametrize("number_of_log_lines", [-1, 2])
def test_suggestions_are_not_made_by_not_similar_found_items(number_of_log_lines: int):
    analyzer_config = AnalyzerConf(numberOfLogLines=number_of_log_lines, analyzerMode="LAUNCH_NAME")
    hits = [build_found_hit(2, [UNRELATED_MESSAGE], analyzer_config, {0: 0})]
    service, boosting_model = build_suggest_service(hits)

    results = service.suggest_items(build_test_item_info(analyzer_config))

    assert results == []
    boosting_model.predict.assert_not_called()


def test_suggestions_are_made_by_similar_found_items():
    analyzer_config = AnalyzerConf(numberOfLogLines=2, analyzerMode="LAUNCH_NAME")
    hits = [
        build_found_hit(2, [UNRELATED_MESSAGE], analyzer_config, {0: 0}),
        build_found_hit(3, [REQUEST_MESSAGE], analyzer_config, {0: 0}),
    ]
    service, boosting_model = build_suggest_service(hits)

    results = service.suggest_items(build_test_item_info(analyzer_config))

    boosting_model.predict.assert_called_once()
    assert len(boosting_model.predict.call_args.args[0]) == 1
    assert [result.relevantItem for result in results] == [3]
    assert [result.relevantLogId for result in results] == [30]
