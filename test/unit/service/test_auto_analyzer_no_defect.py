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

import json
from typing import Any, Optional
from unittest import mock

import pytest

from app.commons.model.db import Hit
from app.commons.model.launch_objects import AnalyzerConf, Launch, Log, TestItem
from app.commons.model.ml import ModelType
from app.commons.model.test_item_index import TestItemIndexData
from app.service.auto_analyzer_service import (
    ANALYSIS_QUERY_SIZE,
    AutoAnalyzerService,
    prepare_request_items_for_launch,
    split_no_defect_candidates,
)
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG

TEST_CASE_HASH = 123
OTHER_TEST_CASE_HASH = 456
REQUEST_MESSAGE = (
    "java.lang.AssertionError: Expected status code <200> but was <500>.\n"
    "\tat com.epam.reportportal.tests.api.LaunchTest.finishLaunch(LaunchTest.java:142)"
)


def build_launch(test_item_id: int, test_case_hash: int = TEST_CASE_HASH) -> Launch:
    return Launch(
        launchId=test_item_id,
        project=1,
        launchName="Launch",
        analyzerConfig=AnalyzerConf(),
        testItems=[
            TestItem(
                testItemId=test_item_id,
                uniqueId="auto:1",
                isAutoAnalyzed=False,
                testCaseHash=test_case_hash,
                logs=[Log(logId=test_item_id * 10, logLevel=40000, message=REQUEST_MESSAGE)],
            )
        ],
    )


def prepare_item(test_item_id: int, test_case_hash: int = TEST_CASE_HASH) -> TestItemIndexData:
    return prepare_request_items_for_launch(build_launch(test_item_id, test_case_hash))[0]


def build_found_hit(
    test_item_id: int, issue_type: str, start_time: str, test_case_hash: int = TEST_CASE_HASH
) -> Hit[TestItemIndexData]:
    found_item = prepare_item(test_item_id, test_case_hash)
    found_logs = found_item.logs or []
    source: dict[str, Any] = found_item.model_dump()
    source["issue_type"] = issue_type
    source["start_time"] = start_time
    return Hit[TestItemIndexData].from_dict(
        {
            "_id": str(test_item_id),
            "_score": 1.0,
            "_source": source,
            "inner_hits": {
                "log_0": {"hits": {"hits": [{"_id": found_logs[0].log_id, "_source": found_logs[0].model_dump()}]}}
            },
        }
    )


def get_ids(hits: list[Hit[TestItemIndexData]]) -> list[str]:
    return [hit.source.test_item_id for hit in hits]


def get_no_defect_ids(no_defect_results: Optional[tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]]):
    return get_ids(no_defect_results[1]) if no_defect_results else None


def test_analysis_query_includes_no_defect_items():
    launch = build_launch(1)
    request_items = prepare_request_items_for_launch(launch)
    os_client = mock.Mock()
    os_client.msearch_grouped.return_value = iter([[]])
    service = AutoAnalyzerService(mock.Mock(), APP_CONFIG, DEFAULT_SEARCH_CONFIG, os_client=os_client)

    service._query_candidates_for_launch(launch, request_items)

    os_client.msearch_grouped.assert_called_once()
    queries = os_client.msearch_grouped.call_args.args[1]
    assert len(queries) == 2
    query = queries[1]
    query_text = json.dumps(query)
    assert '"nd*"' not in query_text
    assert '"ti*"' in query_text
    assert query["size"] == ANALYSIS_QUERY_SIZE
    assert ANALYSIS_QUERY_SIZE == 30


@pytest.mark.parametrize(
    "hits, expected_regular_ids, expected_no_defect_ids",
    [
        (
            [
                build_found_hit(2, "pb001", "2026-10-01 10:00:00"),
                build_found_hit(3, "nd001", "2026-10-02 10:00:00"),
            ],
            ["2"],
            ["3"],
        ),
        (
            [
                build_found_hit(2, "nd001", "2026-10-01 10:00:00"),
                build_found_hit(3, "pb001", "2026-10-02 10:00:00"),
            ],
            ["3"],
            None,
        ),
        (
            [
                build_found_hit(2, "pb001", "2026-10-01 10:00:00"),
                build_found_hit(3, "nd001", "2026-10-02 10:00:00", test_case_hash=OTHER_TEST_CASE_HASH),
            ],
            ["2"],
            None,
        ),
        (
            [
                build_found_hit(2, "ND_custom", "2026-10-02 10:00:00"),
                build_found_hit(3, "nd001", "2026-10-02 10:00:00"),
            ],
            [],
            ["2"],
        ),
        ([], [], None),
    ],
    ids=["latest_is_no_defect", "latest_is_not_no_defect", "other_test_case", "tie_keeps_best_scored", "no_hits"],
)
def test_split_no_defect_candidates(
    hits: list[Hit[TestItemIndexData]], expected_regular_ids: list[str], expected_no_defect_ids: Optional[list[str]]
):
    request_item = prepare_item(1)

    regular_results, no_defect_results = split_no_defect_candidates((request_item, hits))

    assert regular_results[0] is request_item
    assert get_ids(regular_results[1]) == expected_regular_ids
    assert get_no_defect_ids(no_defect_results) == expected_no_defect_ids
    if no_defect_results:
        assert no_defect_results[0] is request_item


def test_split_no_defect_candidates_without_test_case_hash():
    request_item = prepare_item(1, test_case_hash=0)
    hits = [build_found_hit(2, "nd001", "2026-10-02 10:00:00", test_case_hash=0)]

    regular_results, no_defect_results = split_no_defect_candidates((request_item, hits))

    assert get_ids(regular_results[1]) == []
    assert no_defect_results is None


def build_service(
    hits: list[Hit[TestItemIndexData]], predictions: list[tuple[list[int], list[list[float]]]]
) -> tuple[AutoAnalyzerService, mock.Mock, mock.Mock]:
    boosting_model = mock.Mock()
    boosting_model.feature_ids = [2, 3]
    boosting_model.is_custom = False
    boosting_model.get_model_info.return_value = ["global boosting model"]
    boosting_model.predict.side_effect = predictions
    defect_type_model = mock.Mock()
    defect_type_model.get_model_info.return_value = ["global defect type model"]
    model_chooser = mock.Mock()
    model_chooser.choose_model.side_effect = lambda project_id, model_type, **_: (
        boosting_model if model_type is ModelType.auto_analysis else defect_type_model
    )
    os_client = mock.Mock()
    os_client.msearch_grouped.return_value = iter([hits])
    service = AutoAnalyzerService(model_chooser, APP_CONFIG, DEFAULT_SEARCH_CONFIG, os_client=os_client)
    service.namespace_finder = mock.Mock()
    service.namespace_finder.get_chosen_namespaces.return_value = {}
    amqp_client = mock.Mock()
    service.amqp_client = amqp_client
    return service, boosting_model, amqp_client


def get_published_results(amqp_client: mock.Mock) -> list[dict[str, Any]]:
    amqp_client.publish_response.assert_called_once()
    return json.loads(amqp_client.publish_response.call_args.args[0])


ACCEPTED = ([1], [[0.1, 0.9]])
REJECTED = ([0], [[0.9, 0.1]])


def test_analysis_chooses_no_defect_candidate():
    hits = [
        build_found_hit(2, "pb001", "2026-10-01 10:00:00"),
        build_found_hit(3, "nd001", "2026-10-02 10:00:00"),
    ]
    service, boosting_model, amqp_client = build_service(hits, [ACCEPTED])

    service.analyze_logs([build_launch(1)])

    boosting_model.predict.assert_called_once()
    assert get_published_results(amqp_client) == [{"testItem": 1, "issueType": "nd001", "relevantItem": 3}]


def test_analysis_falls_back_to_regular_candidates_if_no_defect_is_rejected():
    hits = [
        build_found_hit(2, "pb001", "2026-10-01 10:00:00"),
        build_found_hit(3, "nd001", "2026-10-02 10:00:00"),
    ]
    service, boosting_model, amqp_client = build_service(hits, [REJECTED, ACCEPTED])

    service.analyze_logs([build_launch(1)])

    assert boosting_model.predict.call_count == 2
    assert get_published_results(amqp_client) == [{"testItem": 1, "issueType": "pb001", "relevantItem": 2}]


def test_analysis_without_no_defect_candidate():
    hits = [
        build_found_hit(2, "nd001", "2026-10-01 10:00:00"),
        build_found_hit(3, "pb001", "2026-10-02 10:00:00"),
    ]
    service, boosting_model, amqp_client = build_service(hits, [ACCEPTED])

    service.analyze_logs([build_launch(1)])

    boosting_model.predict.assert_called_once()
    assert len(boosting_model.predict.call_args.args[0]) == 1
    assert get_published_results(amqp_client) == [{"testItem": 1, "issueType": "pb001", "relevantItem": 3}]
