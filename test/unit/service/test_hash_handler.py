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

"""Unit tests for the error hash handlers of the auto-analysis chain."""

import json
from typing import Any, Iterator, Optional
from unittest import mock

from app.commons.model.db import Hit
from app.commons.model.launch_objects import AnalyzerConf, Launch, Log, TestItem
from app.commons.model.ml import ModelType
from app.commons.model.test_item_index import TestItemIndexData
from app.service.analysis import (
    AnalysisContext,
    AnyTestHashHandler,
    BoostingHandler,
    LaunchAnalysisStats,
    SameTestHashHandler,
)
from app.service.auto_analyzer_service import AutoAnalyzerService, prepare_request_items_for_launch
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG

LAUNCH_ID = 10
PROJECT_ID = 1


def build_launch(item_ids: list[int], test_case_hash: Optional[int] = None) -> Launch:
    return Launch(
        launchId=LAUNCH_ID,
        project=PROJECT_ID,
        launchName="Launch",
        analyzerConfig=AnalyzerConf(analyzerMode="ALL"),
        testItems=[
            TestItem(
                testItemId=item_id,
                isAutoAnalyzed=False,
                testCaseHash=item_id if test_case_hash is None else test_case_hash,
                logs=[Log(logId=item_id * 10, logLevel=40000, message=f"Unable to create user\nerror code {item_id}")],
            )
            for item_id in item_ids
        ],
    )


def build_context(launch: Launch) -> AnalysisContext:
    stats = LaunchAnalysisStats(
        launch_id=launch.launchId,
        launch_name=launch.launchName,
        project_id=launch.project,
        number_of_log_lines=-1,
        min_should_match=80,
        module_version=["test"],
    )
    return AnalysisContext(launch=launch, stats=stats, chosen_namespaces={})


def build_hit(
    test_item_id: int, issue_type: Optional[str], start_time: str = "2026-10-01 10:00:00", test_case_hash: int = 1
) -> Hit:
    source = {
        "test_item_id": str(test_item_id),
        "launch_id": "5",
        "issue_type": issue_type,
        "test_case_hash": test_case_hash,
        "start_time": start_time,
    }
    return Hit[TestItemIndexData].from_dict({"_id": str(test_item_id), "_score": None, "_source": source})


def build_os_client(*groups: list[list[Hit]]) -> mock.Mock:
    """Build an OpenSearch client mock returning the given hit groups, one list of groups per `msearch` call."""
    os_client = mock.Mock()
    calls = iter(groups)

    def msearch_grouped(_project: Any, _queries: list[dict[str, Any]]) -> Iterator[list[Hit]]:
        return iter(next(calls))

    os_client.msearch_grouped.side_effect = msearch_grouped
    return os_client


def build_handler(os_client: mock.Mock, lookup_batch_size: int = 200) -> SameTestHashHandler:
    return SameTestHashHandler(DEFAULT_SEARCH_CONFIG, os_client, lookup_batch_size=lookup_batch_size)


def get_ids(items: list[TestItemIndexData]) -> list[int]:
    return [int(item.test_item_id) for item in items]


def test_single_earlier_run_is_enough() -> None:
    launch = build_launch([1])
    handler = build_handler(build_os_client([[build_hit(7, "pb001")]]))

    results, remaining = handler.handle(build_context(launch), prepare_request_items_for_launch(launch))

    assert [(r.result.testItem, r.result.issueType, r.result.relevantItem, r.handler) for r in results] == [
        (1, "pb001", 7, "hash_same_test")
    ]
    assert remaining == []


def test_agreeing_runs_take_the_newest_as_relevant_item() -> None:
    launch = build_launch([1])
    hits = [
        build_hit(9, "ab_custom", "2026-10-03 10:00:00"),
        build_hit(8, "ab_custom", "2026-10-02 10:00:00"),
        build_hit(7, "ab_custom", "2026-10-01 10:00:00"),
    ]
    handler = build_handler(build_os_client([hits]))

    results, remaining = handler.handle(build_context(launch), prepare_request_items_for_launch(launch))

    assert [(r.result.issueType, r.result.relevantItem) for r in results] == [("ab_custom", 9)]
    assert remaining == []


def test_conflicting_runs_are_passed_on() -> None:
    launch = build_launch([1])
    handler = build_handler(build_os_client([[build_hit(9, "pb001"), build_hit(8, "ab001")]]))

    results, remaining = handler.handle(build_context(launch), prepare_request_items_for_launch(launch))

    assert results == []
    assert get_ids(remaining) == [1]


def test_items_without_earlier_runs_are_passed_on() -> None:
    launch = build_launch([1, 2])
    handler = build_handler(build_os_client([[build_hit(7, "si001")], []]))

    results, remaining = handler.handle(build_context(launch), prepare_request_items_for_launch(launch))

    assert [r.result.testItem for r in results] == [1]
    assert get_ids(remaining) == [2]


def test_runs_without_issue_type_are_passed_on() -> None:
    launch = build_launch([1])
    handler = build_handler(build_os_client([[build_hit(7, "")]]))

    results, remaining = handler.handle(build_context(launch), prepare_request_items_for_launch(launch))

    assert results == []
    assert get_ids(remaining) == [1]


def test_items_without_test_case_hash_are_not_looked_up() -> None:
    launch = build_launch([1, 2], test_case_hash=0)
    os_client = build_os_client()
    handler = build_handler(os_client)

    results, remaining = handler.handle(build_context(launch), prepare_request_items_for_launch(launch))

    os_client.msearch_grouped.assert_not_called()
    assert results == []
    assert get_ids(remaining) == [1, 2]


def test_items_without_error_hash_are_not_looked_up() -> None:
    launch = build_launch([1])
    items = prepare_request_items_for_launch(launch)
    items[0].error_hash = None
    os_client = build_os_client()

    results, remaining = build_handler(os_client).handle(build_context(launch), items)

    os_client.msearch_grouped.assert_not_called()
    assert get_ids(remaining) == [1]


def test_lookups_are_batched() -> None:
    launch = build_launch([1, 2, 3, 4, 5])
    os_client = build_os_client([[], []], [[], []], [[build_hit(7, "pb001")]])
    handler = build_handler(os_client, lookup_batch_size=2)

    results, remaining = handler.handle(build_context(launch), prepare_request_items_for_launch(launch))

    assert os_client.msearch_grouped.call_count == 3
    assert [len(call.args[1]) for call in os_client.msearch_grouped.call_args_list] == [4, 4, 2]
    assert [r.result.testItem for r in results] == [5]
    assert get_ids(remaining) == [1, 2, 3, 4]


def test_items_without_a_response_are_passed_on() -> None:
    launch = build_launch([1, 2])
    handler = build_handler(build_os_client([[build_hit(7, "pb001")]]))

    results, remaining = handler.handle(build_context(launch), prepare_request_items_for_launch(launch))

    assert [r.result.testItem for r in results] == [1]
    assert get_ids(remaining) == [2]


def test_lookup_query_asks_for_the_same_test_case_and_error() -> None:
    launch = build_launch([1])
    items = prepare_request_items_for_launch(launch)
    os_client = build_os_client([[]])

    build_handler(os_client).handle(build_context(launch), items)

    os_client.msearch_grouped.assert_called_once()
    project, queries = os_client.msearch_grouped.call_args.args
    assert project == PROJECT_ID
    assert len(queries) == 2
    query = queries[1]
    assert query["size"] == 3
    assert query["query"]["bool"]["filter"] == [
        {"term": {"error_hash": items[0].error_hash}},
        {"term": {"test_case_hash": 1}},
    ]


def build_service(hash_match_same_test: bool, os_client: mock.Mock) -> tuple[AutoAnalyzerService, mock.Mock]:
    boosting_model = mock.Mock()
    boosting_model.feature_ids = [2, 3]
    boosting_model.is_custom = False
    boosting_model.get_model_info.return_value = ["global boosting model"]
    defect_type_model = mock.Mock()
    defect_type_model.get_model_info.return_value = ["global defect type model"]
    model_chooser = mock.Mock()
    model_chooser.choose_model.side_effect = lambda project_id, model_type, **_: (
        boosting_model if model_type is ModelType.auto_analysis else defect_type_model
    )
    namespace_finder = mock.Mock()
    namespace_finder.get_chosen_namespaces.return_value = {}
    search_config = DEFAULT_SEARCH_CONFIG.model_copy(update={"HashMatchSameTest": hash_match_same_test})
    service = AutoAnalyzerService(
        model_chooser, APP_CONFIG, search_config, os_client=os_client, namespace_finder=namespace_finder
    )
    amqp_client = mock.Mock()
    service.amqp_client = amqp_client
    return service, amqp_client


def test_chain_starts_with_boosting_when_hash_match_is_disabled() -> None:
    service, _ = build_service(False, mock.Mock())

    assert isinstance(service.chain, BoostingHandler)


def test_chain_starts_with_same_test_hash_match_when_enabled() -> None:
    service, _ = build_service(True, mock.Mock())

    assert isinstance(service.chain, SameTestHashHandler)
    assert isinstance(service.chain._next, BoostingHandler)


def test_hash_matched_items_are_not_searched_by_boosting() -> None:
    os_client = build_os_client([[build_hit(7, "pb001")], []], [[]])
    service, amqp_client = build_service(True, os_client)

    service.analyze_logs([build_launch([1, 2])])

    assert os_client.msearch_grouped.call_count == 2
    lookup_queries = os_client.msearch_grouped.call_args_list[0].args[1]
    boosting_queries = os_client.msearch_grouped.call_args_list[1].args[1]
    assert len(lookup_queries) == 4
    assert len(boosting_queries) == 2
    assert "more_like_this" in json.dumps(boosting_queries[1])
    amqp_client.publish_response.assert_called_once()
    assert json.loads(amqp_client.publish_response.call_args.args[0]) == [
        {"testItem": 1, "issueType": "pb001", "relevantItem": 7}
    ]


OTHER_TEST = 2


def build_any_test_handler(os_client: mock.Mock) -> AnyTestHashHandler:
    return AnyTestHashHandler(DEFAULT_SEARCH_CONFIG, os_client)


def analyze_by_any_test(hits: list[Hit]) -> tuple[list, list[TestItemIndexData]]:
    launch = build_launch([1])
    handler = build_any_test_handler(build_os_client([hits]))
    return handler.handle(build_context(launch), prepare_request_items_for_launch(launch))


def test_any_test_takes_the_issue_type_all_test_cases_agree_on() -> None:
    results, remaining = analyze_by_any_test(
        [
            build_hit(9, "pb001", "2026-10-03 10:00:00", test_case_hash=OTHER_TEST),
            build_hit(8, "pb001", "2026-10-02 10:00:00", test_case_hash=3),
        ]
    )

    assert [(r.result.testItem, r.result.issueType, r.result.relevantItem, r.handler) for r in results] == [
        (1, "pb001", 9, "hash_any_test")
    ]
    assert remaining == []


def test_any_test_passes_on_conflicting_test_cases() -> None:
    results, remaining = analyze_by_any_test(
        [build_hit(9, "pb001", test_case_hash=OTHER_TEST), build_hit(8, "ab001", test_case_hash=3)]
    )

    assert results == []
    assert get_ids(remaining) == [1]


def test_any_test_takes_no_defect_of_the_newest_run_of_the_same_test_case() -> None:
    results, _ = analyze_by_any_test(
        [
            build_hit(9, "pb001", "2026-10-03 10:00:00", test_case_hash=OTHER_TEST),
            build_hit(8, "nd_flaky", "2026-10-02 10:00:00", test_case_hash=1),
            build_hit(7, "pb001", "2026-10-01 10:00:00", test_case_hash=1),
        ]
    )

    assert [(r.result.issueType, r.result.relevantItem) for r in results] == [("nd_flaky", 8)]


def test_any_test_skips_no_defect_when_the_newest_run_of_the_same_test_case_is_not_one() -> None:
    results, _ = analyze_by_any_test(
        [
            build_hit(9, "pb001", "2026-10-03 10:00:00", test_case_hash=1),
            build_hit(8, "nd001", "2026-10-02 10:00:00", test_case_hash=1),
            build_hit(7, "nd001", "2026-10-01 10:00:00", test_case_hash=OTHER_TEST),
            build_hit(6, "pb001", "2026-09-30 10:00:00", test_case_hash=OTHER_TEST),
        ]
    )

    assert [(r.result.issueType, r.result.relevantItem) for r in results] == [("pb001", 9)]


def test_any_test_ignores_no_defect_of_other_test_cases() -> None:
    results, _ = analyze_by_any_test(
        [
            build_hit(9, "nd001", "2026-10-03 10:00:00", test_case_hash=OTHER_TEST),
            build_hit(8, "si001", "2026-10-02 10:00:00", test_case_hash=3),
        ]
    )

    assert [(r.result.issueType, r.result.relevantItem) for r in results] == [("si001", 8)]


def test_any_test_passes_on_only_no_defect_of_other_test_cases() -> None:
    results, remaining = analyze_by_any_test([build_hit(9, "nd001", test_case_hash=OTHER_TEST)])

    assert results == []
    assert get_ids(remaining) == [1]


def test_any_test_looks_up_items_without_test_case_hash() -> None:
    launch = build_launch([1], test_case_hash=0)
    os_client = build_os_client([[build_hit(9, "nd001", test_case_hash=0), build_hit(8, "ab001", test_case_hash=0)]])

    results, remaining = build_any_test_handler(os_client).handle(
        build_context(launch), prepare_request_items_for_launch(launch)
    )

    os_client.msearch_grouped.assert_called_once()
    query = os_client.msearch_grouped.call_args.args[1][1]
    assert query["size"] == 20
    assert len(query["query"]["bool"]["filter"]) == 1
    assert [(r.result.issueType, r.result.relevantItem) for r in results] == [("ab001", 8)]
    assert remaining == []


def build_service_with_flags(
    same_test: bool, any_test: bool, os_client: mock.Mock
) -> tuple[AutoAnalyzerService, mock.Mock]:
    service, amqp_client = build_service(same_test, os_client)
    search_config = service.search_cfg.model_copy(update={"HashMatchAnyTest": any_test})
    service = AutoAnalyzerService(
        service.model_chooser,
        APP_CONFIG,
        search_config,
        os_client=os_client,
        namespace_finder=service.namespace_finder,
    )
    service.amqp_client = amqp_client
    return service, amqp_client


def test_chain_order_with_both_hash_matches_enabled() -> None:
    service, _ = build_service_with_flags(True, True, mock.Mock())

    assert isinstance(service.chain, SameTestHashHandler)
    assert isinstance(service.chain._next, AnyTestHashHandler)
    assert isinstance(service.chain._next._next, BoostingHandler)


def test_chain_with_only_any_test_hash_match_enabled() -> None:
    service, _ = build_service_with_flags(False, True, mock.Mock())

    assert isinstance(service.chain, AnyTestHashHandler)
    assert isinstance(service.chain._next, BoostingHandler)


def test_each_link_gets_only_the_items_previous_links_did_not_analyze() -> None:
    os_client = build_os_client(
        [[build_hit(7, "pb001")], [], []],
        [[build_hit(8, "ab001", test_case_hash=OTHER_TEST)], []],
        [[]],
    )
    service, amqp_client = build_service_with_flags(True, True, os_client)

    service.analyze_logs([build_launch([1, 2, 3])])

    assert [len(call.args[1]) for call in os_client.msearch_grouped.call_args_list] == [6, 4, 2]
    assert "more_like_this" in json.dumps(os_client.msearch_grouped.call_args_list[2].args[1][1])
    amqp_client.publish_response.assert_called_once()
    assert json.loads(amqp_client.publish_response.call_args.args[0]) == [
        {"testItem": 1, "issueType": "pb001", "relevantItem": 7},
        {"testItem": 2, "issueType": "ab001", "relevantItem": 8},
    ]
