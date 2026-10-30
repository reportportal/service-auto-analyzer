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

"""Unit tests for the auto-analysis handler chain and its orchestration by AutoAnalyzerService."""

import json
from datetime import datetime
from typing import Optional
from unittest import mock

from app.commons.model.launch_objects import AnalysisResult, AnalyzerConf, Launch, Log, TestItem
from app.commons.model.test_item_index import TestItemIndexData
from app.service.analysis import AnalysisContext, AnalysisHandler, HandlerResult, LaunchAnalysisStats
from app.service.auto_analyzer_service import AutoAnalyzerService, prepare_request_items_for_launch
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG


class ResolvingHandler(AnalysisHandler):
    """Resolve the Test Items with the given IDs and record every batch it receives."""

    def __init__(self, name: str, resolved_ids: set[int], next_handler: Optional[AnalysisHandler] = None) -> None:
        super().__init__(next_handler)
        self.name = name
        self.resolved_ids = resolved_ids
        self.received: list[list[int]] = []

    def _analyze(
        self, context: AnalysisContext, items: list[TestItemIndexData]
    ) -> tuple[list[HandlerResult], list[TestItemIndexData]]:
        self.received.append([int(item.test_item_id) for item in items])
        results = [
            HandlerResult(
                result=AnalysisResult(testItem=int(item.test_item_id), issueType=f"{self.name}001", relevantItem=1),
                handler=self.name,
            )
            for item in items
            if int(item.test_item_id) in self.resolved_ids
        ]
        remaining = [item for item in items if int(item.test_item_id) not in self.resolved_ids]
        return results, remaining


class FailingHandler(AnalysisHandler):
    name = "failing"

    def _analyze(
        self, context: AnalysisContext, items: list[TestItemIndexData]
    ) -> tuple[list[HandlerResult], list[TestItemIndexData]]:
        raise RuntimeError("handler failure")


def build_launch(launch_id: int, item_ids: list[int], with_logs: bool = True) -> Launch:
    return Launch(
        launchId=launch_id,
        project=1,
        launchName="Launch",
        analyzerConfig=AnalyzerConf(),
        testItems=[
            TestItem(
                testItemId=item_id,
                isAutoAnalyzed=False,
                testCaseHash=item_id,
                logs=(
                    [Log(logId=item_id * 10, logLevel=40000, message=f"Unable to create user {item_id}")]
                    if with_logs
                    else []
                ),
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


def get_ids(items: list[TestItemIndexData]) -> list[int]:
    return [int(item.test_item_id) for item in items]


def build_service(chain: AnalysisHandler, max_items: Optional[int] = None) -> tuple[AutoAnalyzerService, mock.Mock]:
    search_config = DEFAULT_SEARCH_CONFIG.model_copy()
    if max_items is not None:
        search_config.MaxAutoAnalysisItemsToProcess = max_items
    service = AutoAnalyzerService(
        mock.Mock(), APP_CONFIG, search_config, os_client=mock.Mock(), namespace_finder=mock.Mock()
    )
    service.chain = chain
    amqp_client = mock.Mock()
    service.amqp_client = amqp_client
    return service, amqp_client


def get_published_item_ids(amqp_client: mock.Mock) -> list[int]:
    amqp_client.publish_response.assert_called_once()
    return [result["testItem"] for result in json.loads(amqp_client.publish_response.call_args.args[0])]


def test_chain_passes_only_remaining_items_to_the_next_handler() -> None:
    last = ResolvingHandler("last", {3})
    first = ResolvingHandler("first", {1}, next_handler=last)
    launch = build_launch(1, [1, 2, 3])

    results, remaining = first.handle(build_context(launch), prepare_request_items_for_launch(launch))

    assert first.received == [[1, 2, 3]]
    assert last.received == [[2, 3]]
    assert [(result.result.testItem, result.handler) for result in results] == [(1, "first"), (3, "last")]
    assert get_ids(remaining) == [2]


def test_chain_does_not_call_the_next_handler_when_nothing_remains() -> None:
    last = ResolvingHandler("last", set())
    first = ResolvingHandler("first", {1, 2}, next_handler=last)
    launch = build_launch(1, [1, 2])

    results, remaining = first.handle(build_context(launch), prepare_request_items_for_launch(launch))

    assert len(results) == 2
    assert remaining == []
    assert last.received == []


def test_last_handler_returns_its_remaining_items() -> None:
    handler = ResolvingHandler("only", {2})
    launch = build_launch(1, [1, 2])

    results, remaining = handler.handle(build_context(launch), prepare_request_items_for_launch(launch))

    assert [result.result.testItem for result in results] == [2]
    assert get_ids(remaining) == [1]


def test_service_publishes_results_of_all_handlers_once() -> None:
    chain = ResolvingHandler("first", {1}, next_handler=ResolvingHandler("last", {3}))
    service, amqp_client = build_service(chain)

    service.analyze_logs([build_launch(1, [1, 2]), build_launch(2, [3, 4])])

    assert get_published_item_ids(amqp_client) == [1, 3]
    assert chain.received == [[1, 2], [3, 4]]


def test_service_skips_chain_for_launch_without_logs() -> None:
    chain = ResolvingHandler("only", {1})
    service, amqp_client = build_service(chain)

    service.analyze_logs([build_launch(1, [5], with_logs=False), build_launch(2, [1])])

    assert chain.received == [[1]]
    assert get_published_item_ids(amqp_client) == [1]


def test_service_caps_items_to_process_across_launches() -> None:
    chain = ResolvingHandler("only", {1, 2, 3, 4})
    service, amqp_client = build_service(chain, max_items=3)

    service.analyze_logs([build_launch(1, [1, 2]), build_launch(2, [3, 4]), build_launch(3, [5])])

    assert chain.received == [[1, 2], [3]]
    assert get_published_item_ids(amqp_client) == [1, 2, 3]


def test_service_publishes_empty_results_when_chain_fails() -> None:
    service, amqp_client = build_service(FailingHandler())

    service.analyze_logs([build_launch(1, [1])])

    assert get_published_item_ids(amqp_client) == []


def test_service_shares_chosen_namespaces_cache_between_launches() -> None:
    contexts: list[AnalysisContext] = []

    class RecordingHandler(ResolvingHandler):
        def _analyze(
            self, context: AnalysisContext, items: list[TestItemIndexData]
        ) -> tuple[list[HandlerResult], list[TestItemIndexData]]:
            contexts.append(context)
            return super()._analyze(context, items)

    service, _ = build_service(RecordingHandler("only", set()))

    service.analyze_logs([build_launch(1, [1]), build_launch(2, [2])])

    assert len(contexts) == 2
    assert contexts[0].chosen_namespaces is contexts[1].chosen_namespaces
    assert contexts[0].stats is not contexts[1].stats
    assert contexts[0].stats.items_to_process == 1
    assert contexts[0].stats.not_found == 1


def test_launch_stats_report() -> None:
    stats = LaunchAnalysisStats(
        launch_id=1,
        launch_name="Launch",
        project_id=2,
        number_of_log_lines=-1,
        min_should_match=80,
        module_version=["test"],
        gathered_at=datetime(2026, 10, 31, 9, 5, 7),
    )
    stats.model_info.update({"global boosting model", "global boosting model", "global defect type model"})
    stats.errors.append("RuntimeError: failure\n")

    report = stats.model_dump(mode="json")

    assert set(report) == {
        "not_found",
        "items_to_process",
        "processed_time",
        "launch_id",
        "launch_name",
        "project_id",
        "method",
        "gather_date",
        "gather_datetime",
        "number_of_log_lines",
        "min_should_match",
        "model_info",
        "module_version",
        "errors",
        "errors_count",
    }
    assert sorted(report["model_info"]) == ["global boosting model", "global defect type model"]
    assert report["errors_count"] == 1
    assert report["gather_date"] == "2026-10-31"
    assert report["gather_datetime"] == "2026-10-31 09:05:07"
    assert report["method"] == "auto_analysis"
    assert json.loads(json.dumps(report)) == report
