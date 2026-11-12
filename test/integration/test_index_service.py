import datetime
import json
from unittest import mock

from app.commons.model.launch_objects import AnalyzerConf, BulkResponse, DefectUpdate, Launch
from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model.processing import Reply, ReplyType
from app.commons.model.test_item_index import TestItemHistoryData, TestItemIndexData
from app.service.index_service import IndexService
from test import APP_CONFIG


def _make_index_item(item_id: str, issue_type: str = "pb001") -> TestItemIndexData:
    """Create a minimal TestItemIndexData object for assertions."""
    return TestItemIndexData(
        test_item_id=str(item_id),
        test_item_name=f"item-{item_id}",
        unique_id=f"uid-{item_id}",
        test_case_hash=1,
        launch_id="10",
        launch_name="launch",
        launch_number="1",
        launch_start_time="2025-01-01 00:00:00",
        is_auto_analyzed=False,
        issue_type=issue_type,
        start_time="2025-01-01 00:00:00",
        log_count=0,
        logs=[],
    )


def train_replies(project: int, defects: int) -> list[Reply]:
    return [
        Reply(
            TrainInfo(model_type=model_type, project=project, gathered_metric_total=defects).model_dump_json(),
            ReplyType.QUEUE_SEND,
            "train_models",
        )
        for model_type in (ModelType.suggestion, ModelType.auto_analysis, ModelType.defect_type)
    ]


def test_index_logs_without_launches_replies_nothing() -> None:
    assert IndexService(APP_CONFIG, os_client=mock.Mock()).index_logs([]) is None


def test_index_logs_groups_by_project_and_passes_config() -> None:
    os_client = mock.Mock()
    os_client.bulk_index.side_effect = [
        BulkResponse(took=2, errors=False),
        BulkResponse(took=3, errors=False),
    ]

    service = IndexService(APP_CONFIG, os_client=os_client)

    analyzer_conf_first = AnalyzerConf(numberOfLogsToIndex=5, minimumLogLevel=30000, similarityThresholdToDrop=0.8)
    analyzer_conf_second = AnalyzerConf(numberOfLogsToIndex=7, minimumLogLevel=35000, similarityThresholdToDrop=0.5)

    launch_first = Launch(launchId=1, project=101, analyzerConfig=analyzer_conf_first, testItems=[])
    launch_second = Launch(launchId=2, project=202, analyzerConfig=analyzer_conf_second, testItems=[])

    prepared_first_items = [_make_index_item("1", "ab001")]
    prepared_second_items = [_make_index_item("2", "ti001")]

    with mock.patch(
        "app.service.index_service.request_factory.prepare_test_items",
        side_effect=[prepared_first_items, prepared_second_items],
    ) as prepare_mock:
        replies = service.index_logs([launch_first, launch_second])

    prepare_mock.assert_has_calls(
        [
            mock.call(
                launch_first,
                number_of_logs_to_index=analyzer_conf_first.numberOfLogsToIndex,
                minimal_log_level=analyzer_conf_first.minimumLogLevel,
                similarity_threshold_to_drop=analyzer_conf_first.similarityThresholdToDrop,
            ),
            mock.call(
                launch_second,
                number_of_logs_to_index=analyzer_conf_second.numberOfLogsToIndex,
                minimal_log_level=analyzer_conf_second.minimumLogLevel,
                similarity_threshold_to_drop=analyzer_conf_second.similarityThresholdToDrop,
            ),
        ]
    )

    os_client.bulk_index.assert_has_calls(
        [mock.call(101, prepared_first_items), mock.call(202, prepared_second_items)], any_order=False
    )
    # Training is requested only for the project with defect types, "ti" is not one
    assert replies == train_replies(101, 1)


def test_defect_update_updates_issue_history_and_docs() -> None:
    os_client = mock.Mock()
    # Test Item 9999 is not indexed, so its script update fails with a missing document error
    os_client.bulk_update_issue_history.return_value = BulkResponse(took=2, errors=True, failed_ids=["9999"])

    service = IndexService(APP_CONFIG, os_client=os_client)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    defect_update_info = {
        "project": 123,
        "itemsToUpdate": {
            "1001": {
                "issueType": "PB001",
                "issueComment": "user fix",
                "timestamp": list(datetime.datetime.fromisoformat("2025-01-02 10:00:00").timetuple())[:7],
            },
            "1002": "SI002",
            "9999": "AB003",
        },
    }

    replies = service.defect_update(DefectUpdate(**defect_update_info))
    not_updated = json.loads(replies[0].payload)

    # Updates are sent for all requested Test Items at once, without searching for them first
    os_client.search.assert_not_called()
    os_client.msearch_grouped.assert_not_called()
    os_client.bulk_update_issue_history.assert_called_once()
    project_id, history_updates = os_client.bulk_update_issue_history.call_args[0]
    assert project_id == 123
    assert len(history_updates) == 3
    assert [entry.test_item_id for entry in history_updates] == ["1001", "1002", "9999"]
    assert [entry.issue_type for entry in history_updates] == ["pb001", "si002", "ab003"]

    first_entry = next(entry for entry in history_updates if entry.test_item_id == "1001")
    assert first_entry.issue_type == "pb001"
    assert first_entry.issue_comment == "user fix"
    assert first_entry.timestamp == "2025-01-02 10:00:00"
    assert first_entry.is_auto_analyzed is False

    second_entry = next(entry for entry in history_updates if entry.test_item_id == "1002")
    assert second_entry.issue_type == "si002"
    assert second_entry.issue_comment == ""
    assert second_entry.timestamp == timestamp
    assert second_entry.is_auto_analyzed is False

    assert set(not_updated) == {9999}
    assert replies[1:] == train_replies(123, 2)


def test_defect_update_reports_items_which_disappeared_before_update() -> None:
    os_client = mock.Mock()
    os_client.bulk_update_issue_history.return_value = BulkResponse(took=1, errors=True, failed_ids=["1002", "9999"])

    service = IndexService(APP_CONFIG, os_client=os_client)

    defect_update_info = {"project": 123, "itemsToUpdate": {"1001": "ab001", "1002": "ab001", "9999": "ab001"}}
    replies = service.defect_update(DefectUpdate(**defect_update_info))
    assert replies[0].type is ReplyType.DIRECT_REPLY
    not_updated = json.loads(replies[0].payload)

    os_client.bulk_update_issue_history.assert_called_once()
    assert sorted(not_updated) == [1002, 9999]


def test_defect_update_reports_all_items_when_the_update_request_fails() -> None:
    """If the whole update failed, e.g. the index does not exist, no Test Item was updated."""
    os_client = mock.Mock()
    os_client.bulk_update_issue_history.return_value = BulkResponse(took=0, errors=True)

    service = IndexService(APP_CONFIG, os_client=os_client)

    defect_update_info = {"project": 123, "itemsToUpdate": {"1001": "ab001", "1002": "ab001"}}
    not_updated = service.defect_update(DefectUpdate(**defect_update_info))

    os_client.bulk_update_issue_history.assert_called_once()
    assert len(not_updated) == 1
    assert json.loads(not_updated[0].payload) == [1001, 1002]


def test_defect_update_accepts_timestamp_with_nanoseconds() -> None:
    os_client = mock.Mock()
    os_client.bulk_update_issue_history.return_value = BulkResponse(took=1, errors=False)

    service = IndexService(APP_CONFIG, os_client=os_client)

    defect_update_info = {
        "project": 123,
        "itemsToUpdate": {"1001": {"issueType": "AB001", "timestamp": [2026, 10, 18, 17, 59, 2, 797000000]}},
    }
    replies = service.defect_update(DefectUpdate(**defect_update_info))
    assert replies[0].type is ReplyType.DIRECT_REPLY
    not_updated = json.loads(replies[0].payload)

    history_updates: list[TestItemHistoryData] = os_client.bulk_update_issue_history.call_args[0][1]
    assert len(history_updates) == 1
    assert history_updates[0].timestamp == "2026-10-18 17:59:02"
    assert history_updates[0].issue_type == "ab001"
    assert not_updated == []
