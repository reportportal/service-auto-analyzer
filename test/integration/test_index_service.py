import datetime
from unittest import mock

from app.commons.model.launch_objects import AnalyzerConf, BulkResponse, DefectUpdate, Launch
from app.commons.model.test_item_index import TestItemIndexData, TestItemUpdateData
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
        response = service.index_logs([launch_first, launch_second])

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
    assert response.took == 5
    assert response.errors is False


def test_defect_update_updates_issue_history_and_docs() -> None:
    os_client = mock.Mock()
    existing_items = [
        _make_index_item("1001", "ti001"),
        _make_index_item("1002", "ti002"),
    ]
    os_client.get_test_items_by_ids.return_value = existing_items
    os_client.bulk_index_raw.return_value = BulkResponse(took=2, errors=False)
    os_client.bulk_update_issue_history.return_value = BulkResponse(took=2, errors=False)

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

    not_updated = service.defect_update(DefectUpdate(**defect_update_info))

    os_client.get_test_items_by_ids.assert_called_once_with(123, ["1001", "1002", "9999"])

    project_id, doc_updates = os_client.bulk_update_issue_history.call_args[0]
    assert project_id == 123
    assert len(doc_updates) == 2
    doc_ids = {doc.test_item_id for doc in doc_updates}
    assert doc_ids == {"1001", "1002"}
    assert {doc.issue_type for doc in doc_updates} == {"pb001", "si002"}

    history_updates: list[TestItemUpdateData] = os_client.bulk_update_issue_history.call_args[0][1]
    assert {entry.test_item_id for entry in history_updates} == {"1001", "1002"}

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
