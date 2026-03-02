from pathlib import Path
from unittest import mock

from opensearchpy import OpenSearch

from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model.test_item_index import LogData, TestItemHistoryData, TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.os_client import OsClient, get_test_item_index_name
from app.ml.training.train_defect_type_model import DefectTypeModelTraining
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "res" / "model"


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
        message_for_clustering=message,
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


def _make_search_config():
    return DEFAULT_SEARCH_CONFIG.model_copy(
        update={
            "GlobalDefectTypeModelFolder": str(MODEL_DIR / "defect_type_model_2025-08-12"),
        }
    )


def test_train_uses_os_client_issue_history_query() -> None:
    search_cfg = _make_search_config()
    object_saver = mock.Mock()
    object_saver.get_folder_objects.return_value = []
    model_chooser = ModelChooser(APP_CONFIG, search_cfg, object_saver=object_saver)

    mocked_opensearch = mock.Mock(spec=OpenSearch)
    mocked_opensearch.indices = mock.Mock()
    mocked_opensearch.indices.get.return_value = {}
    os_client = OsClient(APP_CONFIG, os_client=mocked_opensearch)

    logs = [
        _make_log_data("101", 0, "authentication failed for account"),
        _make_log_data("102", 1, "database timeout during login"),
    ]
    history = [
        TestItemHistoryData(
            test_item_id="1001",
            is_auto_analyzed=True,
            issue_type="ab001",
            timestamp="2025-01-01 00:00:00",
            issue_comment="",
        ),
        TestItemHistoryData(
            test_item_id="1001",
            is_auto_analyzed=False,
            issue_type="pb001",
            timestamp="2025-01-02 00:00:00",
            issue_comment="user update",
        ),
    ]
    test_item = TestItemIndexData(
        test_item_id="1001",
        test_item_name="login test",
        unique_id="uid-1001",
        test_case_hash=321,
        launch_id="10",
        launch_name="launch",
        issue_type="pb001",
        is_auto_analyzed=False,
        start_time="2025-01-01 00:00:00",
        logs=logs,
        issue_history=history,
    )
    raw_hit = {"_index": "rp_123", "_id": "1001", "_source": test_item.model_dump()}

    with mock.patch("opensearchpy.helpers.scan", return_value=[raw_hit]) as scan_mock:
        training = DefectTypeModelTraining(
            APP_CONFIG,
            search_cfg,
            model_chooser=model_chooser,
            os_client=os_client,
        )
        with mock.patch.object(
            DefectTypeModelTraining,
            "_train_several_times",
            return_value=([0.1], [0.1], True, 0.0),
        ) as train_mock:
            training.train(TrainInfo(model_type=ModelType.defect_type, project=123))

    scan_mock.assert_called_once()
    args, kwargs = scan_mock.call_args
    assert args[0] is mocked_opensearch
    expected_index = get_test_item_index_name(123, APP_CONFIG.esProjectIndexPrefix)
    assert kwargs["index"] == expected_index
    mocked_opensearch.indices.get.assert_called_once_with(index=expected_index)

    issue_history_query = kwargs["query"]
    assert issue_history_query["query"]["nested"]["path"] == "issue_history"
    assert "logs" in issue_history_query["_source"]
    assert "issue_history" in issue_history_query["_source"]
    assert train_mock.called
