from pathlib import Path
from unittest import mock

import pytest

from app.commons.model.db import Hit
from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model.test_item_index import LogData, TestItemHistoryData, TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.os_client import OsClient
from app.commons.query_builder import get_log_inner_hits_name
from app.ml.training.train_analysis_model import METRIC, AnalysisModelTraining
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


def _make_similar_hit(test_item: TestItemIndexData, score: float) -> Hit[TestItemIndexData]:
    logs = test_item.logs or []
    inner_hits = {
        get_log_inner_hits_name(0): {
            "hits": {"hits": [{"_id": logs[0].log_id, "_score": score, "_source": logs[0].model_dump()}]}
        }
    }
    source = test_item.model_copy(update={"logs": None, "issue_history": None})
    return Hit[TestItemIndexData].from_dict(
        {"_id": test_item.test_item_id, "_score": score, "_source": source.model_dump(), "inner_hits": inner_hits}
    )


def _make_search_config():
    return DEFAULT_SEARCH_CONFIG.model_copy(
        update={
            "BoostModelFolder": str(MODEL_DIR / "auto_analysis_model_2025-08-18"),
            "SuggestBoostModelFolder": str(MODEL_DIR / "suggestion_model_2025-09-04"),
            "GlobalDefectTypeModelFolder": str(MODEL_DIR / "defect_type_model_2025-08-12"),
        }
    )


@pytest.mark.parametrize("model_type", [ModelType.auto_analysis, ModelType.suggestion])
def test_train_uses_os_client_and_issue_history(model_type: ModelType) -> None:
    search_cfg = _make_search_config()
    object_saver = mock.Mock()
    object_saver.get_folder_objects.return_value = []
    model_chooser = ModelChooser(APP_CONFIG, search_cfg, object_saver=object_saver)

    mocked_opensearch = mock.Mock()
    os_client = OsClient(APP_CONFIG, os_client=mocked_opensearch)

    request_logs = [
        _make_log_data("101", 0, "auth failed while reading account"),
        _make_log_data("102", 1, "database timeout on select"),
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
    request_item = TestItemIndexData(
        test_item_id="1001",
        test_item_name="login test",
        unique_id="uid-1001",
        test_case_hash=321,
        launch_id="10",
        launch_name="launch",
        issue_type="pb001",
        is_auto_analyzed=False,
        start_time="2025-01-01 00:00:00",
        logs=request_logs,
        issue_history=history,
    )
    request_hit = Hit[TestItemIndexData].from_dict(
        {"_index": "rp_123", "_id": "1001", "_source": request_item.model_dump()}
    )

    candidate_pb = TestItemIndexData(
        test_item_id="2001",
        test_item_name="candidate pb",
        unique_id="uid-2001",
        test_case_hash=321,
        launch_id="11",
        launch_name="launch",
        issue_type="pb001",
        is_auto_analyzed=False,
        start_time="2025-01-01 00:00:00",
        logs=[_make_log_data("201", 0, "auth failed while reading account")],
        issue_history=[
            TestItemHistoryData(
                test_item_id="2001",
                is_auto_analyzed=False,
                issue_type="pb001",
                timestamp="2025-01-02 00:00:00",
                issue_comment="",
            )
        ],
    )
    candidate_ab = TestItemIndexData(
        test_item_id="2002",
        test_item_name="candidate ab",
        unique_id="uid-2002",
        test_case_hash=321,
        launch_id="11",
        launch_name="launch",
        issue_type="ab001",
        is_auto_analyzed=True,
        start_time="2025-01-01 00:00:00",
        logs=[_make_log_data("202", 0, "serialization error")],
        issue_history=[
            TestItemHistoryData(
                test_item_id="2002",
                is_auto_analyzed=True,
                issue_type="ab001",
                timestamp="2025-01-02 00:00:00",
                issue_comment="",
            )
        ],
    )
    candidate_si = TestItemIndexData(
        test_item_id="2003",
        test_item_name="candidate si",
        unique_id="uid-2003",
        test_case_hash=321,
        launch_id="11",
        launch_name="launch",
        issue_type="si001",
        is_auto_analyzed=False,
        start_time="2025-01-01 00:00:00",
        logs=[_make_log_data("203", 0, "network timeout")],
        issue_history=[
            TestItemHistoryData(
                test_item_id="2003",
                is_auto_analyzed=False,
                issue_type="si001",
                timestamp="2025-01-02 00:00:00",
                issue_comment="",
            )
        ],
    )
    project_hits = [
        request_hit,
        Hit[TestItemIndexData].from_dict({"_index": "rp_123", "_id": "2001", "_source": candidate_pb.model_dump()}),
        Hit[TestItemIndexData].from_dict({"_index": "rp_123", "_id": "2002", "_source": candidate_ab.model_dump()}),
        Hit[TestItemIndexData].from_dict({"_index": "rp_123", "_id": "2003", "_source": candidate_si.model_dump()}),
    ]

    similar_hits = [
        _make_similar_hit(candidate_pb, 3.0),
        _make_similar_hit(candidate_ab, 2.0),
        _make_similar_hit(candidate_si, 1.0),
    ]
    os_client.search = mock.Mock(side_effect=lambda _p, _r: iter(project_hits))
    os_client.msearch_grouped = mock.Mock(side_effect=lambda _p, _q: iter([similar_hits]))

    training = AnalysisModelTraining(
        APP_CONFIG,
        search_cfg,
        model_type,
        model_chooser,
        os_client=os_client,
    )
    training.namespace_finder.get_chosen_namespaces = mock.Mock(return_value={})

    with mock.patch.object(model_chooser, "choose_model", wraps=model_chooser.choose_model) as choose_model_mock:
        with mock.patch.object(
            AnalysisModelTraining,
            "_train_several_times",
            return_value=({METRIC: [0.1]}, {METRIC: [0.1]}, True, 0.0),
        ) as train_mock:
            training.train(TrainInfo(model_type=model_type, project=123))

    choose_model_mock.assert_called_once_with(123, ModelType.defect_type)
    training.namespace_finder.get_chosen_namespaces.assert_called_once_with(123)
    os_client.search.assert_called_once()
    assert os_client.search.call_args_list[0][0][0] == 123
    issue_history_query = os_client.search.call_args_list[0][0][1]
    assert issue_history_query["query"]["nested"]["path"] == "issue_history"

    # One search per Test Item with issue history
    assert os_client.msearch_grouped.call_count == 4
    project_id, queries = os_client.msearch_grouped.call_args_list[0][0]
    assert project_id == 123
    assert len(queries) == 2
    item_query = queries[1]["query"]["function_score"]["query"]["bool"]
    log_clauses = item_query["must"][0]["bool"]["should"]
    assert [clause["nested"]["inner_hits"]["name"] for clause in log_clauses] == ["log_0", "log_1"]
    assert {"term": {"test_item_id": "1001"}} in item_query["must_not"]

    train_mock.assert_called_once()
    _, train_data, labels = train_mock.call_args[0]
    # Every Test Item gives one positive and two negative rows, one row per found Test Item
    assert len(labels) == 12
    assert sum(labels) == 4
    assert len(train_data) == 12
    assert all(row == [0.0] * len(training.features) for row in train_data)
