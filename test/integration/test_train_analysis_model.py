from unittest import mock

import pytest

from app.commons.model.db import Hit
from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model.test_item_index import LogData, TestItemHistoryData, TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.os_client import OsClient
from app.commons.query_builder import get_log_inner_hits_name
from app.ml.training.train_analysis_model import METRIC, AnalysisModelTraining
from app.utils import utils
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
    model_settings = utils.read_json_file("res", "model_settings.json", to_json=True)
    return DEFAULT_SEARCH_CONFIG.model_copy(
        update={
            "BoostModelFolder": utils.strip_path(model_settings["BOOST_MODEL_FOLDER"]),
            "SuggestBoostModelFolder": utils.strip_path(model_settings["SUGGEST_BOOST_MODEL_FOLDER"]),
            "GlobalDefectTypeModelFolder": utils.strip_path(model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"]),
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
    os_client.msearch_grouped = mock.Mock(side_effect=lambda _p, queries: iter([similar_hits] * (len(queries) // 2)))

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

    # One query per Test Item with issue history, all of them in one multi search
    os_client.msearch_grouped.assert_called_once()
    project_id, queries = os_client.msearch_grouped.call_args_list[0][0]
    assert project_id == 123
    assert len(queries) == 8, "multi search should have a header and a body for each of 4 Test Items"
    assert queries[0::2] == [{}] * 4
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
    for row in train_data:
        assert len(row) == len(training.features)
        assert all(isinstance(value, float) and 0.0 <= value <= 1.0 for value in row)


def test_unsupported_configured_features_are_excluded() -> None:
    search_cfg = DEFAULT_SEARCH_CONFIG.model_copy(
        update={"SuggestBoostModelFeatures": "40-45", "SuggestBoostModelMonotonousFeatures": "40,43,44"}
    )
    training = AnalysisModelTraining(
        APP_CONFIG,
        search_cfg,
        ModelType.suggestion,
        ModelChooser(APP_CONFIG, search_cfg, object_saver=mock.Mock()),
        os_client=mock.Mock(),
    )

    assert training.baseline_model is None
    assert training.features == [40, 41, 42]
    assert training.monotonous_features == [40]


def _make_item(test_item_id: str, issue_type: str, message: str) -> TestItemIndexData:
    return TestItemIndexData(
        test_item_id=test_item_id,
        test_item_name=f"item {test_item_id}",
        unique_id=f"uid-{test_item_id}",
        test_case_hash=int(test_item_id),
        launch_id="10",
        launch_name="launch",
        issue_type=issue_type,
        is_auto_analyzed=False,
        start_time="2025-01-01 00:00:00",
        logs=[_make_log_data(f"{test_item_id}0", 0, message)],
    )


def _get_query_excluded_issue_type(query: dict) -> str:
    """Get the issue type excluded by a similar items query, or an empty string if no issue type is excluded."""
    must_not = query["query"]["function_score"]["query"]["bool"]["must_not"]
    return next((clause["term"]["issue_type"] for clause in must_not if "issue_type" in clause.get("term", {})), "")


def test_similar_items_are_searched_in_batches_with_second_round_for_negatives() -> None:
    """First searches of all request Test Items go in batches, then the searches for negative cases, in batches too."""
    search_cfg = _make_search_config().model_copy(update={"AnalysisQueryBatchSize": 2})
    object_saver = mock.Mock()
    object_saver.get_folder_objects.return_value = []
    model_chooser = ModelChooser(APP_CONFIG, search_cfg, object_saver=object_saver)
    os_client = OsClient(APP_CONFIG, os_client=mock.Mock())
    training = AnalysisModelTraining(
        APP_CONFIG, search_cfg, ModelType.auto_analysis, model_chooser, os_client=os_client
    )

    # Item 1 finds only Test Items of its own issue type, item 2 finds mixed ones, item 3 finds nothing
    requests = [
        (_make_item("1", "pb001", "auth failed while reading account"), "pb001"),
        (_make_item("2", "ab001", "database timeout on select"), "ab001"),
        (_make_item("3", "si001", "network is unreachable"), "si001"),
    ]
    same_type_hit = _make_similar_hit(_make_item("11", "pb001", "auth failed while reading account"), 3.0)
    mixed_hits = [
        _make_similar_hit(_make_item("21", "ab001", "database timeout on select"), 3.0),
        _make_similar_hit(_make_item("22", "pb001", "database timeout on select"), 2.0),
    ]
    first_round = {"1": [same_type_hit], "2": mixed_hits, "3": []}
    negative_hits = {
        "1": [same_type_hit, _make_similar_hit(_make_item("12", "ab001", "auth failed while reading account"), 1.0)],
        "3": [_make_similar_hit(_make_item("31", "pb001", "network is unreachable"), 1.0)],
    }

    def respond(_project_id, body):
        groups = []
        for query in body[1::2]:
            must_not = query["query"]["function_score"]["query"]["bool"]["must_not"]
            request_id = next(
                clause["term"]["test_item_id"] for clause in must_not if "test_item_id" in clause.get("term", {})
            )
            groups.append(
                negative_hits[request_id] if _get_query_excluded_issue_type(query) else first_round[request_id]
            )
        return iter(groups)

    os_client.msearch_grouped = mock.Mock(side_effect=respond)

    results = training._collect_similar_hits(123, requests)

    # 3 first searches in batches of 2, then 2 searches for negatives in one batch
    calls = os_client.msearch_grouped.call_args_list
    assert len(calls) == 3
    assert [len(msearch_call[0][1]) // 2 for msearch_call in calls] == [2, 1, 2]
    assert all(msearch_call[0][0] == 123 for msearch_call in calls)
    first_round_queries = calls[0][0][1][1::2] + calls[1][0][1][1::2]
    assert [_get_query_excluded_issue_type(query) for query in first_round_queries] == ["", "", ""]
    second_round_queries = calls[2][0][1][1::2]
    assert [_get_query_excluded_issue_type(query) for query in second_round_queries] == ["pb001", "si001"]

    assert [[hit.source.test_item_id for hit in hits] for hits in results] == [["11", "12"], ["21", "22"], ["31"]]
