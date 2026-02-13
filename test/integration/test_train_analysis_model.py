from pathlib import Path
from unittest import mock

import pytest

from app.commons.model.db import Hit
from app.commons.model.log_item_index import LogItemIndexData
from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model.test_item_index import LogData, TestItemHistoryData, TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.os_client import OsClient
from app.ml.boosting_featurizer import BoostingFeaturizer
from app.ml.suggest_boosting_featurizer import SuggestBoostingFeaturizer
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

    os_client.search = mock.Mock(return_value=iter(project_hits))

    def fake_gather_features_info(self):  # noqa: ANN001
        issue_type_names = list(self.find_most_relevant_by_type().keys())
        feature_vector = [0.1 for _ in self.feature_ids]
        return [feature_vector for _ in issue_type_names], issue_type_names

    def fake_find_most_relevant_by_type(self):  # noqa: ANN001
        def make_hit(issue_type: str) -> Hit[LogItemIndexData]:
            return Hit[LogItemIndexData].from_dict({"_score": 1.0, "_source": LogItemIndexData(issue_type=issue_type)})

        return {
            "pb001": {"mrHit": make_hit("pb001")},
            "ab001": {"mrHit": make_hit("ab001")},
            "si001": {"mrHit": make_hit("si001")},
        }

    training = AnalysisModelTraining(
        APP_CONFIG,
        search_cfg,
        model_type,
        model_chooser,
        os_client=os_client,
    )
    training.namespace_finder.get_chosen_namespaces = mock.Mock(return_value={})

    with mock.patch.object(model_chooser, "choose_model", wraps=model_chooser.choose_model) as choose_model_mock:
        with mock.patch.object(BoostingFeaturizer, "gather_features_info", new=fake_gather_features_info):
            with mock.patch.object(
                BoostingFeaturizer, "find_most_relevant_by_type", new=fake_find_most_relevant_by_type
            ):
                with mock.patch.object(
                    SuggestBoostingFeaturizer,
                    "find_most_relevant_by_type",
                    new=fake_find_most_relevant_by_type,
                ):
                    with mock.patch.object(
                        AnalysisModelTraining,
                        "_train_several_times",
                        return_value=({METRIC: [0.1]}, {METRIC: [0.1]}, True, 0.0),
                    ):
                        training.train(TrainInfo(model_type=model_type, project=123))

    choose_model_mock.assert_called_once_with(123, ModelType.defect_type)
    training.namespace_finder.get_chosen_namespaces.assert_called_once_with(123)
    assert os_client.search.call_count == 1
    assert os_client.search.call_args_list[0][0][0] == 123

    issue_history_query = os_client.search.call_args_list[0][0][1]
    assert issue_history_query["query"]["nested"]["path"] == "issue_history"
