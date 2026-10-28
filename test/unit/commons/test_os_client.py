from typing import Any
from unittest import mock

import pytest
from opensearchpy import ConnectionError as OsConnectionError
from opensearchpy import TransportError

from app.commons.model.test_item_index import (
    LogClusterData,
    LogData,
    MergedLogData,
    MergedLogsClusterData,
    TestItemHistoryData,
    TestItemIndexData,
)
from app.commons.os_client import HEALTH_CHECK_TIMEOUT, OsClient, get_test_item_index_name
from test import APP_CONFIG

PROJECT_ID = "42"


@pytest.fixture
def app_config():
    return APP_CONFIG.model_copy()


@pytest.fixture
def os_client_mock():
    client = mock.Mock()
    client.indices = mock.Mock()
    return client


@pytest.fixture
def test_item():
    log = LogData(
        log_id="1",
        log_order=0,
        log_time="2025-01-01T00:00:00Z",
        log_level=40000,
        cluster_id="",
        cluster_message="",
        cluster_with_numbers=False,
        original_message="original",
        message="clean",
        message_lines=1,
        message_words_number=1,
        message_extended="clean",
        message_without_params_extended="clean",
        message_without_params_and_brackets="clean",
        detected_message="clean",
        detected_message_with_numbers="clean",
        detected_message_extended="clean",
        detected_message_without_params_extended="clean",
        detected_message_without_params_and_brackets="clean",
        stacktrace="trace",
        stacktrace_extended="trace",
        only_numbers="",
        potential_status_codes="",
        found_exceptions="",
        found_exceptions_extended="",
        found_tests_and_methods="",
        urls="",
        paths="",
        message_params="",
        whole_message="whole",
    )
    return TestItemIndexData(
        test_item_id="ti-1",
        test_item_name="name",
        unique_id="uid",
        test_case_hash=123,
        launch_id="l-1",
        launch_name="launch",
        launch_number="1",
        launch_start_time="2025-01-01T00:00:00Z",
        is_auto_analyzed=True,
        issue_type="ab001",
        start_time="2025-01-01T00:00:00Z",
        log_count=1,
        logs=[log],
        issue_history=[],
    )


def test_bulk_index_creates_index_and_calls_bulk(monkeypatch, os_client_mock, app_config, test_item):
    os_client_mock.indices.get.side_effect = Exception("missing index")
    os_client_mock.indices.create.return_value = {"acknowledged": True}

    captured: dict[str, Any] = {}

    monkeypatch.setattr("app.commons.os_client.utils.read_resource_file", lambda *args, **kwargs: {})

    # noinspection PyUnusedLocal
    def fake_bulk(
        self: Any, request: list, chunk_size: int, request_timeout: int, refresh: bool, raise_on_error: bool
    ):
        captured["client"] = self
        captured["bodies"] = request
        captured["chunk_size"] = chunk_size
        captured["refresh"] = refresh
        return len(request), []

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", fake_bulk)

    client = OsClient(app_config, os_client=os_client_mock)
    response = client.bulk_index(PROJECT_ID, [test_item], refresh=False, chunk_size=5)

    os_client_mock.indices.create.assert_called_once()
    assert captured["client"] is os_client_mock
    assert captured["chunk_size"] == 5
    assert captured["refresh"] is False

    bodies: list = captured["bodies"]
    assert len(bodies) == 1
    assert bodies[0]["_index"] == get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    assert bodies[0]["_id"] == test_item.test_item_id
    assert bodies[0]["_source"] == test_item.to_index_dict()
    assert response.errors is False


def test_bulk_update_issue_history_not_creates_index_and_return_error(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing index")
    scan_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", scan_mock)

    client = OsClient(app_config, os_client=os_client_mock)
    updates = [
        TestItemHistoryData(
            test_item_id="ti-1",
            is_auto_analyzed=True,
            issue_type="pb001",
            timestamp="2025-01-02T00:00:00Z",
            issue_comment="comment",
        )
    ]

    response = client.bulk_update_issue_history(PROJECT_ID, updates)

    os_client_mock.indices.create.assert_not_called()
    scan_mock.assert_not_called()

    assert response.errors is True
    assert not response.items


def test_bulk_update_issue_history_success(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}

    captured: dict[str, Any] = {}

    # noinspection PyUnusedLocal
    def fake_bulk(
        os_client: Any, request: list, chunk_size: int, request_timeout: int, refresh: bool, raise_on_error: bool
    ):
        captured["client"] = os_client
        captured["bodies"] = request
        captured["chunk_size"] = chunk_size
        captured["refresh"] = refresh
        return len(request), []

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", fake_bulk)
    monkeypatch.setattr("app.commons.os_client.utils.read_resource_file", lambda *args, **kwargs: {})

    client = OsClient(app_config, os_client=os_client_mock)
    updates = [
        TestItemHistoryData(
            test_item_id="ti-1",
            is_auto_analyzed=True,
            issue_type="pb001",
            timestamp="2025-01-02T00:00:00Z",
            issue_comment="comment",
        )
    ]

    response = client.bulk_update_issue_history(PROJECT_ID, updates)

    os_client_mock.indices.create.assert_not_called()
    assert response.errors is False

    bodies: list = captured["bodies"]
    assert len(bodies) == 1
    body = bodies[0]
    assert body["_op_type"] == "update"
    assert body["_index"] == get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    assert body["_id"] == "ti-1"
    assert body["script"]["params"]["entry"]["issue_type"] == "pb001"
    assert body["script"]["params"]["entry"]["timestamp"] == "2025-01-02T00:00:00Z"
    assert body["script"]["params"]["entry"]["issue_comment"] == "comment"


def test_bulk_update_issue_history_reports_missing_documents(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    missing_error = {
        "update": {
            "_index": PROJECT_ID,
            "_id": "ti-2",
            "status": 404,
            "error": {"type": "document_missing_exception", "reason": "[ti-2]: document missing"},
        }
    }
    bulk_mock = mock.Mock(return_value=(1, [missing_error]))
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", bulk_mock)

    client = OsClient(app_config, os_client=os_client_mock)
    updates = [
        TestItemHistoryData(test_item_id=item_id, is_auto_analyzed=True, issue_type="pb001", timestamp="2025-01-02")
        for item_id in ["ti-1", "ti-2"]
    ]

    response = client.bulk_update_issue_history(PROJECT_ID, updates)

    bulk_mock.assert_called_once()
    assert bulk_mock.call_args.kwargs["raise_on_error"] is False
    assert response.errors is True
    assert response.took == 1
    assert response.failed_ids == ["ti-2"]


def test_search_returns_empty_when_index_missing(os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing index")
    client = OsClient(app_config, os_client=os_client_mock)

    results = list(client.search(PROJECT_ID, {"query": {"match_all": {}}}))

    assert results == []
    os_client_mock.search.assert_not_called()


def test_msearch_grouped_returns_empty_when_index_missing(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing index")
    scan_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", scan_mock)
    client = OsClient(app_config, os_client=os_client_mock)

    results = list(client.msearch_grouped(PROJECT_ID, [{"query": {"match_all": {}}}]))

    assert results == []
    os_client_mock.msearch.assert_not_called()
    scan_mock.assert_not_called()


def test_delete_test_items_builds_terms_query(os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    os_client_mock.delete_by_query.return_value = {"deleted": 2}
    client = OsClient(app_config, os_client=os_client_mock)

    deleted = client.delete_test_items(PROJECT_ID, ["t1", "t2"])

    os_client_mock.delete_by_query.assert_called_once_with(
        index=get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix),
        body={"query": {"terms": {"test_item_id": ["t1", "t2"]}}},
        refresh=True,
        conflicts="proceed",
    )
    assert deleted == 2


def test_delete_by_launch_start_time_range_builds_range_query(os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    os_client_mock.delete_by_query.return_value = {"deleted": 3}
    client = OsClient(app_config, os_client=os_client_mock)

    deleted = client.delete_by_launch_start_time_range(PROJECT_ID, "2025-01-01", "2025-01-02")

    os_client_mock.delete_by_query.assert_called_once_with(
        index=get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix),
        body={"query": {"range": {"launch_start_time": {"gte": "2025-01-01", "lte": "2025-01-02"}}}},
        refresh=True,
        conflicts="proceed",
    )
    assert deleted == 3


def test_delete_test_items_retries_on_version_conflicts(os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    os_client_mock.delete_by_query.side_effect = [
        {"deleted": 1, "version_conflicts": 1},
        {"deleted": 1, "version_conflicts": 0},
    ]
    client = OsClient(app_config, os_client=os_client_mock)

    deleted = client.delete_test_items(PROJECT_ID, ["t1", "t2"])

    expected_call = mock.call(
        index=get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix),
        body={"query": {"terms": {"test_item_id": ["t1", "t2"]}}},
        refresh=True,
        conflicts="proceed",
    )
    assert os_client_mock.delete_by_query.call_args_list == [expected_call, expected_call]
    assert deleted == 2


def test_delete_by_launch_ids_builds_terms_query(os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    os_client_mock.delete_by_query.return_value = {"deleted": 2}
    client = OsClient(app_config, os_client=os_client_mock)

    deleted = client.delete_by_launch_ids(PROJECT_ID, ["l1", "l2"])

    os_client_mock.delete_by_query.assert_called_once_with(
        index=get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix),
        body={"query": {"terms": {"launch_id": ["l1", "l2"]}}},
        refresh=True,
        conflicts="proceed",
    )
    assert deleted == 2


def test_get_test_items_by_ids_returns_empty_when_index_missing(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing index")
    scan_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", scan_mock)
    client = OsClient(app_config, os_client=os_client_mock)

    result = client.get_test_items_by_ids(PROJECT_ID, ["t1", "t2"])

    assert result == []
    scan_mock.assert_not_called()


def test_get_test_items_by_ids_builds_terms_query(monkeypatch, os_client_mock, app_config, test_item):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, object] = {}

    def fake_scan(self, query, index):
        captured["client"] = self
        captured["query"] = query
        captured["index"] = index
        yield {"_source": test_item.to_index_dict(), "_id": test_item.test_item_id, "_index": index}

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", fake_scan)
    client = OsClient(app_config, os_client=os_client_mock)

    results = client.get_test_items_by_ids(PROJECT_ID, [test_item.test_item_id])

    assert len(results) == 1
    assert results[0].test_item_id == test_item.test_item_id
    assert captured["client"] is os_client_mock
    assert captured["index"] == get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    assert captured["query"] == {
        "size": app_config.esChunkNumber,
        "query": {"terms": {"test_item_id": [test_item.test_item_id]}},
    }


def test_search_without_scroll_calls_scan(monkeypatch, os_client_mock, app_config, test_item):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, object] = {}

    def fake_scan(self, query, index):
        captured["client"] = self
        captured["query"] = query
        captured["index"] = index
        yield {
            "_index": index,
            "_id": test_item.test_item_id,
            "_score": 1.0,
            "_source": test_item.to_index_dict(),
        }

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", fake_scan)
    client = OsClient(app_config, os_client=os_client_mock)
    request = {"query": {"match": {"test_item_id": test_item.test_item_id}}}

    results = list(client.search(PROJECT_ID, request))

    assert len(results) == 1
    assert results[0].source.test_item_id == test_item.test_item_id
    assert captured["client"] is os_client_mock
    assert captured["index"] == get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    assert captured["query"] == request


def test_search_calls_scan_when_size_provided(monkeypatch, os_client_mock, app_config, test_item):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, object] = {}

    def fake_scan(self, query, index, size):
        captured["client"] = self
        captured["query"] = query
        captured["index"] = index
        captured["size"] = size
        yield {
            "_index": index,
            "_id": test_item.test_item_id,
            "_score": 1.0,
            "_source": test_item.to_index_dict(),
        }

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", fake_scan)

    client = OsClient(app_config, os_client=os_client_mock)
    request = {"query": {"match": {"test_item_id": test_item.test_item_id}}}
    results = list(client.search(PROJECT_ID, request, size=100))

    assert len(results) == 1
    assert results[0].source.test_item_id == test_item.test_item_id
    assert captured["client"] is os_client_mock
    assert captured["index"] == get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    assert captured["query"] == request
    assert captured["size"] == 100


def _hit(index_name, test_item):
    return {
        "_index": index_name,
        "_id": test_item.test_item_id,
        "_score": 1.0,
        "_source": test_item.to_index_dict(),
    }


def test_msearch_grouped_calls_opensearch_msearch(monkeypatch, os_client_mock, app_config, test_item):
    os_client_mock.indices.get.return_value = {}
    scan_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", scan_mock)
    client = OsClient(app_config, os_client=os_client_mock)
    index_name = get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    queries = [{"query": {"match": {"test_item_id": test_item.test_item_id}}}]
    os_client_mock.msearch.return_value = {"responses": [{"hits": {"hits": [_hit(index_name, test_item)]}}]}

    results = list(client.msearch_grouped(PROJECT_ID, queries))

    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].source.test_item_id == test_item.test_item_id
    os_client_mock.msearch.assert_called_once_with(body=queries, index=index_name)
    scan_mock.assert_not_called()


def test_msearch_grouped_keeps_alignment_when_one_query_fails(os_client_mock, app_config, test_item):
    """A query that fails mid-batch must cost its own group only, not the ones after it."""
    os_client_mock.indices.get.return_value = {}
    client = OsClient(app_config, os_client=os_client_mock)
    index_name = get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    os_client_mock.msearch.return_value = {
        "responses": [
            {"hits": {"hits": [_hit(index_name, test_item)]}},
            {"status": 400, "error": {"type": "search_phase_execution_exception", "reason": "boom"}},
            {"hits": {"hits": [_hit(index_name, test_item)]}},
        ]
    }

    results = list(client.msearch_grouped(PROJECT_ID, [{}, {}, {}]))

    assert len(results) == 3, "a failed query must still occupy its position"
    assert len(results[0]) == 1
    assert results[1] == []
    assert len(results[2]) == 1


def test_msearch_grouped_keeps_alignment_when_a_hit_is_malformed(os_client_mock, app_config, test_item):
    """An unreadable response must not abort the responses that follow it."""
    os_client_mock.indices.get.return_value = {}
    client = OsClient(app_config, os_client=os_client_mock)
    index_name = get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    os_client_mock.msearch.return_value = {
        "responses": [
            {"hits": {"hits": [{"_id": "1", "_score": 1.0}]}},  # no _source -> validation error
            {"hits": {"hits": [_hit(index_name, test_item)]}},
        ]
    }

    results = list(client.msearch_grouped(PROJECT_ID, [{}, {}]))

    assert len(results) == 2
    assert results[0] == []
    assert len(results[1]) == 1
    assert results[1][0].source.test_item_id == test_item.test_item_id


def test_msearch_grouped_returns_nothing_when_the_request_fails(os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    os_client_mock.msearch.side_effect = Exception("transport blew up")
    client = OsClient(app_config, os_client=os_client_mock)

    assert list(client.msearch_grouped(PROJECT_ID, [{}, {}])) == []


def test_delete_index_calls_opensearch_delete(os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = lambda idx: None if idx.endswith("_suggest") else {}
    client = OsClient(app_config, os_client=os_client_mock)

    result = client.delete_index(PROJECT_ID)

    os_client_mock.indices.delete.assert_called_once_with(
        index=get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    )
    assert result is True


def test_delete_index_returns_false_on_exception(os_client_mock, app_config):
    os_client_mock.indices.delete.side_effect = Exception("boom")
    client = OsClient(app_config, os_client=os_client_mock)

    result = client.delete_index(PROJECT_ID)

    assert result is False


@pytest.mark.parametrize(
    "cluster_status, expected",
    [
        ("green", True),
        ("yellow", True),
        ("red", False),
    ],
)
def test_is_healthy_uses_os_client(os_client_mock, app_config, cluster_status, expected):
    os_client_mock.cluster.health.return_value = {"status": cluster_status}
    client = OsClient(app_config, os_client=os_client_mock)

    assert client.is_healthy() is expected
    os_client_mock.cluster.health.assert_called_once_with(params={"request_timeout": HEALTH_CHECK_TIMEOUT})


def test_is_healthy_returns_false_on_os_client_error(os_client_mock, app_config):
    os_client_mock.cluster.health.side_effect = OsConnectionError("N/A", "certificate verify failed", Exception())
    client = OsClient(app_config, os_client=os_client_mock)

    assert client.is_healthy() is False
    os_client_mock.cluster.health.assert_called_once()


def test_is_healthy_returns_false_without_host(os_client_mock, app_config):
    app_config.esHost = ""
    client = OsClient(app_config, os_client=os_client_mock)

    assert client.is_healthy() is False


def test_delete_by_launch_ids_returns_zero_when_index_missing(os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing")
    client = OsClient(app_config, os_client=os_client_mock)

    deleted = client.delete_by_launch_ids(PROJECT_ID, ["l1", "l2"])

    assert deleted == 0
    os_client_mock.delete_by_query.assert_not_called()


def test_delete_logs_by_ids_removes_logs(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, Any] = {}

    # noinspection PyUnusedLocal
    def fake_bulk(client: Any, bodies, chunk_size: int, request_timeout: int, refresh: bool, raise_on_error: bool):
        captured["client"] = client
        captured["bodies"] = bodies
        captured["chunk_size"] = chunk_size
        captured["refresh"] = refresh
        return len(bodies), []

    base_log = LogData(
        log_id="remove_me",
        log_order=0,
        log_time="2025-01-01T00:00:00Z",
        log_level=40000,
        cluster_id="",
        cluster_message="",
        cluster_with_numbers=False,
        original_message="original",
        message="clean",
        message_lines=1,
        message_words_number=1,
        message_extended="clean",
        message_without_params_extended="clean",
        message_without_params_and_brackets="clean",
        detected_message="clean",
        detected_message_with_numbers="clean",
        detected_message_extended="clean",
        detected_message_without_params_extended="clean",
        detected_message_without_params_and_brackets="clean",
        stacktrace="trace",
        stacktrace_extended="trace",
        only_numbers="",
        potential_status_codes="",
        found_exceptions="",
        found_exceptions_extended="",
        found_tests_and_methods="",
        urls="",
        paths="",
        message_params="",
        whole_message="whole",
    )
    keep_log = base_log.model_copy(update={"log_id": "keep", "log_time": "2025-02-01T00:00:00Z"})
    test_item = TestItemIndexData(
        test_item_id="ti-1",
        test_item_name="name",
        unique_id="uid",
        test_case_hash=123,
        launch_id="l-1",
        launch_name="launch",
        launch_number="1",
        launch_start_time="2025-01-01T00:00:00Z",
        is_auto_analyzed=True,
        issue_type="ab001",
        start_time="2025-01-01T00:00:00Z",
        log_count=2,
        logs=[base_log, keep_log],
        issue_history=[],
    )

    # noinspection PyUnusedLocal
    def fake_scan(client, query, index, **kwargs):
        return iter([{"_index": index, "_id": test_item.test_item_id, "_source": test_item.to_index_dict()}])

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", fake_scan)
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", fake_bulk)

    os_client = OsClient(app_config, os_client=os_client_mock)
    removed = os_client.delete_logs_by_ids(PROJECT_ID, ["remove_me"])

    assert removed == 1
    captured_bodies = captured["bodies"]
    assert len(captured_bodies) == 1
    body = captured_bodies[0]
    assert body["_op_type"] == "update"
    assert body["_index"] == get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    assert body["doc"]["log_count"] == 1
    assert body["doc"]["logs"][0]["log_id"] == "keep"


def test_bulk_update_cluster_info_updates_logs_and_merged_small_logs(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, Any] = {}

    # noinspection PyUnusedLocal
    def fake_bulk(
        os_client: Any, request: list, chunk_size: int, request_timeout: int, refresh: bool, raise_on_error: bool
    ):
        captured["bodies"] = request
        captured["chunk_size"] = chunk_size
        return len(request), []

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", fake_bulk)

    client = OsClient(app_config, os_client=os_client_mock)
    log_update = LogClusterData(
        log_id="1", test_item_id="ti-1", cluster_id="123", cluster_message="first\nsecond", cluster_with_numbers=True
    )
    merged_log = MergedLogData(
        log_ids=["1", "2"],
        log_level=40000,
        whole_message="first\nsecond",
        cluster_id="123",
        cluster_message="first\nsecond",
        cluster_with_numbers=True,
    )
    merged_update = MergedLogsClusterData(test_item_id="ti-1", merged_small_logs=[merged_log])

    response = client.bulk_update_cluster_info(PROJECT_ID, [log_update], merged_updates=[merged_update], chunk_size=7)

    assert response.errors is False
    assert captured["chunk_size"] == 7
    bodies: list = captured["bodies"]
    assert len(bodies) == 2
    index_name = get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    for body in bodies:
        assert body["_op_type"] == "update"
        assert body["_index"] == index_name
        assert body["_id"] == "ti-1"
    assert bodies[0]["script"]["params"] == log_update.to_update_params()
    assert "ctx._source.merged_small_logs = params.merged_small_logs" in bodies[1]["script"]["source"]
    assert bodies[1]["script"]["params"]["merged_small_logs"] == [merged_log.model_dump()]


def test_bulk_update_cluster_info_skips_empty_updates(monkeypatch, os_client_mock, app_config):
    bulk_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", bulk_mock)

    client = OsClient(app_config, os_client=os_client_mock)
    response = client.bulk_update_cluster_info(PROJECT_ID, [], merged_updates=[])

    assert response.errors is False
    bulk_mock.assert_not_called()
    os_client_mock.indices.get.assert_not_called()


def test_delete_logs_by_ids_removes_merged_small_logs_with_removed_logs(
    monkeypatch, os_client_mock, app_config, test_item
):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, Any] = {}

    # noinspection PyUnusedLocal
    def fake_bulk(client: Any, bodies, chunk_size: int, request_timeout: int, refresh: bool, raise_on_error: bool):
        captured["bodies"] = bodies
        return len(bodies), []

    base_log = test_item.logs[0]
    item = test_item.model_copy(
        update={
            "logs": [
                base_log.model_copy(update={"log_id": "remove_me", "log_order": 0}),
                base_log.model_copy(update={"log_id": "keep", "log_order": 1}),
                base_log.model_copy(update={"log_id": "keep_fatal", "log_order": 2, "log_level": 50000}),
            ],
            "merged_small_logs": [
                MergedLogData(log_ids=["remove_me", "keep"], log_level=40000, whole_message="a\nb"),
                MergedLogData(log_ids=["keep_fatal"], log_level=50000, whole_message="c"),
            ],
        }
    )

    # noinspection PyUnusedLocal
    def fake_scan(client, query, index, **kwargs):
        return iter([{"_index": index, "_id": item.test_item_id, "_source": item.to_index_dict()}])

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", fake_scan)
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", fake_bulk)

    os_client = OsClient(app_config, os_client=os_client_mock)
    removed = os_client.delete_logs_by_ids(PROJECT_ID, ["remove_me"])

    assert removed == 1
    bodies = captured["bodies"]
    assert len(bodies) == 1
    doc = bodies[0]["doc"]
    assert [log["log_id"] for log in doc["logs"]] == ["keep", "keep_fatal"]
    assert [merged["log_ids"] for merged in doc["merged_small_logs"]] == [["keep_fatal"]]


def test_delete_by_log_time_range_deletes_empty_items(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, Any] = {}

    # noinspection PyUnusedLocal
    def fake_bulk(client: Any, bodies, chunk_size: int, request_timeout: int, refresh: bool, raise_on_error: bool):
        captured["client"] = client
        captured["bodies"] = bodies
        captured["chunk_size"] = chunk_size
        captured["refresh"] = refresh
        return len(bodies), []

    log_one = LogData(
        log_id="log-1",
        log_order=0,
        log_time="2025-03-01T00:00:00Z",
        log_level=40000,
        cluster_id="",
        cluster_message="",
        cluster_with_numbers=False,
        original_message="original",
        message="clean",
        message_lines=1,
        message_words_number=1,
        message_extended="clean",
        message_without_params_extended="clean",
        message_without_params_and_brackets="clean",
        detected_message="clean",
        detected_message_with_numbers="clean",
        detected_message_extended="clean",
        detected_message_without_params_extended="clean",
        detected_message_without_params_and_brackets="clean",
        stacktrace="trace",
        stacktrace_extended="trace",
        only_numbers="",
        potential_status_codes="",
        found_exceptions="",
        found_exceptions_extended="",
        found_tests_and_methods="",
        urls="",
        paths="",
        message_params="",
        whole_message="whole",
    )
    log_two = log_one.model_copy(update={"log_id": "log-2", "log_time": "2025-03-02T00:00:00Z"})
    test_item = TestItemIndexData(
        test_item_id="ti-2",
        test_item_name="name",
        unique_id="uid",
        test_case_hash=321,
        launch_id="l-2",
        launch_name="launch",
        launch_number="2",
        launch_start_time="2025-03-01T00:00:00Z",
        is_auto_analyzed=False,
        issue_type="pb001",
        start_time="2025-03-01T00:00:00Z",
        log_count=2,
        logs=[log_one, log_two],
        issue_history=[],
    )

    # noinspection PyUnusedLocal
    def fake_scan(client, query, index, **kwargs):
        return iter([{"_index": index, "_id": test_item.test_item_id, "_source": test_item.to_index_dict()}])

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", fake_scan)
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", fake_bulk)

    os_client = OsClient(app_config, os_client=os_client_mock)
    removed = os_client.delete_by_log_time_range(PROJECT_ID, "2025-03-01", "2025-03-03")

    assert removed == 2
    captured_bodies = captured["bodies"]
    assert len(captured_bodies) == 1
    assert captured_bodies[0]["_op_type"] == "delete"
    assert captured_bodies[0]["_id"] == test_item.test_item_id


def test_create_os_client_passes_tls_settings(monkeypatch, app_config):
    app_config.esUseSsl = True
    app_config.esVerifyCerts = True
    app_config.esSslShowWarn = True
    app_config.esCAcert = "/certs/ca.pem"
    app_config.esClientCert = "/certs/client.pem"
    app_config.esClientKey = "/certs/client.key"
    app_config.esUser = "admin"
    app_config.esPassword = "secret"
    app_config.turnOffSslVerification = False
    opensearch_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.OpenSearch", opensearch_mock)

    OsClient(app_config)

    opensearch_mock.assert_called_once()
    args, kwargs = opensearch_mock.call_args
    assert args == ([app_config.esHost],)
    assert kwargs["use_ssl"] is True
    assert kwargs["verify_certs"] is True
    assert kwargs["ssl_show_warn"] is True
    assert kwargs["ca_certs"] == "/certs/ca.pem"
    assert kwargs["client_cert"] == "/certs/client.pem"
    assert kwargs["client_key"] == "/certs/client.key"
    assert kwargs["http_auth"] == ("admin", "secret")
    assert "connection_class" not in kwargs


def test_bulk_index_resets_read_only_settings_with_os_client(monkeypatch, os_client_mock, app_config, test_item):
    os_client_mock.indices.get.return_value = {"index": {}}
    bulk_mock = mock.Mock(side_effect=[TransportError(403, "cluster_block_exception"), (1, [])])
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", bulk_mock)

    client = OsClient(app_config, os_client=os_client_mock)
    response = client.bulk_index(PROJECT_ID, [test_item])

    assert bulk_mock.call_count == 2
    os_client_mock.indices.put_settings.assert_called_once_with(
        index="_all", body={"index.blocks.read_only_allow_delete": None}
    )
    assert response.took == 1
    assert response.errors is False


def test_bulk_index_retries_when_read_only_reset_fails(monkeypatch, os_client_mock, app_config, test_item):
    os_client_mock.indices.get.return_value = {"index": {}}
    os_client_mock.indices.put_settings.side_effect = OsConnectionError(
        "N/A", "certificate verify failed", Exception()
    )
    bulk_mock = mock.Mock(side_effect=[TransportError(403, "cluster_block_exception"), (1, [])])
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.bulk", bulk_mock)

    client = OsClient(app_config, os_client=os_client_mock)
    response = client.bulk_index(PROJECT_ID, [test_item])

    os_client_mock.indices.put_settings.assert_called_once()
    assert bulk_mock.call_count == 2
    assert response.took == 1
