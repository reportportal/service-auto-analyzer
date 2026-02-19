from typing import Any
from unittest import mock

import pytest

from app.commons.model.test_item_index import LogData, TestItemHistoryData, TestItemIndexData
from app.commons.os_client import OsClient, get_test_item_index_name
from test import APP_CONFIG, DEFAULT_ES_CONFIG

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
    def fake_bulk(self: Any, request: list, chunk_size: int, request_timeout: int, refresh: bool):
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
    def fake_bulk(client: Any, request: list, chunk_size: int, request_timeout: int, refresh: bool):
        captured["client"] = client
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


def test_search_returns_empty_when_index_missing(os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing index")
    client = OsClient(app_config, os_client=os_client_mock)

    results = list(client.search(PROJECT_ID, {"query": {"match_all": {}}}))

    assert results == []
    os_client_mock.search.assert_not_called()


def test_msearch_returns_empty_when_index_missing(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing index")
    scan_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", scan_mock)
    client = OsClient(app_config, os_client=os_client_mock)

    results = list(client.msearch(PROJECT_ID, [{"query": {"match_all": {}}}]))

    assert results == []
    os_client_mock.msearch.assert_not_called()
    scan_mock.assert_not_called()


def test_get_test_item_returns_none_when_index_missing(os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing index")
    client = OsClient(app_config, os_client=os_client_mock)

    result = client.get_test_item(PROJECT_ID, "ti-1")

    assert result is None
    os_client_mock.get.assert_not_called()


def test_get_launch_ids_returns_empty_when_index_missing(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing index")
    scan_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", scan_mock)
    client = OsClient(app_config, os_client=os_client_mock)

    result = client.get_launch_ids_by_start_time_range(PROJECT_ID, "2025-01-01", "2025-01-02")

    assert result == []
    scan_mock.assert_not_called()


def test_delete_test_items_builds_terms_query(os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    os_client_mock.delete_by_query.return_value = {"deleted": 2}
    client = OsClient(app_config, os_client=os_client_mock)

    deleted = client.delete_test_items(PROJECT_ID, ["t1", "t2"])

    os_client_mock.delete_by_query.assert_called_once_with(
        index=get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix),
        body={"query": {"terms": {"test_item_id": ["t1", "t2"]}}},
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
    )
    assert deleted == 3


def test_delete_by_test_item_start_time_range_builds_range_query(os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    os_client_mock.delete_by_query.return_value = {"deleted": 4}
    client = OsClient(app_config, os_client=os_client_mock)

    deleted = client.delete_by_test_item_start_time_range(PROJECT_ID, "2025-01-01", "2025-01-02")

    os_client_mock.delete_by_query.assert_called_once_with(
        index=get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix),
        body={"query": {"range": {"start_time": {"gte": "2025-01-01", "lte": "2025-01-02"}}}},
    )
    assert deleted == 4


def test_delete_by_launch_ids_builds_terms_query(os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    os_client_mock.delete_by_query.return_value = {"deleted": 2}
    client = OsClient(app_config, os_client=os_client_mock)

    deleted = client.delete_by_launch_ids(PROJECT_ID, ["l1", "l2"])

    os_client_mock.delete_by_query.assert_called_once_with(
        index=get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix),
        body={"query": {"terms": {"launch_id": ["l1", "l2"]}}},
    )
    assert deleted == 2


def test_get_launch_ids_by_start_time_range_builds_query(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, object] = {}

    def fake_scan(self, query, index):
        captured["client"] = self
        captured["query"] = query
        captured["index"] = index
        yield {"_source": {"launch_id": "l1", "test_item_id": "t1"}, "_index": index, "_id": "t1"}
        yield {"_source": {"launch_id": "l2", "test_item_id": "t2"}, "_index": index, "_id": "t2"}

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", fake_scan)

    client = OsClient(app_config, os_client=os_client_mock)
    result = client.get_launch_ids_by_start_time_range(PROJECT_ID, "2025-01-01", "2025-01-02")

    assert set(result) == {"l1", "l2"}
    assert captured["client"] is os_client_mock
    assert captured["index"] == get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    assert captured["query"] == {
        "_source": ["launch_id", "test_item_id"],
        "query": {"range": {"launch_start_time": {"gte": "2025-01-01", "lte": "2025-01-02"}}},
        "size": app_config.esChunkNumber,
    }


def test_get_test_item_returns_instance(os_client_mock, app_config, test_item):
    os_client_mock.indices.get.return_value = {}
    os_client_mock.get.return_value = {"_source": test_item.to_index_dict()}
    client = OsClient(app_config, os_client=os_client_mock)

    result = client.get_test_item(PROJECT_ID, test_item.test_item_id)

    os_client_mock.get.assert_called_once_with(
        index=get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix),
        id=test_item.test_item_id,
    )
    assert result is not None
    assert result.test_item_id == test_item.test_item_id
    assert result.launch_id == test_item.launch_id


def test_get_test_item_ids_returns_empty_when_index_missing(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing index")
    scan_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", scan_mock)
    client = OsClient(app_config, os_client=os_client_mock)

    result = client.get_test_item_ids_by_start_time_range(PROJECT_ID, "2025-01-01", "2025-01-02")

    assert result == []
    scan_mock.assert_not_called()


def test_get_test_item_ids_by_start_time_range_builds_query(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, object] = {}

    def fake_scan(self, query, index):
        captured["client"] = self
        captured["query"] = query
        captured["index"] = index
        yield {"_source": {"launch_id": "l1", "test_item_id": "t1"}, "_index": index, "_id": "t1"}
        yield {"_source": {"launch_id": "l2", "test_item_id": "t2"}, "_index": index, "_id": "t2"}

    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", fake_scan)

    client = OsClient(app_config, os_client=os_client_mock)
    result = client.get_test_item_ids_by_start_time_range(PROJECT_ID, "2025-01-01", "2025-01-02")

    assert set(result) == {"t1", "t2"}
    assert captured["client"] is os_client_mock
    assert captured["index"] == get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    assert captured["query"] == {
        "_source": ["launch_id", "test_item_id"],
        "query": {"range": {"start_time": {"gte": "2025-01-01", "lte": "2025-01-02"}}},
        "size": app_config.esChunkNumber,
    }


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


def test_msearch_calls_opensearch_msearch(monkeypatch, os_client_mock, app_config, test_item):
    os_client_mock.indices.get.return_value = {}
    scan_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", scan_mock)
    client = OsClient(app_config, os_client=os_client_mock)
    index_name = get_test_item_index_name(PROJECT_ID, app_config.esProjectIndexPrefix)
    queries = [{"query": {"match": {"test_item_id": test_item.test_item_id}}}]
    os_client_mock.msearch.return_value = {
        "responses": [
            {
                "hits": {
                    "hits": [
                        {
                            "_index": index_name,
                            "_id": test_item.test_item_id,
                            "_score": 1.0,
                            "_source": test_item.to_index_dict(),
                        }
                    ]
                }
            }
        ]
    }

    results = list(client.msearch(PROJECT_ID, queries))

    assert len(results) == 1
    assert results[0].source.test_item_id == test_item.test_item_id
    os_client_mock.msearch.assert_called_once_with(body=queries, index=index_name)
    scan_mock.assert_not_called()


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


def test_is_healthy_calls_send_request(monkeypatch, os_client_mock, app_config):
    captured: dict[str, object] = {}

    def fake_send_request(url, method, user, password):
        captured["url"] = url
        captured["method"] = method
        captured["user"] = user
        captured["password"] = password
        return {"status": "green"}

    monkeypatch.setattr("app.commons.os_client.utils.send_request", fake_send_request)
    client = OsClient(app_config, os_client=os_client_mock)

    assert client.is_healthy() is True
    assert captured["url"] == "http://localhost:9200/_cluster/health"
    assert captured["method"] == "GET"
    assert captured["user"] == app_config.esUser
    assert captured["password"] == app_config.esPassword


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


def test_delete_by_test_item_start_time_range_returns_zero_when_index_missing(os_client_mock, app_config):
    os_client_mock.indices.get.side_effect = Exception("missing")
    client = OsClient(app_config, os_client=os_client_mock)

    deleted = client.delete_by_test_item_start_time_range(PROJECT_ID, "2025-01-01", "2025-01-02")

    assert deleted == 0
    os_client_mock.delete_by_query.assert_not_called()


def test_get_test_item_ids_by_start_time_range_returns_empty_when_index_missing(
    monkeypatch, os_client_mock, app_config
):
    os_client_mock.indices.get.side_effect = Exception("missing")
    scan_mock = mock.Mock()
    monkeypatch.setattr("app.commons.os_client.opensearchpy.helpers.scan", scan_mock)
    client = OsClient(app_config, os_client=os_client_mock)

    result = client.get_test_item_ids_by_start_time_range(PROJECT_ID, "2025-01-01", "2025-01-02")

    assert result == []
    scan_mock.assert_not_called()


def test_delete_logs_by_ids_removes_logs(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, Any] = {}

    # noinspection PyUnusedLocal
    def fake_bulk(client: Any, bodies, chunk_size: int, request_timeout: int, refresh: bool):
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


def test_delete_by_log_time_range_deletes_empty_items(monkeypatch, os_client_mock, app_config):
    os_client_mock.indices.get.return_value = {}
    captured: dict[str, Any] = {}

    # noinspection PyUnusedLocal
    def fake_bulk(client: Any, bodies, chunk_size: int, request_timeout: int, refresh: bool):
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


@pytest.mark.parametrize(
    "host, use_ssl, expected",
    [
        # 1. No protocol specified in the host param
        ("elastic_host", False, "http://elastic_host"),  # NOSONAR
        ("elastic_host", True, "https://elastic_host"),  # NOSONAR
        # 2. http protocol specified in the host parameter
        ("http://elastic_host", False, "http://elastic_host"),  # NOSONAR
        ("http://elastic_host", True, "http://elastic_host"),  # NOSONAR
        # 3. No protocol, but basic HTTP credentials are present in host parameter
        ("username:password@elastic_host", False, "http://username:password@elastic_host"),  # NOSONAR
        ("username:password@elastic_host", True, "https://username:password@elastic_host"),  # NOSONAR
        # 4. Protocol and credentials are present in host parameter -> same URL, no changes
        ("http://username:password@elastic_host", False, "http://username:password@elastic_host"),  # NOSONAR
        ("http://username:password@elastic_host", True, "http://username:password@elastic_host"),  # NOSONAR
    ],
)
def test_get_base_url_variations(host, use_ssl, expected):
    os_mock = mock.Mock()
    config = DEFAULT_ES_CONFIG.model_copy(
        update={
            "esHost": host,
            "esUseSsl": use_ssl,
        }
    )
    client = OsClient(config, os_client=os_mock)
    # Accessing name-mangled private method
    base_url = client._get_base_url()
    assert base_url == expected
