#  Copyright 2025 EPAM Systems
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

from unittest import mock

import pytest
from opensearchpy import OpenSearch
from opensearchpy.client import IndicesClient

from app.commons.esclient import EsClient
from app.commons.model.launch_objects import CleanIndexStrIds, SuggestAnalysisResult
from app.service.suggest_info_service import (
    RP_SUGGEST_METRICS_INDEX_TEMPLATE,
    SuggestInfoService,
)
from app.utils.utils import read_json_file
from test import APP_CONFIG


@pytest.fixture
def test_data() -> dict:
    """Load test data with suggest info and scan results."""
    return read_json_file("test_res", "suggest_info_service_test_data.json", to_json=True)


@pytest.fixture
def mocked_opensearch_client() -> OpenSearch:
    """Create a mocked OpenSearch client instance."""
    mock_client = mock.Mock(OpenSearch)
    mock_client.indices = mock.Mock(IndicesClient)

    # Mock indices.get for index_exists checks
    mock_client.indices.get.return_value = {"rp_123_suggest": "exists"}

    # Mock indices.create for index creation
    mock_client.indices.create.return_value = {"acknowledged": True}

    # Mock delete_by_query for clean operations
    mock_client.delete_by_query.return_value = {"deleted": 3}

    # Mock delete for index deletion
    mock_client.indices.delete.return_value = {"acknowledged": True}

    return mock_client


@pytest.fixture
def suggest_info_service(mocked_opensearch_client: OpenSearch) -> SuggestInfoService:
    """Create SuggestInfoService with real EsClient and mocked OpenSearch client."""
    # Create real EsClient with mocked OpenSearch client
    es_client = EsClient(APP_CONFIG, es_client=mocked_opensearch_client)

    # Create SuggestInfoService with real EsClient
    service = SuggestInfoService(APP_CONFIG, es_client=es_client)

    return service


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_index_suggest_info_calls_correct_services(
    mock_bulk,
    suggest_info_service: SuggestInfoService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that index_suggest_info method calls internal services with correct arguments."""
    # Prepare test data
    suggest_info_list = [SuggestAnalysisResult(**item) for item in test_data["suggest_info_list"]]
    test_project_id = suggest_info_list[0].project
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}_suggest"

    # Configure mock_bulk to return success
    mock_bulk.return_value = (3, [])  # (success_count, errors)

    # Execute the method
    result = suggest_info_service.index_suggest_info(suggest_info_list)

    # Verify index_exists was called exactly twice: once for metrics index, once for project index
    assert (
        mocked_opensearch_client.indices.get.call_count == 2
    ), "indices.get should be called exactly twice: for metrics and project indexes"

    # Verify the exact index names that were checked
    indices_get_calls = [call[1]["index"] for call in mocked_opensearch_client.indices.get.call_args_list]
    assert len(indices_get_calls) == 2, "Should check exactly 2 indexes"
    assert (
        RP_SUGGEST_METRICS_INDEX_TEMPLATE in indices_get_calls
    ), f"Should check for {RP_SUGGEST_METRICS_INDEX_TEMPLATE} index"
    assert expected_index_name in indices_get_calls, f"Should check for {expected_index_name} index"

    # Verify bulk was called twice: once for project data, once for metrics data
    assert mock_bulk.call_count == 2, "bulk should be called twice: once for project data, once for metrics"

    # Verify first bulk call is for project data
    first_bulk_call = mock_bulk.call_args_list[0]
    bulk_bodies = first_bulk_call[0][1]
    assert len(bulk_bodies) == 3, "Should index 3 suggest info items"
    assert first_bulk_call[0][0] == mocked_opensearch_client, "bulk should use mocked OpenSearch client"

    # Verify bulk bodies structure for project data
    for body in bulk_bodies:
        assert "_index" in body, "Each body should have _index"
        assert "_source" in body, "Each body should have _source"
        assert body["_index"] == expected_index_name, "Body should use correct index name"
        source = body["_source"]
        assert "project" in source, "Body should have project"
        assert "testItem" in source, "Body should have testItem"
        assert "issueType" in source, "Body should have issueType"
        assert "savedDate" in source, "Body should have savedDate"
        assert "modelInfo" in source, "Body should have modelInfo"
        assert isinstance(source["modelInfo"], list), "modelInfo should be a list"
        assert "module_version" in source, "Body should have module_version"

    # Verify second bulk call is for metrics data
    second_bulk_call = mock_bulk.call_args_list[1]
    metrics_bodies = second_bulk_call[0][1]
    # We have 3 items, but only 2 should go to metrics (methodName != "auto_analysis")
    assert len(metrics_bodies) == 2, "Should index 2 metrics items (excluding auto_analysis)"
    assert second_bulk_call[0][0] == mocked_opensearch_client, "metrics bulk should use mocked OpenSearch client"

    # Verify metrics bodies structure
    for body in metrics_bodies:
        assert "_index" in body, "Each metrics body should have _index"
        assert body["_index"] == RP_SUGGEST_METRICS_INDEX_TEMPLATE, "Metrics should use correct index"
        assert "_source" in body, "Each metrics body should have _source"
        source = body["_source"]
        assert "reciprocalRank" in source, "Metrics should have reciprocalRank"
        assert "notFoundResults" in source, "Metrics should have notFoundResults"
        assert source["methodName"] != "auto_analysis", "Metrics should not include auto_analysis"

    # Verify result structure
    assert result is not None, "result should not be None"
    assert result.took > 0, "result should have took count"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_index_suggest_info_with_empty_list(
    mock_bulk,
    suggest_info_service: SuggestInfoService,
    mocked_opensearch_client: OpenSearch,
) -> None:
    """Test index_suggest_info with empty list."""
    # Execute with empty list
    result = suggest_info_service.index_suggest_info([])

    # Verify no OpenSearch operations were called
    mocked_opensearch_client.indices.get.assert_not_called()
    mocked_opensearch_client.indices.create.assert_not_called()
    mock_bulk.assert_not_called()

    # Verify result
    assert result.took == 0, "result should have took=0 for empty list"


def test_remove_suggest_info_calls_correct_services(
    suggest_info_service: SuggestInfoService,
    mocked_opensearch_client: OpenSearch,
) -> None:
    """Test that remove_suggest_info method calls internal services with correct arguments."""
    test_project_id = 123
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}_suggest"

    # Execute the method
    result = suggest_info_service.remove_suggest_info(test_project_id)

    # Verify delete_index was called with correct index name
    mocked_opensearch_client.indices.delete.assert_called_once()
    delete_call = mocked_opensearch_client.indices.delete.call_args
    assert delete_call[1]["index"] == expected_index_name, "Should delete correct index"

    # Verify result
    assert result is True, "result should be True when deletion succeeds"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_clean_suggest_info_logs_calls_correct_services(
    mock_scan,
    mock_bulk,
    suggest_info_service: SuggestInfoService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that clean_suggest_info_logs method calls internal services with correct arguments."""
    clean_index = CleanIndexStrIds(**test_data["clean_index_str_ids"])
    test_project_id = clean_index.project
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}_suggest"

    # Configure mock_scan to return scan results
    mock_scan.return_value = iter(test_data["scan_results_for_clean_logs"])

    # Configure mock_bulk to return success
    mock_bulk.return_value = (2, [])  # (success_count, errors)

    # Execute the method
    result = suggest_info_service.clean_suggest_info_logs(clean_index)

    # Verify index_exists was called with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify opensearchpy.helpers.scan was called to find suggest logs
    mock_scan.assert_called_once()
    scan_call = mock_scan.call_args
    assert scan_call[0][0] == mocked_opensearch_client, "scan should use mocked OpenSearch client"
    assert scan_call[1]["index"] == expected_index_name, f"scan should use index '{expected_index_name}'"

    # Verify scan query structure
    scan_query = scan_call[1]["query"]
    assert "query" in scan_query, "scan query should have 'query' key"
    assert "bool" in scan_query["query"], "scan query should have bool query"
    assert "should" in scan_query["query"]["bool"], "scan query should have should clause"

    # Verify the query searches for correct log IDs
    should_clause = scan_query["query"]["bool"]["should"]
    assert len(should_clause) == 2, "Should clause should have 2 terms queries"
    # Check for testItemLogId and relevantLogId terms
    test_item_log_id_query = next((q for q in should_clause if "terms" in q and "testItemLogId" in q["terms"]), None)
    relevant_log_id_query = next((q for q in should_clause if "terms" in q and "relevantLogId" in q["terms"]), None)
    assert test_item_log_id_query is not None, "Query should search testItemLogId"
    assert relevant_log_id_query is not None, "Query should search relevantLogId"
    assert test_item_log_id_query["terms"]["testItemLogId"] == clean_index.ids
    assert relevant_log_id_query["terms"]["relevantLogId"] == clean_index.ids

    # Verify opensearchpy.helpers.bulk was called to delete logs
    mock_bulk.assert_called_once()
    bulk_call = mock_bulk.call_args
    assert bulk_call[0][0] == mocked_opensearch_client, "bulk should use mocked OpenSearch client"

    # Verify bulk delete bodies
    bulk_bodies = bulk_call[0][1]
    assert len(bulk_bodies) == 2, "Should delete 2 suggest logs"

    # Verify delete body structure
    for body in bulk_bodies:
        assert body["_op_type"] == "delete", "Body should be a delete operation"
        assert "_id" in body, "Body should have _id"
        assert "_index" in body, "Body should have _index"
        assert body["_index"] == expected_index_name, "Delete body should use correct index name"

    # Verify result
    assert result > 0, "result should return took time"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_clean_suggest_info_logs_with_nonexistent_index(
    mock_scan,
    mock_bulk,
    suggest_info_service: SuggestInfoService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test clean_suggest_info_logs when index does not exist."""
    clean_index = CleanIndexStrIds(**test_data["clean_index_str_ids"])
    test_project_id = clean_index.project
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}_suggest"

    # Configure mock to raise exception for non-existent index
    mocked_opensearch_client.indices.get.side_effect = Exception("Index not found")

    # Execute the method
    result = suggest_info_service.clean_suggest_info_logs(clean_index)

    # Verify index_exists was called with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was NOT called since index doesn't exist
    mock_scan.assert_not_called()

    # Verify bulk was NOT called since index doesn't exist
    mock_bulk.assert_not_called()

    # Verify result is 0
    assert result == 0, "Should return 0 when index doesn't exist"


def test_clean_suggest_info_logs_by_test_item_calls_correct_services(
    suggest_info_service: SuggestInfoService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that clean_suggest_info_logs_by_test_item method calls internal services with correct arguments."""
    remove_items_info = test_data["remove_items_info"]
    test_project_id = remove_items_info["project"]
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}_suggest"

    # Execute the method
    result = suggest_info_service.clean_suggest_info_logs_by_test_item(remove_items_info)

    # Verify delete_by_query was called with correct index name
    mocked_opensearch_client.delete_by_query.assert_called_once()
    delete_call = mocked_opensearch_client.delete_by_query.call_args
    assert delete_call[1]["index"] == expected_index_name, "Should delete from correct index"

    # Verify query structure
    query = delete_call[1]["body"]
    assert "query" in query, "delete_by_query should have query"
    assert "bool" in query["query"], "query should have bool clause"
    assert "should" in query["query"]["bool"], "query should have should clause"

    # Verify the query searches for correct test items
    should_clause = query["query"]["bool"]["should"]
    assert len(should_clause) == 2, "Should clause should have 2 terms queries"
    # Check for testItem and relevantItem terms
    test_item_query = next((q for q in should_clause if "terms" in q and "testItem" in q["terms"]), None)
    relevant_item_query = next((q for q in should_clause if "terms" in q and "relevantItem" in q["terms"]), None)
    assert test_item_query is not None, "Query should search testItem"
    assert relevant_item_query is not None, "Query should search relevantItem"
    assert test_item_query["terms"]["testItem"] == remove_items_info["itemsToDelete"]
    assert relevant_item_query["terms"]["relevantItem"] == remove_items_info["itemsToDelete"]

    # Verify result
    assert result == 3, "result should return number of deleted items"


def test_clean_suggest_info_logs_by_launch_id_calls_correct_services(
    suggest_info_service: SuggestInfoService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that clean_suggest_info_logs_by_launch_id method calls internal services with correct arguments."""
    launch_remove_info = test_data["launch_remove_info"]
    test_project_id = launch_remove_info["project"]
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}_suggest"

    # Execute the method
    result = suggest_info_service.clean_suggest_info_logs_by_launch_id(launch_remove_info)

    # Verify delete_by_query was called with correct index name
    mocked_opensearch_client.delete_by_query.assert_called_once()
    delete_call = mocked_opensearch_client.delete_by_query.call_args
    assert delete_call[1]["index"] == expected_index_name, "Should delete from correct index"

    # Verify query structure
    query = delete_call[1]["body"]
    assert "query" in query, "delete_by_query should have query"
    assert "bool" in query["query"], "query should have bool clause"
    assert "filter" in query["query"]["bool"], "query should have filter clause"

    # Verify the query searches for correct launch IDs
    filter_clause = query["query"]["bool"]["filter"]
    assert len(filter_clause) == 1, "Filter should have 1 terms query"
    launch_id_query = filter_clause[0]
    assert "terms" in launch_id_query, "Filter should use terms query"
    assert "launchId" in launch_id_query["terms"], "Terms query should filter launchId field"
    assert launch_id_query["terms"]["launchId"] == launch_remove_info["launch_ids"]

    # Verify result
    assert result == 3, "result should return number of deleted items"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_update_suggest_info_calls_correct_services(
    mock_scan,
    mock_bulk,
    suggest_info_service: SuggestInfoService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that update_suggest_info method calls internal services with correct arguments."""
    defect_update_info = test_data["defect_update_info"]
    test_project_id = defect_update_info["project"]
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}_suggest"

    # Configure mock_scan to return scan results
    mock_scan.return_value = iter(test_data["scan_results_for_update"])

    # Configure mock_bulk to return success
    mock_bulk.return_value = (3, [])  # (success_count, errors)

    # Execute the method
    result = suggest_info_service.update_suggest_info(defect_update_info)

    # Verify index_exists was called with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify opensearchpy.helpers.scan was called to get suggest info
    mock_scan.assert_called_once()
    scan_call = mock_scan.call_args
    assert scan_call[0][0] == mocked_opensearch_client, "scan should use mocked OpenSearch client"
    assert scan_call[1]["index"] == expected_index_name, f"scan should use index '{expected_index_name}'"

    # Verify scan query structure
    scan_query = scan_call[1]["query"]
    assert "query" in scan_query, "scan query should have 'query' key"
    assert "bool" in scan_query["query"], "scan query should have bool query"
    assert "must" in scan_query["query"]["bool"], "scan query should have must clause"

    # Verify the query filters correctly
    must_clause = scan_query["query"]["bool"]["must"]
    assert len(must_clause) == 3, "Must clause should have 3 conditions"
    # Check for testItem terms query
    test_item_query = next((q for q in must_clause if "terms" in q and "testItem" in q["terms"]), None)
    assert test_item_query is not None, "Query should filter by testItem"
    # Check for methodName term query
    method_name_query = next((q for q in must_clause if "term" in q and "methodName" in q["term"]), None)
    assert method_name_query is not None, "Query should filter by methodName"
    assert method_name_query["term"]["methodName"] == "auto_analysis"
    # Check for userChoice term query
    user_choice_query = next((q for q in must_clause if "term" in q and "userChoice" in q["term"]), None)
    assert user_choice_query is not None, "Query should filter by userChoice"
    assert user_choice_query["term"]["userChoice"] == 1

    # Verify opensearchpy.helpers.bulk was called to update logs
    mock_bulk.assert_called_once()
    bulk_call = mock_bulk.call_args
    assert bulk_call[0][0] == mocked_opensearch_client, "bulk should use mocked OpenSearch client"

    # Verify bulk update bodies
    bulk_bodies = bulk_call[0][1]
    assert len(bulk_bodies) == 3, "Should update 3 suggest info items"

    # Verify update body structure
    for body in bulk_bodies:
        assert body["_op_type"] == "update", "Body should be an update operation"
        assert "_id" in body, "Body should have _id"
        assert "_index" in body, "Body should have _index"
        assert "doc" in body, "Body should have doc with update fields"
        assert "userChoice" in body["doc"], "Update should include userChoice"
        assert body["doc"]["userChoice"] == 0, "userChoice should be set to 0"

    # Verify result
    assert result > 0, "result should return took time"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_update_suggest_info_with_nonexistent_index(
    mock_scan,
    mock_bulk,
    suggest_info_service: SuggestInfoService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test update_suggest_info when index does not exist."""
    defect_update_info = test_data["defect_update_info"]
    test_project_id = defect_update_info["project"]
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}_suggest"

    # Configure mock to raise exception for non-existent index
    mocked_opensearch_client.indices.get.side_effect = Exception("Index not found")

    # Execute the method
    result = suggest_info_service.update_suggest_info(defect_update_info)

    # Verify index_exists was called with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was NOT called since index doesn't exist
    mock_scan.assert_not_called()

    # Verify bulk was NOT called since index doesn't exist
    mock_bulk.assert_not_called()

    # Verify result is 0
    assert result == 0, "Should return 0 when index doesn't exist"


# noinspection PyUnresolvedReferences
@mock.patch("app.service.suggest_info_service.AmqpClient")
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_update_suggest_info_with_amqp_enabled(
    mock_scan,
    mock_bulk,
    mock_amqp_client_class,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that update_suggest_info calls AmqpClient when amqpUrl is configured."""
    # Create a config with amqpUrl enabled
    from app.commons.model.launch_objects import ApplicationConfig

    app_config_with_amqp = ApplicationConfig(**{**APP_CONFIG.dict(), "amqpUrl": "amqp://localhost:5672"})

    # Create service with AMQP enabled
    es_client = EsClient(app_config_with_amqp, es_client=mocked_opensearch_client)
    service = SuggestInfoService(app_config_with_amqp, es_client=es_client)

    defect_update_info = test_data["defect_update_info"]

    # Configure mocks
    mock_scan.return_value = iter(test_data["scan_results_for_update"])
    mock_bulk.return_value = (3, [])

    # Create mock AMQP client instance
    mock_amqp_instance = mock.Mock()
    mock_amqp_client_class.return_value = mock_amqp_instance

    # Execute the method
    result = service.update_suggest_info(defect_update_info)

    # Verify AmqpClient was instantiated
    mock_amqp_client_class.assert_called_once_with(app_config_with_amqp)

    # Verify send_to_inner_queue was called twice (for suggestion and auto_analysis models)
    assert mock_amqp_instance.send_to_inner_queue.call_count == 2, "Should send train requests for 2 model types"

    # Verify the train requests
    train_calls = mock_amqp_instance.send_to_inner_queue.call_args_list
    routing_keys = [call[0][0] for call in train_calls]
    assert all(key == "train_models" for key in routing_keys), "All calls should use 'train_models' routing key"

    # Verify close was called
    mock_amqp_instance.close.assert_called_once()

    # Verify result
    assert result > 0, "result should return took time"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_index_suggest_info_creates_indexes_if_not_exist(
    mock_bulk,
    suggest_info_service: SuggestInfoService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that index_suggest_info creates indexes if they don't exist."""
    suggest_info_list = [SuggestAnalysisResult(**item) for item in test_data["suggest_info_list"]]
    test_project_id = suggest_info_list[0].project
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}_suggest"

    # Configure mock to indicate indexes don't exist
    mocked_opensearch_client.indices.get.side_effect = Exception("Index not found")

    # Configure mock_bulk to return success
    mock_bulk.return_value = (3, [])

    # Execute the method
    result = suggest_info_service.index_suggest_info(suggest_info_list)

    # Verify indices.get was called for both indexes
    assert mocked_opensearch_client.indices.get.call_count == 2, "Should check for both metrics and project indexes"

    # Verify indices.create was called twice
    assert mocked_opensearch_client.indices.create.call_count == 2, "Should create both metrics and project indexes"

    # Verify the indexes being created
    create_calls = [call[1]["index"] for call in mocked_opensearch_client.indices.create.call_args_list]
    assert RP_SUGGEST_METRICS_INDEX_TEMPLATE in create_calls, "Should create metrics index"
    assert expected_index_name in create_calls, "Should create project suggest index"

    # Verify indexing still proceeded
    assert mock_bulk.call_count == 2, "Should still index data after creating indexes"
    assert result.took > 0, "Should return success result"
