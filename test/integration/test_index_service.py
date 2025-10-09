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
from app.commons.model.launch_objects import Launch
from app.service.index_service import IndexService
from app.utils.utils import read_json_file
from test import APP_CONFIG


@pytest.fixture
def test_data() -> dict:
    """Load test data with launches and scan results."""
    return read_json_file("test_res", "index_service_test_data.json", to_json=True)


@pytest.fixture
def mocked_opensearch_client() -> OpenSearch:
    """Create a mocked OpenSearch client instance."""
    mock_client = mock.Mock(OpenSearch)
    mock_client.indices = mock.Mock(IndicesClient)

    # Mock indices.get for index_exists checks
    mock_client.indices.get.return_value = {"rp_123": "exists"}

    # Mock indices.create for index creation
    mock_client.indices.create.return_value = {"acknowledged": True}

    return mock_client


@pytest.fixture
def index_service(mocked_opensearch_client: OpenSearch) -> IndexService:
    """Create IndexService with real EsClient and mocked OpenSearch client."""
    # Create real EsClient with mocked OpenSearch client
    es_client = EsClient(APP_CONFIG, es_client=mocked_opensearch_client)

    # Create IndexService with real EsClient
    service = IndexService(APP_CONFIG, es_client=es_client)

    return service


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_index_logs_calls_correct_services(
    mock_scan,
    mock_bulk,
    index_service: IndexService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that index_logs method calls internal services with correct arguments."""
    # Prepare test data
    launches = [Launch(**launch_data) for launch_data in test_data["launches"]]
    test_project_id = launches[0].project
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"

    # Configure mock_scan to return different results for different calls:
    # 1st call: _delete_merged_logs (searches for is_merged=True logs to delete)
    # 2nd call: merge_logs (searches for is_merged=False logs to merge)
    mock_scan.side_effect = [
        iter(test_data["scan_results_for_delete_merged"]),
        iter(test_data["scan_results_for_merge"]),
    ]

    # Configure mock_bulk to return success
    mock_bulk.return_value = (3, [])  # (success_count, errors)

    # Execute the method
    result = index_service.index_logs(launches)

    # Verify index_exists was called exactly once with correct index name (via create_index_if_not_exists)
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify opensearchpy.helpers.bulk was called three times:
    # 1) Initial log indexing
    # 2) Delete old merged logs
    # 3) Create/update merged logs
    assert mock_bulk.call_count == 3, "bulk should be called three times: index, delete merged, and merge"

    # Verify first bulk call is for indexing logs
    first_bulk_call = mock_bulk.call_args_list[0]
    bulk_bodies = first_bulk_call[0][1]
    assert len(bulk_bodies) == 3, "Should index 3 logs (2 from first test item, 1 from second)"
    assert first_bulk_call[0][0] == mocked_opensearch_client, "bulk should use mocked OpenSearch client"

    # Verify bulk bodies structure for initial indexing
    for body in bulk_bodies:
        assert "_index" in body, "Each body should have _index"
        assert "_source" in body, "Each body should have _source"
        assert body["_index"] == expected_index_name, "Body should use correct index name"
        source = body["_source"]
        assert "launch_id" in source, "Body should have launch_id"
        assert "test_item" in source, "Body should have test_item"
        assert "message" in source, "Body should have message"
        assert "detected_message" in source, "Body should have detected_message"

    # Verify second bulk call is for deleting old merged logs
    second_bulk_call = mock_bulk.call_args_list[1]
    delete_bodies = second_bulk_call[0][1]
    assert len(delete_bodies) == 2, "Should delete 2 old merged logs"
    assert second_bulk_call[0][0] == mocked_opensearch_client, "delete bulk should use mocked OpenSearch client"

    # Verify delete bodies structure
    for body in delete_bodies:
        assert "_op_type" in body, "Each delete body should have _op_type"
        assert body["_op_type"] == "delete", "Operation should be delete"
        assert "_id" in body, "Delete body should have _id"
        assert "_index" in body, "Delete body should have _index"
        assert body["_index"] == expected_index_name, "Delete body should use correct index name"

    # Verify third bulk call is for merge_logs operation (updating merged logs)
    third_bulk_call = mock_bulk.call_args_list[2]
    merge_bodies = third_bulk_call[0][1]
    assert third_bulk_call[0][0] == mocked_opensearch_client, "merge bulk should use mocked OpenSearch client"

    # Verify merge bodies structure (should be update operations only)
    for body in merge_bodies:
        assert "_op_type" in body, "Merge body should have _op_type"
        assert body["_op_type"] == "update", "Merge operation should be update"
        assert "_id" in body, "Merge body should have _id"
        assert "_index" in body, "Merge body should have _index"
        assert body["_index"] == expected_index_name, "Merge body should use correct index name"
        assert "doc" in body, "Merge body should have doc field with update data"
        assert "merged_small_logs" in body["doc"], "Update should include merged_small_logs field"

    # Verify opensearchpy.helpers.scan was called twice for merge_logs
    assert mock_scan.call_count == 2, "scan should be called twice: for delete and for merge"

    # Verify first scan call is for _delete_merged_logs (is_merged=True)
    first_scan_call = mock_scan.call_args_list[0]
    assert first_scan_call[0][0] == mocked_opensearch_client, "first scan should use mocked OpenSearch client"
    assert first_scan_call[1]["index"] == expected_index_name, f"first scan should use index '{expected_index_name}'"

    # Verify first scan query searches for is_merged=True logs
    first_scan_query = first_scan_call[1]["query"]
    assert "query" in first_scan_query, "first scan query should have 'query' key"
    assert "bool" in first_scan_query["query"], "first scan query should have bool query"
    assert "filter" in first_scan_query["query"]["bool"], "first scan query should have filter clause"
    # Check for is_merged=True filter
    filters = first_scan_query["query"]["bool"]["filter"]
    is_merged_filter = next((f for f in filters if "term" in f and "is_merged" in f["term"]), None)
    assert is_merged_filter is not None, "first scan should filter by is_merged"
    assert is_merged_filter["term"]["is_merged"] is True, "first scan should search for is_merged=True"

    # Verify second scan call is for merge_logs (is_merged=False)
    second_scan_call = mock_scan.call_args_list[1]
    assert second_scan_call[0][0] == mocked_opensearch_client, "second scan should use mocked OpenSearch client"
    assert second_scan_call[1]["index"] == expected_index_name, f"second scan should use index '{expected_index_name}'"

    # Verify second scan query searches for is_merged=False logs
    second_scan_query = second_scan_call[1]["query"]
    assert "query" in second_scan_query, "second scan query should have 'query' key"
    assert "bool" in second_scan_query["query"], "second scan query should have bool query"
    assert "filter" in second_scan_query["query"]["bool"], "second scan query should have filter clause"
    # Check for is_merged=False filter
    filters = second_scan_query["query"]["bool"]["filter"]
    is_merged_filter = next((f for f in filters if "term" in f and "is_merged" in f["term"]), None)
    assert is_merged_filter is not None, "second scan should filter by is_merged"
    assert is_merged_filter["term"]["is_merged"] is False, "second scan should search for is_merged=False"

    # Verify result structure
    assert result is not None, "result should not be None"
    assert result.took > 0, "result should have took count"
    assert result.errors is False, "result should not have errors"
    assert result.logResults is not None, "result should have logResults"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_index_logs_with_empty_launches(
    mock_scan,
    mock_bulk,
    index_service: IndexService,
    mocked_opensearch_client: OpenSearch,
) -> None:
    """Test index_logs with empty launches list."""
    # Execute with empty launches
    result = index_service.index_logs([])

    # Verify no OpenSearch operations were called
    mocked_opensearch_client.indices.get.assert_not_called()
    mocked_opensearch_client.indices.create.assert_not_called()
    mock_scan.assert_not_called()
    mock_bulk.assert_not_called()

    # Verify result
    assert result.took == 0, "result should have took=0 for empty launches"
    assert result.errors is False, "result should not have errors"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_send_stats_info_calls_correct_services(
    mock_bulk,
    index_service: IndexService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that send_stats_info method calls internal services with correct arguments."""
    stats_info = test_data["stats_info"]

    # Configure mock_bulk to return success
    mock_bulk.return_value = (2, [])  # (success_count, errors)

    # Execute the method
    index_service.send_stats_info(stats_info)

    # Verify create_index_for_stats_info was called for each stat type
    # Should be called exactly twice: once for rp_aa_stats and once for rp_model_train_stats
    assert mocked_opensearch_client.indices.get.call_count == 2, "indices.get should be called twice for stats indexes"
    
    # Verify indices.get was called for correct indexes
    indices_get_calls = [call[1]["index"] for call in mocked_opensearch_client.indices.get.call_args_list]
    assert "rp_aa_stats" in indices_get_calls, "Should check for rp_aa_stats index"
    assert "rp_model_train_stats" in indices_get_calls, "Should check for rp_model_train_stats index"

    # Verify bulk was called for indexing stats
    mock_bulk.assert_called_once()
    bulk_call = mock_bulk.call_args
    assert bulk_call[0][0] == mocked_opensearch_client, "bulk should use mocked OpenSearch client"

    # Verify bulk bodies
    bulk_bodies = bulk_call[0][1]
    assert len(bulk_bodies) == 2, "Should index 2 stats"

    # Verify stat bodies have correct structure
    for body in bulk_bodies:
        assert "_index" in body, "Each body should have _index"
        assert "_source" in body, "Each body should have _source"
        assert body["_index"] in ["rp_aa_stats", "rp_model_train_stats"], "Body should use correct stats index"

    # Verify first stat goes to rp_aa_stats (method=analysis)
    assert bulk_bodies[0]["_index"] == "rp_aa_stats", "Analysis stats should go to rp_aa_stats index"
    assert bulk_bodies[0]["_source"]["method"] == "analysis"

    # Verify second stat goes to rp_model_train_stats (method=training)
    assert bulk_bodies[1]["_index"] == "rp_model_train_stats", "Training stats should go to rp_model_train_stats index"
    assert bulk_bodies[1]["_source"]["method"] == "training"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_defect_update_calls_correct_services(
    mock_scan,
    mock_bulk,
    index_service: IndexService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that defect_update method calls internal services with correct arguments."""
    defect_update_info = test_data["defect_update_info"]
    test_project_id = defect_update_info["project"]
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"

    # Configure mock_scan to return scan results
    mock_scan.return_value = iter(test_data["scan_results_for_defect_update"])

    # Configure mock_bulk to return success
    mock_bulk.return_value = (2, [])  # (success_count, errors)

    # Execute the method
    result = index_service.defect_update(defect_update_info)

    # Verify index_exists was called exactly once with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify opensearchpy.helpers.scan was called to get logs
    mock_scan.assert_called_once()
    scan_call = mock_scan.call_args
    assert scan_call[0][0] == mocked_opensearch_client, "scan should use mocked OpenSearch client"
    assert scan_call[1]["index"] == expected_index_name, f"scan should use index '{expected_index_name}'"

    # Verify scan query structure
    scan_query = scan_call[1]["query"]
    assert "query" in scan_query, "scan query should have 'query' key"
    assert "bool" in scan_query["query"], "scan query should have bool query"
    assert "filter" in scan_query["query"]["bool"], "scan query should have filter clause"

    # Verify the query searches for correct test items using terms filter
    filter_clause = scan_query["query"]["bool"]["filter"]
    assert len(filter_clause) == 1, "Filter should have one terms query"
    terms_query = filter_clause[0]
    assert "terms" in terms_query, "Filter should use terms query"
    assert "test_item" in terms_query["terms"], "Terms query should filter test_item field"
    query_ids = terms_query["terms"]["test_item"]
    assert len(query_ids) == 2, "Query should search for 2 test items"
    assert 2001 in query_ids, "Query should search for test_item 2001"
    assert 2002 in query_ids, "Query should search for test_item 2002"

    # Verify opensearchpy.helpers.bulk was called to update logs
    mock_bulk.assert_called_once()
    bulk_call = mock_bulk.call_args
    assert bulk_call[0][0] == mocked_opensearch_client, "bulk should use mocked OpenSearch client"

    # Verify bulk update bodies
    bulk_bodies = bulk_call[0][1]
    assert len(bulk_bodies) == 2, "Should update 2 logs"

    # Verify update body structure
    for body in bulk_bodies:
        assert body["_op_type"] == "update", "Body should be an update operation"
        assert "_id" in body, "Body should have _id"
        assert "_index" in body, "Body should have _index"
        assert "doc" in body, "Body should have doc with update fields"
        assert "issue_type" in body["doc"], "Update should include issue_type"
        assert "is_auto_analyzed" in body["doc"], "Update should include is_auto_analyzed"
        assert body["doc"]["is_auto_analyzed"] is False, "is_auto_analyzed should be False"

    # Verify result contains empty list (all items were found and updated)
    assert result == [], "Should return empty list when all items are updated"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_defect_update_with_nonexistent_index(
    mock_scan,
    mock_bulk,
    index_service: IndexService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test defect_update when index does not exist."""
    defect_update_info = test_data["defect_update_info"]
    test_project_id = defect_update_info["project"]
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"

    # Configure mock to raise exception for non-existent index
    mocked_opensearch_client.indices.get.side_effect = Exception("Index not found")

    # Execute the method
    result = index_service.defect_update(defect_update_info)

    # Verify index_exists was called exactly once with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was NOT called since index doesn't exist
    mock_scan.assert_not_called()

    # Verify bulk was NOT called since index doesn't exist
    mock_bulk.assert_not_called()

    # Verify result contains all test item IDs (none were updated)
    expected_ids = [2001, 2002]
    assert sorted(result) == sorted(expected_ids), "Should return all test item IDs when index doesn't exist"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_defect_update_with_partial_match(
    mock_scan,
    mock_bulk,
    index_service: IndexService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test defect_update when only some test items are found."""
    defect_update_info = test_data["defect_update_info"]

    # Configure mock_scan to return only one result
    mock_scan.return_value = iter([test_data["scan_results_for_defect_update"][0]])

    # Configure mock_bulk to return success
    mock_bulk.return_value = (1, [])

    # Execute the method
    result = index_service.defect_update(defect_update_info)

    # Verify scan was called
    mock_scan.assert_called_once()

    # Verify bulk was called with one update
    mock_bulk.assert_called_once()
    bulk_bodies = mock_bulk.call_args[0][1]
    assert len(bulk_bodies) == 1, "Should update only 1 log"

    # Verify result contains test item ID that was not found (2002)
    assert result == [2002], "Should return test_item 2002 that was not found"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_index_logs_creates_index_if_not_exists(
    mock_scan,
    mock_bulk,
    index_service: IndexService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that index_logs creates index if it doesn't exist."""
    launches = [Launch(**launch_data) for launch_data in test_data["launches"]]
    test_project_id = launches[0].project
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"

    # Configure mock to indicate index doesn't exist
    mocked_opensearch_client.indices.get.side_effect = Exception("Index not found")

    # Configure mock_scan and mock_bulk (need 2 scan calls: delete and merge)
    mock_scan.side_effect = [
        iter(test_data["scan_results_for_delete_merged"]),
        iter(test_data["scan_results_for_merge"]),
    ]
    mock_bulk.return_value = (3, [])

    # Execute the method
    result = index_service.index_logs(launches)

    # Verify index_exists was called exactly once with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify create_index was called since index doesn't exist with correct index name
    mocked_opensearch_client.indices.create.assert_called_once()
    create_call = mocked_opensearch_client.indices.create.call_args
    assert create_call[1]["index"] == expected_index_name, "Should create correct index"
    assert "body" in create_call[1], "Should provide index body/mappings"

    # Verify indexing still proceeded (3 bulk calls: index, delete, merge)
    assert mock_bulk.call_count == 3, "Should still index logs after creating index (3 bulk calls)"
    assert result.took > 0, "Should return success result"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_send_stats_info_with_empty_stats(
    mock_bulk,
    index_service: IndexService,
    mocked_opensearch_client: OpenSearch,
) -> None:
    """Test send_stats_info with empty stats."""
    # Execute with empty stats
    index_service.send_stats_info({})

    # Verify bulk operation was not called since there's nothing to index
    mock_bulk.assert_not_called()
