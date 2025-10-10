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
from app.commons.model.launch_objects import LaunchInfoForClustering
from app.service.cluster_service import ClusterService
from app.utils.utils import read_json_file
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG


@pytest.fixture
def test_data() -> dict:
    """Load test data with launch info and scan results."""
    return read_json_file("test_res", "cluster_service_test_data.json", to_json=True)


@pytest.fixture
def mocked_opensearch_client() -> OpenSearch:
    """Create a mocked OpenSearch client instance."""
    mock_client = mock.Mock(OpenSearch)
    mock_client.indices = mock.Mock(IndicesClient)

    # Mock indices.get for index_exists checks
    mock_client.indices.get.return_value = {"rp_555": "exists"}

    return mock_client


@pytest.fixture
def cluster_service(mocked_opensearch_client: OpenSearch) -> ClusterService:
    """Create ClusterService with real EsClient and mocked OpenSearch client."""
    # Create real EsClient with mocked OpenSearch client
    es_client = EsClient(APP_CONFIG, es_client=mocked_opensearch_client)

    # Create ClusterService with real EsClient and SearchConfig
    service = ClusterService(APP_CONFIG, DEFAULT_SEARCH_CONFIG, es_client=es_client)

    return service


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_calls_correct_services(
    mock_bulk,
    cluster_service: ClusterService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that find_clusters method calls internal services with correct arguments."""
    # Prepare test data
    launch_info = LaunchInfoForClustering(**test_data["launch_info_for_clustering"])
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{launch_info.project}"

    # Configure mocked_opensearch_client.search to return similar items
    # The search is called during _find_similar_items_from_es for each group
    mocked_opensearch_client.search.return_value = test_data["search_results_for_similar_items"]

    # Configure mock_bulk to return success
    mock_bulk.return_value = (10, [])  # (success_count, errors)

    # Execute the method
    result = cluster_service.find_clusters(launch_info)

    # Verify index_exists was called exactly once with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify es_client.search was called to find similar items
    # In our test data, after grouping by exception type and then clustering by message similarity:
    # - Logs 7001, 7003: NullPointerException with identical message "Cannot invoke method on null object" -> group 1
    # - Log 7002: NullPointerException with different message "Object reference is null" -> group 2
    # - Log 7004: SQLException "Database connection timeout" -> group 3
    # Total: 3 groups, so search should be called exactly 3 times
    assert (
        mocked_opensearch_client.search.call_count == 3
    ), f"search should be called exactly 3 times (once per group), but was called {mocked_opensearch_client.search.call_count} times"

    # Verify search call structure
    search_calls = mocked_opensearch_client.search.call_args_list
    for search_call in search_calls:
        # Verify search is called with correct index
        assert search_call[1]["index"] == expected_index_name, f"search should use index '{expected_index_name}'"

        # Verify search query structure
        search_query = search_call[1]["body"]
        assert "query" in search_query, "search query should have 'query' key"
        assert "function_score" in search_query["query"], "search query should have function_score for time decay"

        # Verify function_score wraps the actual query
        function_score = search_query["query"]["function_score"]
        assert "query" in function_score, "function_score should have query"
        assert "bool" in function_score["query"], "function_score query should have bool query"

        # Verify bool query structure
        bool_query = function_score["query"]["bool"]
        assert "filter" in bool_query, "search query should have filter clause"
        assert "must" in bool_query, "search query should have must clause"
        assert "must_not" in bool_query, "search query should have must_not clause"

        # Verify query filters for log_level >= 40000 (ERROR_LOGGING_LEVEL)
        filters = bool_query["filter"]
        log_level_filter = next((f for f in filters if "range" in f and "log_level" in f["range"]), None)
        assert log_level_filter is not None, "query should filter by log_level"
        assert log_level_filter["range"]["log_level"]["gte"] == 40000, "query should filter log_level >= 40000"

        # Verify query filters for issue_type existence
        issue_type_filter = next((f for f in filters if "exists" in f and f["exists"]["field"] == "issue_type"), None)
        assert issue_type_filter is not None, "query should filter by issue_type existence"

        # Verify query must_not clause excludes current test item
        must_not_clause = bool_query["must_not"]
        test_item_exclude = next((m for m in must_not_clause if "term" in m and "test_item" in m["term"]), None)
        assert test_item_exclude is not None, "query should exclude current test_item"

        # Verify query must clause contains more_like_this for message
        must_clause = bool_query["must"]
        mlt_query = next((m for m in must_clause if "more_like_this" in m), None)
        assert mlt_query is not None, "query should have more_like_this clause"

        # Verify time decay functions
        functions = function_score["functions"]
        assert len(functions) == 2, "function_score should have 2 functions (time decay and constant)"
        exp_function = next((f for f in functions if "exp" in f), None)
        assert exp_function is not None, "function_score should have exponential decay function"
        assert "start_time" in exp_function["exp"], "exponential decay should be on start_time field"

    # Verify opensearchpy.helpers.bulk was called exactly once to update cluster information
    mock_bulk.assert_called_once()

    # Verify bulk call structure
    bulk_call = mock_bulk.call_args
    assert bulk_call[0][0] == mocked_opensearch_client, "bulk should use mocked OpenSearch client"

    # Verify bulk bodies contain cluster updates
    bulk_bodies = bulk_call[0][1]
    # All 4 logs from test data should be updated with cluster information
    assert len(bulk_bodies) == 4, f"bulk should have exactly 4 update operations (one per log), got {len(bulk_bodies)}"

    # Verify bulk bodies structure
    for body in bulk_bodies:
        assert "_op_type" in body, "Each body should have _op_type"
        assert body["_op_type"] == "update", "Operation should be update"
        assert "_id" in body, "Update body should have _id"
        assert "_index" in body, "Update body should have _index"
        assert body["_index"] == expected_index_name, "Update body should use correct index name"
        assert "doc" in body, "Update body should have doc field"

        # Verify cluster information in doc
        doc = body["doc"]
        assert "cluster_id" in doc, "Doc should have cluster_id"
        assert "cluster_message" in doc, "Doc should have cluster_message"
        assert "cluster_with_numbers" in doc, "Doc should have cluster_with_numbers"
        expected_cluster_with_numbers = not launch_info.cleanNumbers
        assert (
            doc["cluster_with_numbers"] == expected_cluster_with_numbers
        ), f"cluster_with_numbers should be {expected_cluster_with_numbers}"

        # Verify cluster_id is a non-empty string
        assert isinstance(doc["cluster_id"], str), "cluster_id should be string"
        assert doc["cluster_id"], "cluster_id should not be empty"

    # Verify result structure
    assert result is not None, "result should not be None"
    assert result.project == launch_info.project, "result should have correct project"
    assert result.launchId == launch_info.launch.launchId, "result should have correct launchId"
    assert isinstance(result.clusters, list), "result.clusters should be a list"
    # We have 3 groups after clustering by message similarity (2 NullPointerException groups + 1 SQLException group)
    assert len(result.clusters) == 3, f"result should have exactly 3 clusters, got {len(result.clusters)}"

    # Verify cluster structure
    total_log_ids = 0
    total_item_ids = set()
    for cluster in result.clusters:
        assert hasattr(cluster, "clusterId"), "cluster should have clusterId"
        assert hasattr(cluster, "clusterMessage"), "cluster should have clusterMessage"
        assert hasattr(cluster, "logIds"), "cluster should have logIds"
        assert hasattr(cluster, "itemIds"), "cluster should have itemIds"
        assert cluster.clusterId > 0, "clusterId should be positive"
        assert isinstance(cluster.clusterMessage, str), "clusterMessage should be string"
        assert len(cluster.logIds) >= 1, "cluster should have at least one log ID"
        assert len(cluster.itemIds) >= 1, "cluster should have at least one item ID"
        total_log_ids += len(cluster.logIds)
        total_item_ids.update(cluster.itemIds)

    # Verify total counts match our test data
    # All 4 logs from our launch should be in the clusters (ES similar items help identify clusters but don't add their log IDs)
    assert total_log_ids == 4, f"should have exactly 4 log IDs total (from our test data), got {total_log_ids}"
    # We have 3 unique test items in our test data: 6001, 6002, 6003
    assert (
        len(total_item_ids) == 3
    ), f"should have exactly 3 unique item IDs (6001, 6002, 6003), got {len(total_item_ids)}"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_with_nonexistent_index(
    mock_bulk,
    cluster_service: ClusterService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test find_clusters when index does not exist."""
    launch_info = LaunchInfoForClustering(**test_data["launch_info_for_clustering"])
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{launch_info.project}"

    # Configure mock to raise exception for non-existent index
    mocked_opensearch_client.indices.get.side_effect = Exception("Index not found")

    # Execute the method
    result = cluster_service.find_clusters(launch_info)

    # Verify index_exists was called with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify bulk was NOT called since index doesn't exist
    mock_bulk.assert_not_called()

    # Verify result is empty cluster result
    assert result is not None, "result should not be None"
    assert result.project == launch_info.project, "result should have correct project"
    assert result.launchId == launch_info.launch.launchId, "result should have correct launchId"
    assert result.clusters == [], "Should return empty clusters list when index doesn't exist"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_with_no_similar_items(
    mock_bulk,
    cluster_service: ClusterService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test find_clusters when no similar items are found in ES."""
    # Prepare test data
    launch_info = LaunchInfoForClustering(**test_data["launch_info_for_clustering"])
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{launch_info.project}"

    # Configure mocked_opensearch_client.search to return no similar items
    mocked_opensearch_client.search.return_value = test_data["search_results_no_similar_items"]

    # Configure mock_bulk to return success
    mock_bulk.return_value = (4, [])  # (success_count, errors)

    # Execute the method
    result = cluster_service.find_clusters(launch_info)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify es_client.search was called
    # With the same test data as test_find_clusters_calls_correct_services:
    # 3 groups (2 different NullPointerException message clusters + 1 SQLException), so 3 search calls
    assert (
        mocked_opensearch_client.search.call_count == 3
    ), f"search should be called exactly 3 times (once per group), but was called {mocked_opensearch_client.search.call_count} times"

    # Verify opensearchpy.helpers.bulk was called to update cluster information
    # Even without similar items, new clusters should be created based on hash calculation
    mock_bulk.assert_called_once()

    # Verify bulk bodies contain cluster updates with new cluster IDs (hash-based)
    bulk_call = mock_bulk.call_args
    bulk_bodies = bulk_call[0][1]
    # All 4 logs should be updated even without similar items
    assert len(bulk_bodies) == 4, f"bulk should have exactly 4 update operations, got {len(bulk_bodies)}"

    # Verify cluster IDs are generated (not from ES)
    for body in bulk_bodies:
        doc = body["doc"]
        assert doc["cluster_id"], "cluster_id should be generated even without similar items"
        assert doc["cluster_message"], "cluster_message should be generated even without similar items"

    # Verify result structure
    assert result is not None, "result should not be None"
    assert isinstance(result.clusters, list), "result.clusters should be a list"
    # Even without similar items, we should have 3 clusters (based on 3 groups from clustering)
    assert (
        len(result.clusters) == 3
    ), f"result should have exactly 3 clusters even without similar items, got {len(result.clusters)}"

    # Verify clusters are created with hash-based IDs
    for cluster in result.clusters:
        assert cluster.clusterId > 0, "clusterId should be positive"
        assert cluster.clusterMessage, "clusterMessage should not be empty"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_with_for_update_mode(
    mock_bulk,
    cluster_service: ClusterService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test find_clusters with forUpdate=True mode (allows same launch results)."""
    # Prepare test data with forUpdate=True
    launch_info = LaunchInfoForClustering(**test_data["launch_info_for_update"])
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{launch_info.project}"

    # Configure mocked_opensearch_client.search to return similar items from same launch
    mocked_opensearch_client.search.return_value = test_data["search_results_for_update"]

    # Configure mock_bulk to return success
    mock_bulk.return_value = (2, [])  # (success_count, errors)

    # Execute the method
    result = cluster_service.find_clusters(launch_info)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify es_client.search was called
    # With 1 log (test item 6005 with IOException), there is 1 group, so 1 search call
    assert (
        mocked_opensearch_client.search.call_count == 1
    ), f"search should be called exactly 1 time (once per group), but was called {mocked_opensearch_client.search.call_count} times"

    # Verify search query structure for forUpdate mode
    search_call = mocked_opensearch_client.search.call_args_list[0]
    search_query = search_call[1]["body"]

    # Extract the bool query from function_score
    bool_query = search_query["query"]["function_score"]["query"]["bool"]

    # Verify that with forUpdate=True, the query should have "should" clause for same launch_id
    # instead of "must_not" clause excluding same launch_id
    should_clause = bool_query.get("should", [])
    launch_id_should = next((s for s in should_clause if "term" in s and "launch_id" in s["term"]), None)
    assert launch_id_should is not None, "forUpdate mode should include same launch_id in 'should' clause"

    # Verify must_not does NOT exclude the same launch (unlike non-update mode)
    must_not_clause = bool_query.get("must_not", [])
    launch_id_must_not = next((m for m in must_not_clause if "term" in m and "launch_id" in m["term"]), None)
    assert launch_id_must_not is None, "forUpdate mode should NOT exclude same launch_id in 'must_not' clause"

    # Verify bulk was called
    mock_bulk.assert_called_once()

    # Verify result structure
    assert result is not None, "result should not be None"
    assert isinstance(result.clusters, list), "result.clusters should be a list"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_with_clean_numbers_mode(
    mock_bulk,
    cluster_service: ClusterService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test find_clusters with cleanNumbers=True mode."""
    # Prepare test data with cleanNumbers=True
    launch_info = LaunchInfoForClustering(**test_data["launch_info_for_clustering"])
    launch_info.cleanNumbers = True
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{launch_info.project}"

    # Configure mocked_opensearch_client.search to return similar items
    mocked_opensearch_client.search.return_value = test_data["search_results_for_similar_items"]

    # Configure mock_bulk to return success
    mock_bulk.return_value = (2, [])  # (success_count, errors)

    # Execute the method
    result = cluster_service.find_clusters(launch_info)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify es_client.search was called
    # With cleanNumbers=True, using full test data creates 3 groups (same as without cleanNumbers mode)
    assert (
        mocked_opensearch_client.search.call_count == 3
    ), f"search should be called exactly 3 times (once per group), but was called {mocked_opensearch_client.search.call_count} times"

    # Verify search query structure for cleanNumbers mode
    search_call = mocked_opensearch_client.search.call_args_list[0]
    search_query = search_call[1]["body"]

    # Extract the bool query from function_score
    bool_query = search_query["query"]["function_score"]["query"]["bool"]

    # Verify query filters for cluster_with_numbers = False (when cleanNumbers=True)
    filters = bool_query["filter"]
    cluster_with_numbers_filter = next(
        (f for f in filters if "term" in f and "cluster_with_numbers" in f["term"]), None
    )
    assert cluster_with_numbers_filter is not None, "query should filter by cluster_with_numbers"
    assert (
        cluster_with_numbers_filter["term"]["cluster_with_numbers"] is False
    ), "cleanNumbers=True should search for cluster_with_numbers=False"

    # Verify bulk was called
    mock_bulk.assert_called_once()

    # Verify bulk bodies have cluster_with_numbers set correctly
    bulk_call = mock_bulk.call_args
    bulk_bodies = bulk_call[0][1]
    for body in bulk_bodies:
        doc = body["doc"]
        assert "cluster_with_numbers" in doc, "Doc should have cluster_with_numbers"
        # cluster_with_numbers = not cleanNumbers, so when cleanNumbers=True, cluster_with_numbers=False
        expected_cluster_with_numbers = not launch_info.cleanNumbers
        assert (
            doc["cluster_with_numbers"] == expected_cluster_with_numbers
        ), f"cluster_with_numbers should be {expected_cluster_with_numbers} when cleanNumbers={launch_info.cleanNumbers}"

    # Verify result structure
    assert result is not None, "result should not be None"
    assert isinstance(result.clusters, list), "result.clusters should be a list"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_verifies_bulk_chunk_size(
    mock_bulk,
    cluster_service: ClusterService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that find_clusters uses correct chunk_size for bulk operations."""
    # Prepare test data
    launch_info = LaunchInfoForClustering(**test_data["launch_info_for_clustering"])

    # Configure mocked_opensearch_client.search to return similar items
    mocked_opensearch_client.search.return_value = test_data["search_results_for_similar_items"]

    # Configure mock_bulk to return success
    mock_bulk.return_value = (1, [])  # (success_count, errors)

    # Execute the method
    cluster_service.find_clusters(launch_info)

    # Verify bulk was called with correct chunk_size parameter
    mock_bulk.assert_called_once()
    bulk_call = mock_bulk.call_args

    # Verify chunk_size is passed correctly (from app_config.esChunkNumberUpdateClusters)
    assert "chunk_size" in bulk_call[1], "bulk should be called with chunk_size parameter"
    assert (
        bulk_call[1]["chunk_size"] == APP_CONFIG.esChunkNumberUpdateClusters
    ), f"chunk_size should be {APP_CONFIG.esChunkNumberUpdateClusters}"

    # Verify refresh parameter
    assert "refresh" in bulk_call[1], "bulk should be called with refresh parameter"
    assert bulk_call[1]["refresh"] is False, "refresh should be False for cluster updates (performance)"
