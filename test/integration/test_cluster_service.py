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

from typing import Any
from unittest import mock

import pytest
from opensearchpy import OpenSearch
from opensearchpy.client import IndicesClient

from app.commons.model.launch_objects import Launch, LaunchInfoForClustering, Log, TestItem
from app.commons.os_client import OsClient
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
    """Create ClusterService with real OsClient and mocked OpenSearch client."""
    # Create real OsClient with mocked OpenSearch client
    os_client = OsClient(APP_CONFIG, os_client=mocked_opensearch_client)

    # Create ClusterService with real OsClient and SearchConfig
    service = ClusterService(APP_CONFIG, DEFAULT_SEARCH_CONFIG, os_client=os_client)

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
    mocked_opensearch_client.scroll.return_value = {}

    # Configure mock_bulk to return success
    mock_bulk.return_value = (10, [])  # (success_count, errors)

    # Execute the method
    result = cluster_service.find_clusters(launch_info)

    # Verify es_client.search was called to find similar items
    # In our test data, after grouping by exception type and then clustering by message similarity:
    # - Logs 7001, 7003: NullPointerException with identical message "Cannot invoke method on null object" -> group 1
    # - Log 7002: NullPointerException with different message "Object reference is null" -> group 2
    # - Log 7004: SQLException "Database connection timeout" -> group 3
    # Total: 3 groups, so search should be called exactly 3 times
    actual_search_call_count = mocked_opensearch_client.search.call_count
    assert actual_search_call_count == 3, "search should be called exactly 3 times (once per group)"

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

        # Verify query filters for issue_type existence
        filters = bool_query["filter"]
        issue_type_filter = next((f for f in filters if "exists" in f and f["exists"]["field"] == "issue_type"), None)
        assert issue_type_filter is not None, "query should filter by issue_type existence"

        # Verify query must_not clause excludes current test item
        must_not_clause = bool_query["must_not"]
        test_item_exclude = next((m for m in must_not_clause if "term" in m and "test_item_id" in m["term"]), None)
        assert test_item_exclude is not None, "query should exclude current test_item_id"

        # Verify nested query structure for logs
        must_clause = bool_query["must"]
        nested_query = next((m.get("nested") for m in must_clause if "nested" in m), None)
        assert nested_query is not None, "query should include nested logs query"
        assert nested_query["path"] == "logs", "nested query should target logs path"
        nested_bool = nested_query["query"]["bool"]
        nested_filters = nested_bool["filter"]
        log_level_filter = next((f for f in nested_filters if "range" in f and "logs.log_level" in f["range"]), None)
        assert log_level_filter is not None, "nested query should filter by logs.log_level"
        assert log_level_filter["range"]["logs.log_level"]["gte"] == 40000, "query should filter log_level >= 40000"
        assert any("more_like_this" in m for m in nested_bool["must"]), "nested query should have more_like_this"

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
    # All 4 logs from test data plus 1 similar log from same launch should be updated
    assert len(bulk_bodies) == 5, f"bulk should have exactly 5 update operations, got {len(bulk_bodies)}"

    # Verify bulk bodies structure
    for body in bulk_bodies:
        assert "_op_type" in body, "Each body should have _op_type"
        assert body["_op_type"] == "update", "Operation should be update"
        assert "_id" in body, "Update body should have _id"
        assert body["_index"] == expected_index_name, "Update body should use correct index name"
        assert "script" in body, "Update body should have script"

        # Verify cluster information in script params
        params = body["script"]["params"]
        assert "cluster_id" in params, "Params should have cluster_id"
        assert "cluster_message" in params, "Params should have cluster_message"
        assert "cluster_with_numbers" in params, "Params should have cluster_with_numbers"
        expected_cluster_with_numbers = not launch_info.cleanNumbers
        assert (
            params["cluster_with_numbers"] == expected_cluster_with_numbers
        ), f"cluster_with_numbers should be {expected_cluster_with_numbers}"

        # Verify cluster_id is a non-empty string
        assert isinstance(params["cluster_id"], str), "cluster_id should be string"
        assert params["cluster_id"], "cluster_id should not be empty"

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
    # All 4 logs from our launch plus 1 similar same-launch log should be in the clusters
    assert total_log_ids == 5, "should have exactly 5 log IDs total (including same-launch similar log)"
    # We have 3 unique test items in our test data: 6001, 6002, 6003; plus 1 similar 6004 same-launch test item from
    # search
    assert (
        len(total_item_ids) == 4
    ), f"should have exactly 4 unique item IDs (6001, 6002, 6003, 6004), got {len(total_item_ids)}"


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
    # Configure mock to raise exception for non-existent index
    mocked_opensearch_client.indices.get.side_effect = Exception("Index not found")
    mocked_opensearch_client.search.return_value = test_data["search_results_no_similar_items"]

    # Execute the method
    result = cluster_service.find_clusters(launch_info)

    # Verify bulk was NOT called since index doesn't exist
    mock_bulk.assert_not_called()

    # Verify result contains computed clusters even without index
    assert result is not None, "result should not be None"
    assert result.project == launch_info.project, "result should have correct project"
    assert result.launchId == launch_info.launch.launchId, "result should have correct launchId"
    assert result.clusters, "Should return clusters even when index doesn't exist"


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
    # Configure mocked_opensearch_client.search to return no similar items
    mocked_opensearch_client.search.return_value = test_data["search_results_no_similar_items"]

    # Configure mock_bulk to return success
    mock_bulk.return_value = (4, [])  # (success_count, errors)

    # Execute the method
    result = cluster_service.find_clusters(launch_info)

    # Verify es_client.search was called
    # With the same test data as test_find_clusters_calls_correct_services:
    # 3 groups (2 different NullPointerException message clusters + 1 SQLException), so 3 search calls
    actual_search_call_count = mocked_opensearch_client.search.call_count
    assert actual_search_call_count == 3, "search should be called exactly 3 times (once per group)"

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
        params = body["script"]["params"]
        assert params["cluster_id"], "cluster_id should be generated even without similar items"
        assert params["cluster_message"], "cluster_message should be generated even without similar items"

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
    # Configure mocked_opensearch_client.search to return similar items from same launch
    mocked_opensearch_client.search.return_value = test_data["search_results_for_update"]

    # Configure mock_bulk to return success
    mock_bulk.return_value = (2, [])  # (success_count, errors)

    # Execute the method
    result = cluster_service.find_clusters(launch_info)

    # Verify es_client.search was called
    # With 1 log (test item 6005 with IOException), there is 1 group, so 1 search call
    actual_search_call_count = mocked_opensearch_client.search.call_count
    assert actual_search_call_count == 1, "search should be called exactly 1 time (once per group)"

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
    # Configure mocked_opensearch_client.search to return similar items
    mocked_opensearch_client.search.return_value = test_data["search_results_for_similar_items"]

    # Configure mock_bulk to return success
    mock_bulk.return_value = (2, [])  # (success_count, errors)

    # Execute the method
    result = cluster_service.find_clusters(launch_info)

    # Verify es_client.search was called
    # With cleanNumbers=True, logs become more similar, reducing the number of groups
    actual_search_call_count = mocked_opensearch_client.search.call_count
    assert actual_search_call_count == 2, "search should be called exactly 2 times (once per group)"

    # Verify search query structure for cleanNumbers mode
    search_call = mocked_opensearch_client.search.call_args_list[0]
    search_query = search_call[1]["body"]

    # Extract the bool query from function_score
    bool_query = search_query["query"]["function_score"]["query"]["bool"]

    # Verify query filters for cluster_with_numbers = False (when cleanNumbers=True)
    nested_query = next((m.get("nested") for m in bool_query["must"] if "nested" in m), None)
    nested_filters = nested_query["query"]["bool"]["filter"]
    cluster_with_numbers_filter = next(
        (f for f in nested_filters if "term" in f and "logs.cluster_with_numbers" in f["term"]), None
    )
    assert cluster_with_numbers_filter is not None, "query should filter by cluster_with_numbers"
    assert (
        cluster_with_numbers_filter["term"]["logs.cluster_with_numbers"] is False
    ), "cleanNumbers=True should search for cluster_with_numbers=False"

    # Verify bulk was called
    mock_bulk.assert_called_once()

    # Verify bulk bodies have cluster_with_numbers set correctly
    bulk_call = mock_bulk.call_args
    bulk_bodies = bulk_call[0][1]
    for body in bulk_bodies:
        params = body["script"]["params"]
        assert "cluster_with_numbers" in params, "Params should have cluster_with_numbers"
        # cluster_with_numbers = not cleanNumbers, so when cleanNumbers=True, cluster_with_numbers=False
        expected_cluster_with_numbers = not launch_info.cleanNumbers
        assert (
            params["cluster_with_numbers"] == expected_cluster_with_numbers
        ), f"cluster_with_numbers should be {expected_cluster_with_numbers}"

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


PROJECT_ID = 555
LAUNCH_ID = 5001
ERROR_LEVEL = 40000
FATAL_LEVEL = 50000
BIG_LOG_TEXT = (
    "big error log Error The Group Services daemon is unable to run the deactivation script Standard fields indicate "
    "that the Group Services daemon is unable to run the script Detailed fields give more information The "
    "deactivation script might not exist or system resources are not sufficient to run the deactivation script"
)
NO_HITS = {"hits": {"total": {"value": 0, "relation": "eq"}, "max_score": None, "hits": []}}


def build_test_item(test_item_id: int, logs: list[tuple[int, int, str]]) -> TestItem:
    return TestItem(
        testItemId=test_item_id,
        isAutoAnalyzed=False,
        uniqueId=f"unique-{test_item_id}",
        testItemName=f"Test Item {test_item_id}",
        logs=[Log(logId=log_id, logLevel=log_level, message=message) for log_id, log_level, message in logs],
    )


def build_launch_info(*test_items: TestItem) -> LaunchInfoForClustering:
    return LaunchInfoForClustering(
        launch=Launch(launchId=LAUNCH_ID, project=PROJECT_ID, launchName="Launch", testItems=list(test_items)),
        project=PROJECT_ID,
        numberOfLogLines=-1,
    )


def get_nested_query(search_call: Any) -> dict[str, Any]:
    must_clause = search_call[1]["body"]["query"]["function_score"]["query"]["bool"]["must"]
    return next(clause["nested"] for clause in must_clause if "nested" in clause)


def get_bulk_bodies(mock_bulk: mock.Mock) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split bulk bodies into nested log updates and merged small log updates."""
    mock_bulk.assert_called_once()
    bodies = mock_bulk.call_args[0][1]
    merged_bodies = [body for body in bodies if "merged_small_logs" in body["script"]["params"]]
    log_bodies = [body for body in bodies if "log_id" in body["script"]["params"]]
    return log_bodies, merged_bodies


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_merges_small_logs_of_one_level(
    mock_bulk, cluster_service: ClusterService, mocked_opensearch_client: OpenSearch
) -> None:
    """EPMRPP-122245: small logs of one level of a Test Item are united into one cluster."""
    launch_info = build_launch_info(
        build_test_item(
            6001,
            [
                (7001, ERROR_LEVEL, "First small error log"),
                (7002, ERROR_LEVEL, "Second small error log"),
                (7003, ERROR_LEVEL, "Third small error log"),
            ],
        )
    )
    mocked_opensearch_client.search.return_value = NO_HITS
    mock_bulk.return_value = (1, [])

    result = cluster_service.find_clusters(launch_info)

    assert len(result.clusters) == 1, "small logs of one level should give exactly 1 cluster"
    cluster = result.clusters[0]
    assert cluster.logIds == [7001, 7002, 7003]
    assert cluster.itemIds == [6001]
    assert cluster.clusterMessage == "First small error log\nSecond small error log\nThird small error log"

    mocked_opensearch_client.search.assert_called_once()
    nested_query = get_nested_query(mocked_opensearch_client.search.call_args)
    assert nested_query["path"] == "merged_small_logs", "merged log should be searched among merged logs"
    assert all(field.startswith("merged_small_logs.") for field in nested_query["inner_hits"]["_source"])

    mock_bulk.assert_called_once()
    log_bodies, merged_bodies = get_bulk_bodies(mock_bulk)
    assert [body["script"]["params"]["log_id"] for body in log_bodies] == ["7001", "7002", "7003"]
    assert all(body["script"]["params"]["cluster_id"] == str(cluster.clusterId) for body in log_bodies)
    assert len(merged_bodies) == 1
    assert merged_bodies[0]["_id"] == "6001"
    merged_logs = merged_bodies[0]["script"]["params"]["merged_small_logs"]
    assert len(merged_logs) == 1
    assert merged_logs[0]["log_ids"] == ["7001", "7002", "7003"]
    assert merged_logs[0]["log_level"] == ERROR_LEVEL
    assert merged_logs[0]["cluster_id"] == str(cluster.clusterId)
    assert merged_logs[0]["cluster_message"] == cluster.clusterMessage
    assert merged_logs[0]["cluster_with_numbers"] is True


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_creates_cluster_per_level_for_small_logs(
    mock_bulk, cluster_service: ClusterService, mocked_opensearch_client: OpenSearch
) -> None:
    """EPMRPP-74489: small logs of different levels in one Test Item give a cluster per level."""
    launch_info = build_launch_info(
        build_test_item(6001, [(7001, ERROR_LEVEL, "Small error log"), (7002, FATAL_LEVEL, "Small fatal log")])
    )
    mocked_opensearch_client.search.return_value = NO_HITS
    mock_bulk.return_value = (1, [])

    result = cluster_service.find_clusters(launch_info)

    assert len(result.clusters) == 2, "small logs of two levels should give exactly 2 clusters"
    assert sorted(cluster.logIds for cluster in result.clusters) == [[7001], [7002]]
    assert mocked_opensearch_client.search.call_count == 2
    _, merged_bodies = get_bulk_bodies(mock_bulk)
    assert len(merged_bodies) == 1
    merged_logs = merged_bodies[0]["script"]["params"]["merged_small_logs"]
    assert [merged_log["log_level"] for merged_log in merged_logs] == [ERROR_LEVEL, FATAL_LEVEL]


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_drops_small_logs_of_level_with_big_log(
    mock_bulk, cluster_service: ClusterService, mocked_opensearch_client: OpenSearch
) -> None:
    """EPMRPP-74485: a big log has priority over small logs of the same level."""
    launch_info = build_launch_info(
        build_test_item(6001, [(7001, ERROR_LEVEL, "First small error log"), (7002, ERROR_LEVEL, BIG_LOG_TEXT)])
    )
    mocked_opensearch_client.search.return_value = NO_HITS
    mock_bulk.return_value = (1, [])

    result = cluster_service.find_clusters(launch_info)

    assert len(result.clusters) == 1, "only the big log should give a cluster"
    assert result.clusters[0].logIds == [7002]
    assert result.clusters[0].clusterMessage.startswith("big error log")
    nested_query = get_nested_query(mocked_opensearch_client.search.call_args)
    assert nested_query["path"] == "logs"
    mock_bulk.assert_called_once()
    log_bodies, merged_bodies = get_bulk_bodies(mock_bulk)
    assert [body["script"]["params"]["log_id"] for body in log_bodies] == ["7002"]
    assert merged_bodies == []


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_treats_similar_big_logs_of_one_item_as_one_log(
    mock_bulk, cluster_service: ClusterService, mocked_opensearch_client: OpenSearch
) -> None:
    """Big logs of one Test Item at or above the similarity threshold are deduplicated into the last one."""
    launch_info = build_launch_info(
        build_test_item(
            6001, [(7001, ERROR_LEVEL, f"First {BIG_LOG_TEXT}"), (7002, ERROR_LEVEL, f"Second {BIG_LOG_TEXT}")]
        )
    )
    mocked_opensearch_client.search.return_value = NO_HITS
    mock_bulk.return_value = (1, [])

    result = cluster_service.find_clusters(launch_info)

    assert len(result.clusters) == 1, "similar big logs should give exactly 1 cluster"
    assert result.clusters[0].logIds == [7002]
    assert result.clusters[0].clusterMessage.startswith("Second big error log")


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_keeps_different_big_logs_of_one_item_apart(
    mock_bulk, cluster_service: ClusterService, mocked_opensearch_client: OpenSearch
) -> None:
    """Big logs of one Test Item below the similarity threshold give separate clusters."""
    other_big_log_text = (
        "Connection to the database server was refused because the maximum number of client connections was reached "
        "and the connection pool could not allocate a new connection for the request within the configured timeout"
    )
    launch_info = build_launch_info(
        build_test_item(6001, [(7001, ERROR_LEVEL, BIG_LOG_TEXT), (7002, ERROR_LEVEL, other_big_log_text)])
    )
    mocked_opensearch_client.search.return_value = NO_HITS
    mock_bulk.return_value = (1, [])

    result = cluster_service.find_clusters(launch_info)

    assert len(result.clusters) == 2, "different big logs should give exactly 2 clusters"
    assert sorted(cluster.logIds for cluster in result.clusters) == [[7001], [7002]]


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_groups_same_logs_of_different_items(
    mock_bulk, cluster_service: ClusterService, mocked_opensearch_client: OpenSearch
) -> None:
    """The same big and small logs in different Test Items give one cluster with all these Test Items."""
    launch_info = build_launch_info(
        build_test_item(6001, [(7001, ERROR_LEVEL, BIG_LOG_TEXT)]),
        build_test_item(6002, [(7002, ERROR_LEVEL, BIG_LOG_TEXT)]),
        build_test_item(6003, [(7003, ERROR_LEVEL, "Small error log 1"), (7004, ERROR_LEVEL, "Small error log 2")]),
        build_test_item(6004, [(7005, ERROR_LEVEL, "Small error log 1"), (7006, ERROR_LEVEL, "Small error log 2")]),
    )
    mocked_opensearch_client.search.return_value = NO_HITS
    mock_bulk.return_value = (1, [])

    result = cluster_service.find_clusters(launch_info)

    assert len(result.clusters) == 2, "same logs should give exactly 1 cluster for big and 1 for merged small logs"
    clusters = sorted(result.clusters, key=lambda cluster: cluster.logIds)
    assert clusters[0].logIds == [7001, 7002]
    assert sorted(clusters[0].itemIds) == [6001, 6002]
    assert clusters[1].logIds == [7003, 7004, 7005, 7006]
    assert sorted(clusters[1].itemIds) == [6003, 6004]


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
def test_find_clusters_reuses_cluster_of_similar_merged_logs(
    mock_bulk, cluster_service: ClusterService, mocked_opensearch_client: OpenSearch
) -> None:
    """A merged entry found among merged entries of previous launches gives its cluster ID and message."""
    launch_info = build_launch_info(
        build_test_item(
            6001, [(7001, ERROR_LEVEL, "First small error log"), (7002, ERROR_LEVEL, "Second small error log")]
        )
    )
    previous_cluster_message = "First small error log\nSecond small error log"
    mocked_opensearch_client.search.return_value = {
        "_scroll_id": "test_scroll_id",
        "hits": {
            "total": {"value": 1, "relation": "eq"},
            "max_score": 9.5,
            "hits": [
                {
                    "_id": "6100",
                    "_index": f"{APP_CONFIG.esProjectIndexPrefix}{PROJECT_ID}",
                    "_score": 9.5,
                    "_source": {"launch_id": "5000", "test_item_id": "6100", "launch_name": "Launch"},
                    "inner_hits": {
                        "merged_small_logs": {
                            "hits": {
                                "hits": [
                                    {
                                        "_source": {
                                            "log_ids": ["8001", "8002"],
                                            "log_level": ERROR_LEVEL,
                                            "whole_message": "First small error log\nSecond small error log",
                                            "found_exceptions": "",
                                            "potential_status_codes": "",
                                            "cluster_id": "12345678901",
                                            "cluster_message": previous_cluster_message,
                                            "cluster_with_numbers": True,
                                        }
                                    }
                                ]
                            }
                        }
                    },
                }
            ],
        },
    }
    mocked_opensearch_client.scroll.return_value = NO_HITS
    mock_bulk.return_value = (1, [])

    result = cluster_service.find_clusters(launch_info)

    assert len(result.clusters) == 1
    cluster = result.clusters[0]
    assert cluster.clusterId == 12345678901
    assert cluster.clusterMessage == previous_cluster_message
    assert cluster.logIds == [7001, 7002], "logs of previous launches should not be returned"
    _, merged_bodies = get_bulk_bodies(mock_bulk)
    assert len(merged_bodies) == 1
    assert merged_bodies[0]["script"]["params"]["merged_small_logs"][0]["cluster_id"] == "12345678901"
