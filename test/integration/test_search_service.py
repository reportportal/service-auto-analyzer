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
from app.commons.model.launch_objects import SearchLogs
from app.service.search_service import SearchService
from app.utils.utils import read_json_file
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG


@pytest.fixture
def test_data() -> dict:
    """Load test data with search requests and scan results."""
    return read_json_file("test_res", "search_service_test_data.json", to_json=True)


@pytest.fixture
def mocked_opensearch_client() -> OpenSearch:
    """Create a mocked OpenSearch client instance."""
    mock_client = mock.Mock(OpenSearch)
    mock_client.indices = mock.Mock(IndicesClient)

    # Mock indices.get for index_exists checks
    mock_client.indices.get.return_value = {"rp_123": "exists"}

    return mock_client


@pytest.fixture
def search_service(mocked_opensearch_client: OpenSearch) -> SearchService:
    """Create SearchService with real EsClient and mocked OpenSearch client."""
    # Create real EsClient with mocked OpenSearch client
    es_client = EsClient(APP_CONFIG, es_client=mocked_opensearch_client)

    # Create SearchService with real EsClient and SearchConfig
    service = SearchService(APP_CONFIG, DEFAULT_SEARCH_CONFIG, es_client=es_client)

    return service


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_search_logs_calls_correct_services(
    mock_scan,
    search_service: SearchService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that search_logs method calls internal services with correct arguments."""
    # Prepare test data
    search_request = SearchLogs(**test_data["search_request"])
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{search_request.projectId}"

    # Configure mock_scan to return different results for different calls:
    # 1st call: search for first message
    # 2nd call: search for second message
    mock_scan.side_effect = [
        iter(test_data["search_results_for_first_message"]),
        iter(test_data["search_results_for_second_message"]),
    ]

    # Execute the method
    result = search_service.search_logs(search_request)

    # Verify index_exists was called exactly once with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify opensearchpy.helpers.scan was called twice (once per log message)
    assert mock_scan.call_count == 2, "scan should be called twice: once for each log message"

    # Verify first scan call for first message
    first_scan_call = mock_scan.call_args_list[0]
    assert first_scan_call[0][0] == mocked_opensearch_client, "first scan should use mocked OpenSearch client"
    assert first_scan_call[1]["index"] == expected_index_name, f"first scan should use index '{expected_index_name}'"

    # Verify first scan query structure
    first_scan_query = first_scan_call[1]["query"]
    assert "query" in first_scan_query, "first scan query should have 'query' key"
    assert "bool" in first_scan_query["query"], "first scan query should have bool query"
    assert "filter" in first_scan_query["query"]["bool"], "first scan query should have filter clause"
    assert "must" in first_scan_query["query"]["bool"], "first scan query should have must clause"
    assert "must_not" in first_scan_query["query"]["bool"], "first scan query should have must_not clause"

    # Verify query filters for log_level >= 40000 (ERROR_LOGGING_LEVEL)
    filters = first_scan_query["query"]["bool"]["filter"]
    log_level_filter = next((f for f in filters if "range" in f and "log_level" in f["range"]), None)
    assert log_level_filter is not None, "query should filter by log_level"
    assert log_level_filter["range"]["log_level"]["gte"] == 40000, "query should filter log_level >= 40000"

    # Verify query filters for issue_type existence
    issue_type_filter = next((f for f in filters if "exists" in f and f["exists"]["field"] == "issue_type"), None)
    assert issue_type_filter is not None, "query should filter by issue_type existence"

    # Verify query filters for is_merged = False (since message is not empty)
    is_merged_filter = next((f for f in filters if "term" in f and "is_merged" in f["term"]), None)
    assert is_merged_filter is not None, "query should filter by is_merged"
    assert is_merged_filter["term"]["is_merged"] is False, "query should filter is_merged = False"

    # Verify query must clause contains more_like_this for message
    must_clause = first_scan_query["query"]["bool"]["must"]
    mlt_query = next((m for m in must_clause if "more_like_this" in m), None)
    assert mlt_query is not None, "query should have more_like_this clause"
    assert "message" in mlt_query["more_like_this"]["fields"], "more_like_this should search message field"

    # Verify query must clause contains launch_id filter
    launch_id_query = next((m for m in must_clause if "terms" in m and "launch_id" in m["terms"]), None)
    assert launch_id_query is not None, "query should filter by launch_id"
    assert launch_id_query["terms"]["launch_id"] == search_request.filteredLaunchIds

    # Verify query must_not clause excludes current test item
    must_not_clause = first_scan_query["query"]["bool"]["must_not"]
    test_item_exclude = next((m for m in must_not_clause if "term" in m and "test_item" in m["term"]), None)
    assert test_item_exclude is not None, "query should exclude current test_item"
    assert test_item_exclude["term"]["test_item"]["value"] == search_request.itemId

    # Verify second scan call for second message
    second_scan_call = mock_scan.call_args_list[1]
    assert second_scan_call[0][0] == mocked_opensearch_client, "second scan should use mocked OpenSearch client"
    assert second_scan_call[1]["index"] == expected_index_name, f"second scan should use index '{expected_index_name}'"

    # Verify second scan query has similar structure
    second_scan_query = second_scan_call[1]["query"]
    assert "query" in second_scan_query, "second scan query should have 'query' key"
    assert "bool" in second_scan_query["query"], "second scan query should have bool query"

    # Verify result structure
    assert result is not None, "result should not be None"
    assert isinstance(result, list), "result should be a list"
    assert (
        len(result) == 1
    ), "result should have exactly 1 item (only highest scoring result from similarity calculation)"

    # Verify result items have correct structure
    for item in result:
        assert hasattr(item, "logId"), "result item should have logId"
        assert hasattr(item, "testItemId"), "result item should have testItemId"
        assert hasattr(item, "matchScore"), "result item should have matchScore"
        assert item.matchScore >= 0, "matchScore should be non-negative"
        assert item.matchScore <= 100, "matchScore should be <= 100"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_search_logs_with_nonexistent_index(
    mock_scan,
    search_service: SearchService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test search_logs when index does not exist."""
    search_request = SearchLogs(**test_data["search_request"])
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{search_request.projectId}"

    # Configure mock to raise exception for non-existent index
    mocked_opensearch_client.indices.get.side_effect = Exception("Index not found")

    # Execute the method
    result = search_service.search_logs(search_request)

    # Verify index_exists was called with correct index name
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was NOT called since index doesn't exist
    mock_scan.assert_not_called()

    # Verify result is empty list
    assert result == [], "Should return empty list when index doesn't exist"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_search_logs_with_all_messages_should_match(
    mock_scan,
    search_service: SearchService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test search_logs with allMessagesShouldMatch enabled filters test items that don't match all messages."""
    # Prepare test data - use the main search_request which has 2 distinct messages
    search_request = SearchLogs(**test_data["search_request"])
    # Enable allMessagesShouldMatch
    search_request.analyzerConfig.allMessagesShouldMatch = True
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{search_request.projectId}"

    # Configure mock_scan to return results where:
    # - test_item 3001 appears in both result sets → should be in final results
    # - test_item 3002 appears only in first result set → should NOT be in final results
    # - test_item 3003 appears only in second result set → should NOT be in final results
    mock_scan.side_effect = [
        iter(test_data["search_results_for_all_messages_match_first"]),  # Returns 3001, 3002
        iter(test_data["search_results_for_all_messages_match_second"]),  # Returns 3001, 3003
    ]

    # Execute the method
    result = search_service.search_logs(search_request)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was called exactly twice (once for each message)
    assert mock_scan.call_count == 2, "scan should be called twice: once for each log message"

    # Verify scan query structure for both calls
    first_scan_call = mock_scan.call_args_list[0]
    first_scan_query = first_scan_call[1]["query"]
    assert "query" in first_scan_query, "first scan query should have 'query' key"
    assert "bool" in first_scan_query["query"], "first scan query should have bool query"

    second_scan_call = mock_scan.call_args_list[1]
    second_scan_query = second_scan_call[1]["query"]
    assert "query" in second_scan_query, "second scan query should have 'query' key"
    assert "bool" in second_scan_query["query"], "second scan query should have bool query"

    # CRITICAL: Verify that allMessagesShouldMatch filtering works correctly
    # When allMessagesShouldMatch is True, the service filters to keep only test items
    # that appeared in results for ALL messages (in this case, both messages)
    # Based on mock data:
    # - test_item 3001 appears in both result sets → should be in final results
    # - test_item 3002 appears only in first result set → should NOT be in final results
    # - test_item 3003 appears only in second result set → should NOT be in final results

    # Verify we have results (test_item 3001 should match both messages and pass similarity threshold)
    assert (
        len(result) == 2
    ), "result should have exactly 2 items (2 logs from test_item 3001 that matched both messages)"

    # Extract test item IDs from results
    test_item_ids = {item.testItemId for item in result}

    # Verify only test_item 3001 is in results (matched ALL messages)
    assert test_item_ids == {3001}, "only test_item 3001 should be in results (matched all messages)"
    assert 3002 not in test_item_ids, "test_item 3002 should NOT be in results (matched only 1 message)"
    assert 3003 not in test_item_ids, "test_item 3003 should NOT be in results (matched only 1 message)"

    # Verify the specific log IDs that should be returned (from test_item 3001)
    log_ids = {item.logId for item in result}
    assert 300 in log_ids, "log 300 should be in results (from test_item 3001, first message)"
    assert 302 in log_ids, "log 302 should be in results (from test_item 3001, second message)"
    assert 301 not in log_ids, "log 301 should NOT be in results (from test_item 3002)"
    assert 303 not in log_ids, "log 303 should NOT be in results (from test_item 3003)"

    # Verify result items have correct structure
    for item in result:
        assert hasattr(item, "logId"), "result item should have logId"
        assert hasattr(item, "testItemId"), "result item should have testItemId"
        assert hasattr(item, "matchScore"), "result item should have matchScore"
        assert item.matchScore >= 0, "matchScore should be non-negative"
        assert item.matchScore <= 100, "matchScore should be <= 100"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_search_logs_with_merged_logs(
    mock_scan,
    search_service: SearchService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test search_logs correctly handles merged logs by finding all log IDs for merged test items."""
    # Prepare test data
    search_request = SearchLogs(**test_data["search_request"])
    # Use a single message that will return merged logs
    search_request.logMessages = ["Small log lines unique content extra words for processing"]
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{search_request.projectId}"

    # Configure mock_scan to return merged log with is_merged=True
    # The merged log needs to pass similarity threshold to trigger merged log expansion
    mock_scan.return_value = iter(test_data["search_results_with_merged_logs"])

    # Execute the method
    result = search_service.search_logs(search_request)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was called exactly once for searching
    # Note: Second scan for expanding merged logs is only triggered if results pass similarity threshold
    assert mock_scan.call_count == 1, "scan should be called once for searching"

    # Verify first scan is for searching logs
    first_scan_call = mock_scan.call_args_list[0]
    assert first_scan_call[0][0] == mocked_opensearch_client, "scan should use mocked OpenSearch client"
    assert first_scan_call[1]["index"] == expected_index_name, f"scan should use index '{expected_index_name}'"

    # Verify scan query structure for merged logs (message is empty, searches merged_small_logs)
    scan_query = first_scan_call[1]["query"]
    assert "query" in scan_query, "scan query should have 'query' key"
    assert "bool" in scan_query["query"], "scan query should have bool query"

    # Verify result structure
    assert isinstance(result, list), "result should be a list"

    # Verify result items have correct structure (if any results returned)
    for item in result:
        assert hasattr(item, "logId"), "result item should have logId"
        assert hasattr(item, "testItemId"), "result item should have testItemId"
        assert hasattr(item, "matchScore"), "result item should have matchScore"
        assert item.matchScore >= 0, "matchScore should be non-negative"
        assert item.matchScore <= 100, "matchScore should be <= 100"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_search_logs_with_merged_logs_expansion(
    mock_scan,
    search_service: SearchService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test search_logs expands merged logs by finding all log IDs when similarity threshold is met."""
    # Prepare test data with modified merged log that will pass similarity threshold
    search_request = SearchLogs(**test_data["search_request"])
    search_request.logMessages = ["Small log line Small log line Small log line"]  # Repeated for similarity
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{search_request.projectId}"

    # Modify the merged log data to have matching content for similarity
    merged_log_result = test_data["search_results_with_merged_logs"][0].copy()
    merged_log_result["_source"] = merged_log_result["_source"].copy()
    merged_log_result["_source"]["merged_small_logs"] = "Small log line\nSmall log line\nSmall log line"

    # Configure mock_scan:
    # 1st call: search returns merged log that passes similarity threshold
    # 2nd call: find all log IDs for test items with merged logs
    mock_scan.side_effect = [
        iter([merged_log_result]),
        iter(test_data["scan_results_for_test_items_with_merged_logs"]),
    ]

    # Execute the method
    result = search_service.search_logs(search_request)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was called exactly twice: once for search, once for expanding merged logs
    assert mock_scan.call_count == 2, "scan should be called twice: search and expand merged logs"

    # Verify first scan is for searching logs
    first_scan_call = mock_scan.call_args_list[0]
    assert first_scan_call[0][0] == mocked_opensearch_client, "first scan should use mocked OpenSearch client"
    assert first_scan_call[1]["index"] == expected_index_name, f"first scan should use index '{expected_index_name}'"

    # Verify second scan is for finding test items with merged logs
    second_scan_call = mock_scan.call_args_list[1]
    assert second_scan_call[0][0] == mocked_opensearch_client, "second scan should use mocked OpenSearch client"
    assert second_scan_call[1]["index"] == expected_index_name, f"second scan should use index '{expected_index_name}'"

    # Verify second scan query is for finding test items
    second_scan_query = second_scan_call[1]["query"]
    assert "query" in second_scan_query, "second scan query should have 'query' key"
    assert "bool" in second_scan_query["query"], "second scan query should have bool query"
    assert "filter" in second_scan_query["query"]["bool"], "second scan query should have filter clause"

    # Verify the query filters for test_item and is_merged
    filters = second_scan_query["query"]["bool"]["filter"]
    test_item_filter = next((f for f in filters if "terms" in f and "test_item" in f["terms"]), None)
    assert test_item_filter is not None, "second scan should filter by test_item"

    is_merged_filter = next((f for f in filters if "term" in f and "is_merged" in f["term"]), None)
    assert is_merged_filter is not None, "second scan should filter by is_merged"
    assert is_merged_filter["term"]["is_merged"] is False, "second scan should search for is_merged = False"

    # Verify result contains log IDs for the merged test item
    # Note: The expansion logic returns the found logs based on the scan results
    assert len(result) == 1, "result should contain exactly 1 log ID (the merged log)"

    # Verify all results have the test item ID 3004
    test_item_ids = {item.testItemId for item in result}
    assert 3004 in test_item_ids, "results should contain test_item 3004"

    # Verify result items have correct structure
    for item in result:
        assert hasattr(item, "logId"), "result item should have logId"
        assert hasattr(item, "testItemId"), "result item should have testItemId"
        assert hasattr(item, "matchScore"), "result item should have matchScore"
        assert item.matchScore >= 0, "matchScore should be non-negative"
        assert item.matchScore <= 100, "matchScore should be <= 100"
        assert item.testItemId == 3004, "all results should be for test_item 3004"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_search_logs_with_empty_log_messages(
    mock_scan,
    search_service: SearchService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test search_logs with empty log messages returns empty results."""
    # Prepare test data with empty log messages
    search_request = SearchLogs(**test_data["search_request"])
    search_request.logMessages = []
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{search_request.projectId}"

    # Execute the method
    result = search_service.search_logs(search_request)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was NOT called since there are no log messages
    mock_scan.assert_not_called()

    # Verify result is empty
    assert result == [], "Should return empty list when log messages are empty"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_search_logs_query_structure_for_merged_logs(
    mock_scan,
    search_service: SearchService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that search_logs builds correct query structure when message is empty (merged logs case)."""
    # Prepare test data with empty message (triggers merged logs query path)
    search_request = SearchLogs(**test_data["search_request"])
    search_request.logMessages = [""]  # Empty message
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{search_request.projectId}"

    # Configure mock_scan to return empty results (we only care about query structure)
    mock_scan.return_value = iter([])

    # Execute the method
    result = search_service.search_logs(search_request)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was NOT called because empty message is filtered out in _prepare_messages_for_queries
    mock_scan.assert_not_called()

    # Verify result is empty
    assert result == [], "Should return empty list when all messages are empty"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_search_logs_builds_query_with_found_exceptions(
    mock_scan,
    search_service: SearchService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that search_logs includes found_exceptions in query when exceptions are detected."""
    # Prepare test data
    search_request = SearchLogs(**test_data["search_request"])
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{search_request.projectId}"

    # Configure mock_scan to return results for both messages
    mock_scan.side_effect = [
        iter(test_data["search_results_for_first_message"]),
        iter(test_data["search_results_for_second_message"]),
    ]

    # Execute the method
    result = search_service.search_logs(search_request)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was called exactly twice (once for each message)
    assert mock_scan.call_count == 2, "scan should be called twice: once for each log message"

    # Get the first query that was passed to scan
    first_scan_call = mock_scan.call_args_list[0]
    scan_query = first_scan_call[1]["query"]

    # Verify query contains must clause
    must_clause = scan_query["query"]["bool"]["must"]

    # The query should contain more_like_this for message and potentially found_exceptions
    # Since the test message contains "NullPointerException", it should be detected
    mlt_queries = [m for m in must_clause if "more_like_this" in m]
    assert len(mlt_queries) == 2, "query should have exactly 2 more_like_this clauses (message and found_exceptions)"

    # Verify found_exceptions more_like_this is present (since test message contains NullPointerException)
    exception_mlt = next(
        (m for m in mlt_queries if "found_exceptions" in m["more_like_this"]["fields"]),
        None,
    )
    assert (
        exception_mlt is not None
    ), "found_exceptions more_like_this query should be present when exception is detected in message"

    # Verify found_exceptions query has the expected structure
    assert "found_exceptions" in exception_mlt["more_like_this"]["fields"], "found_exceptions should be in fields"
    assert "like" in exception_mlt["more_like_this"], "found_exceptions query should have 'like' field"

    # The minimum_should_match should be set to "1" for found_exceptions
    assert (
        "minimum_should_match" in exception_mlt["more_like_this"]
    ), "found_exceptions query should have minimum_should_match parameter"
    assert (
        exception_mlt["more_like_this"]["minimum_should_match"] == "1"
    ), "found_exceptions should use minimum_should_match=1"

    # Verify result
    assert isinstance(result, list), "result should be a list"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_search_logs_with_potential_status_codes(
    mock_scan,
    search_service: SearchService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict,
) -> None:
    """Test that search_logs correctly filters results based on potential_status_codes matching.

    This test verifies that:
    1. Logs with matching status codes (404, 500) are included in results
    2. Logs with different status codes (403, 502) are filtered out due to <99% similarity
    3. The status code filtering logic works correctly when both query and result have status codes
    """
    # Prepare test data with messages containing status codes
    search_request = SearchLogs(**test_data["search_request_with_status_codes"])
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{search_request.projectId}"

    # Configure mock_scan to return results with various status codes
    # First scan returns logs with status codes 404 and 403
    # Second scan returns logs with status codes 500 and 502
    mock_scan.side_effect = [
        iter(test_data["search_results_for_status_code_404"]),  # Returns logs with 404 and 403
        iter(test_data["search_results_for_status_code_500"]),  # Returns logs with 500 and 502
    ]

    # Execute the method
    result = search_service.search_logs(search_request)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once_with(index=expected_index_name)

    # Verify scan was called exactly twice (once for each message)
    assert mock_scan.call_count == 2, "scan should be called twice: once for each log message"

    # Verify result structure
    assert isinstance(result, list), "result should be a list"
    assert len(result) == 2, "result should contain exactly 2 matching logs (one for each message)"

    # Extract log IDs and test item IDs from results
    log_ids = {item.logId for item in result}
    test_item_ids = {item.testItemId for item in result}

    # Verify that logs with exact matching status codes and high similarity are included
    # Log 400 has status code 404 and is an exact match for the first message
    # Log 403 has status code 500 and is an exact match for the second message
    assert 400 in log_ids, "log 400 with status code 404 (exact match) should be in results"
    assert 403 in log_ids, "log 403 with status code 500 (exact match) should be in results"

    # Verify that logs with non-matching status codes are excluded
    # Log 401 has status code 403 (should not match query status code 404)
    # Log 404 has status code 502 (should not match query status code 500)
    assert 401 not in log_ids, "log 401 with status code 403 should NOT be in results (mismatched status code)"
    assert 404 not in log_ids, "log 404 with status code 502 should NOT be in results (mismatched status code)"

    # Verify test item IDs include the matching ones
    assert 4001 in test_item_ids, "test_item 4001 (status 404) should be in results"
    assert 4004 in test_item_ids, "test_item 4004 (status 500) should be in results"
    # Verify test items with wrong status codes are excluded
    assert 4002 not in test_item_ids, "test_item 4002 (status 403) should NOT be in results"
    assert 4005 not in test_item_ids, "test_item 4005 (status 502) should NOT be in results"

    # Verify result items have correct structure
    for item in result:
        assert hasattr(item, "logId"), "result item should have logId"
        assert hasattr(item, "testItemId"), "result item should have testItemId"
        assert hasattr(item, "matchScore"), "result item should have matchScore"
        assert item.matchScore >= 0, "matchScore should be non-negative"
        assert item.matchScore <= 100, "matchScore should be <= 100"
        # Verify that only test items with correct status codes are present
        assert item.testItemId not in {4002, 4005}, "test_items with wrong status codes should be filtered out"

    # Additional verification: check that potential_status_codes filtering is working
    # All results should have status codes that match the query (either 404 or 500)
    # This is validated by the fact that logs 401 and 404 (with status 403 and 502) are excluded
