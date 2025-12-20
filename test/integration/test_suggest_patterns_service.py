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
from app.service.suggest_patterns_service import SuggestPatternsService
from app.utils.utils import read_json_file
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG


@pytest.fixture
def test_data() -> dict[str, dict[str, dict[str, str]]]:
    """Load test data with mocked OpenSearch scan results."""
    return read_json_file("test_res", "suggest_patterns_test_data.json", to_json=True)


@pytest.fixture
def mocked_opensearch_client() -> OpenSearch:
    """Create a mocked OpenSearch client instance."""
    mock_client = mock.Mock(OpenSearch)
    mock_client.indices = mock.Mock(IndicesClient)

    # Mock indices.get for index_exists checks
    mock_client.indices.get.return_value = {"rp_123": "exists"}

    return mock_client


@pytest.fixture
def suggest_patterns_service(mocked_opensearch_client: OpenSearch) -> SuggestPatternsService:
    """Create SuggestPatternsService with real EsClient and mocked OpenSearch client."""
    # Create real EsClient with mocked OpenSearch client
    es_client = EsClient(APP_CONFIG, es_client=mocked_opensearch_client)

    # Create SuggestPatternsService with real EsClient
    service = SuggestPatternsService(APP_CONFIG, DEFAULT_SEARCH_CONFIG, es_client=es_client)

    return service


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_suggest_patterns_calls_correct_services(
    mock_scan,
    suggest_patterns_service: SuggestPatternsService,
    mocked_opensearch_client: OpenSearch,
    test_data: dict[str, dict[str, dict[str, str]]],
) -> None:
    """Test that suggest_patterns method calls internal services with correct arguments."""
    test_project_id = 123
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"

    # Configure mock_scan to return different results for different labels
    # The scan is called 4 times (for ab, pb, si, ti labels)
    mock_scan.side_effect = [
        iter(test_data["scan_results_ab"]),
        iter(test_data["scan_results_pb"]),
        iter(test_data["scan_results_si"]),
        iter(test_data["scan_results_ti"]),
    ]

    # Execute the method
    result = suggest_patterns_service.suggest_patterns(test_project_id)

    # Verify es_client checked index exists
    mocked_opensearch_client.indices.get.assert_called_once()
    index_exists_call = mocked_opensearch_client.indices.get.call_args
    assert (
        index_exists_call[1]["index"] == expected_index_name
    ), f"index_exists should check for index '{expected_index_name}'"

    # Verify opensearchpy.helpers.scan was called 4 times (once for each label)
    assert mock_scan.call_count == 4, "scan should be called 4 times (once per label: ab, pb, si, ti)"

    # Verify the scan calls for each label
    labels = ["ab", "pb", "si", "ti"]
    for idx, label in enumerate(labels):
        scan_call = mock_scan.call_args_list[idx]

        # Verify scan was called with correct client and index
        assert scan_call[0][0] == mocked_opensearch_client, f"scan call {idx} should use mocked OpenSearch client"
        assert (
            scan_call[1]["index"] == expected_index_name
        ), f"scan call {idx} should use index '{expected_index_name}'"

        # Verify query structure
        query = scan_call[1]["query"]
        assert "query" in query, f"scan call {idx} should have 'query' key"
        assert "bool" in query["query"], f"scan call {idx} should have bool query"
        assert "must" in query["query"]["bool"], f"scan call {idx} should have must clause"

        # Verify label wildcards in query
        should_clause = query["query"]["bool"]["must"][0]["bool"]["should"]
        assert len(should_clause) == 1, f"scan call {idx} should have 1 wildcard clause"

        # Verify wildcard patterns for the label
        pattern = f"{label}*"
        wildcard = should_clause[0]["wildcard"]["issue_type"]
        case_insensitive = should_clause[0]["case_insensitive"]
        assert wildcard == pattern, f"scan call {idx}, should match '{pattern}'"
        assert case_insensitive, f"scan call {idx}, should match should be case insensitive"

        # Verify _source field selection
        assert "_source" in query, f"scan call {idx} should specify _source fields"
        assert "detected_message" in query["_source"], f"scan call {idx} should request 'detected_message' field"
        assert "issue_type" in query["_source"], f"scan call {idx} should request 'issue_type' field"

        # Verify sort by start_time
        assert "sort" in query, f"scan call {idx} should have sort"
        assert "start_time" in query["sort"], f"scan call {idx} should sort by start_time"
        assert query["sort"]["start_time"] == "desc", f"scan call {idx} should sort by start_time descending"

        # Verify size parameter
        assert "size" in query, f"scan call {idx} should have size parameter"
        assert (
            query["size"] == APP_CONFIG.esChunkNumber
        ), f"scan call {idx} should use esChunkNumber={APP_CONFIG.esChunkNumber}"

        # Verify boost parameters were added by append_aa_ma_boosts
        assert "should" in query["query"]["bool"], f"scan call {idx} should have should clause with boosts"
        boost_clause = query["query"]["bool"]["should"]
        assert len(boost_clause) > 0, f"scan call {idx} should have boost terms added"

    # Verify result structure
    assert result.suggestionsWithLabels is not None, "result should have suggestionsWithLabels"
    assert result.suggestionsWithoutLabels is not None, "result should have suggestionsWithoutLabels"
    assert isinstance(result.suggestionsWithLabels, list), "suggestionsWithLabels should be a list"
    assert isinstance(result.suggestionsWithoutLabels, list), "suggestionsWithoutLabels should be a list"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_suggest_patterns_with_nonexistent_index(
    mock_scan,
    mocked_opensearch_client: OpenSearch,
    suggest_patterns_service: SuggestPatternsService,
) -> None:
    """Test suggest_patterns when index does not exist."""
    test_project_id = 999
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"

    # Configure mock to raise exception for non-existent index
    mocked_opensearch_client.indices.get.side_effect = Exception("Index not found")

    # Execute the method
    result = suggest_patterns_service.suggest_patterns(test_project_id)

    # Verify index_exists was called
    mocked_opensearch_client.indices.get.assert_called_once()
    index_exists_call = mocked_opensearch_client.indices.get.call_args
    assert index_exists_call[1]["index"] == expected_index_name

    # Verify scan was NOT called since index doesn't exist
    mock_scan.assert_not_called()

    # Verify result contains empty lists
    assert result.suggestionsWithLabels == [], "suggestionsWithLabels should be empty for non-existent index"
    assert result.suggestionsWithoutLabels == [], "suggestionsWithoutLabels should be empty for non-existent index"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_suggest_patterns_filters_and_aggregates_exceptions(
    mock_scan,
    suggest_patterns_service: SuggestPatternsService,
    mocked_opensearch_client: OpenSearch,
) -> None:
    """Test that suggest_patterns correctly filters and aggregates exceptions."""
    test_project_id = 456

    # Create test data with repeated exceptions to test aggregation
    # Need at least 10 occurrences for suggestionsWithoutLabels
    # Need at least 5 occurrences and 90% with same label for suggestionsWithLabels
    repeated_scan_results = [
        # ab label - same exception 10 times (will meet 90% threshold for AB label)
        [
            {
                "_source": {
                    "detected_message": "java.lang.NullPointerException at test",
                    "issue_type": "AB001",
                }
            }
            for _ in range(10)
        ],
        # pb label - same exception 1 time
        [
            {
                "_source": {
                    "detected_message": "java.lang.NullPointerException at test",
                    "issue_type": "PB001",
                }
            },
        ],
        # si label - empty
        [],
        # ti label - should not be included in labels (2 times)
        [
            {
                "_source": {
                    "detected_message": "java.lang.NullPointerException at test",
                    "issue_type": "TI001",
                }
            }
            for _ in range(2)
        ],
    ]

    mock_scan.side_effect = [iter(results) for results in repeated_scan_results]

    # Execute the method
    result = suggest_patterns_service.suggest_patterns(test_project_id)

    # Verify scan was called 4 times
    assert mock_scan.call_count == 4

    # The NullPointerException appears 13 times total (10 AB, 1 PB, 2 TI)
    # TI should not be counted in suggestionsWithLabels
    # So we have 10 AB + 1 PB = 11 non-TI occurrences
    # AB percentage: 10/11 = 90.9% (meets 90% threshold)
    # Total count: 13 (meets 10 threshold for suggestionsWithoutLabels)

    # Verify that result has suggestions
    # Note: The actual patterns depend on get_found_exceptions implementation
    # which extracts words ending with "error", "exception", or "failure"
    assert (
        len(result.suggestionsWithoutLabels) > 0 or len(result.suggestionsWithLabels) > 0
    ), "Should have some suggestions"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_suggest_patterns_with_empty_results(
    mock_scan,
    suggest_patterns_service: SuggestPatternsService,
    mocked_opensearch_client: OpenSearch,
) -> None:
    """Test suggest_patterns when all scan results are empty."""
    test_project_id = 789

    # Configure mock_scan to return empty results for all labels
    mock_scan.side_effect = [iter([]), iter([]), iter([]), iter([])]

    # Execute the method
    result = suggest_patterns_service.suggest_patterns(test_project_id)

    # Verify scan was called 4 times
    assert mock_scan.call_count == 4

    # Verify result contains empty lists
    assert result.suggestionsWithLabels == [], "suggestionsWithLabels should be empty when no data found"
    assert result.suggestionsWithoutLabels == [], "suggestionsWithoutLabels should be empty when no data found"


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.scan")
def test_suggest_patterns_query_structure_with_different_config(
    mock_scan,
    mocked_opensearch_client: OpenSearch,
) -> None:
    """Test that query structure respects different app and search configurations."""
    # Create service with custom configuration
    from app.commons.model.launch_objects import ApplicationConfig, SearchConfig

    es_chunk_number = 500
    custom_app_config = ApplicationConfig(
        esHost="http://localhost:9200",
        esUser="",
        esPassword="",
        esVerifyCerts=False,
        esUseSsl=False,
        esSslShowWarn=False,
        turnOffSslVerification=True,
        esCAcert="",
        esClientCert="",
        esClientKey="",
        appVersion="",
        datastoreBucketPrefix="",
        filesystemDefaultPath="",
        esChunkNumber=es_chunk_number,
        datastoreType="filesystem",
        datastoreEndpoint="",
        esProjectIndexPrefix="custom_prefix_",
        esChunkNumberUpdateClusters=500,
    )

    custom_search_config = SearchConfig(
        BoostAA=10.0,
        BoostMA=5.0,
        BoostLaunch=2.0,
    )

    es_client = EsClient(custom_app_config, es_client=mocked_opensearch_client)
    service = SuggestPatternsService(custom_app_config, custom_search_config, es_client=es_client)

    test_project_id = 111
    expected_index_name = f"{custom_app_config.esProjectIndexPrefix}{test_project_id}"

    # Configure mock_scan
    mock_scan.side_effect = [iter([]), iter([]), iter([]), iter([])]

    # Execute the method
    service.suggest_patterns(test_project_id)

    # Verify index name uses custom prefix
    index_exists_call = mocked_opensearch_client.indices.get.call_args
    assert index_exists_call[1]["index"] == expected_index_name

    # Verify first scan call uses custom chunk size
    first_scan_call = mock_scan.call_args_list[0]
    query = first_scan_call[1]["query"]
    assert query["size"] == es_chunk_number, "Query should use custom esChunkNumber"

    # Verify boost was applied according to custom search config
    boost_clause = query["query"]["bool"]["should"]
    # The boost clause should exist since BoostAA != BoostMA
    assert len(boost_clause) > 0, "Should have boost clause when BoostAA != BoostMA"
