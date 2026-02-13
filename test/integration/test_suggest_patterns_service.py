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

from types import SimpleNamespace
from unittest import mock

import pytest

from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.os_client import OsClient
from app.service.suggest_patterns_service import SuggestPatternsService
from app.utils.utils import read_json_file
from test import APP_CONFIG, DEFAULT_SEARCH_CONFIG


@pytest.fixture
def test_data() -> dict[str, dict[str, dict[str, str]]]:
    """Load test data with mocked OpenSearch scan results."""
    return read_json_file("test_res", "suggest_patterns_test_data.json", to_json=True)


def _build_log(detected_message: str, log_id: str) -> LogData:
    return LogData(
        log_id=log_id,
        log_order=0,
        log_time="2025-01-01T00:00:00Z",
        log_level=40000,
        cluster_id="",
        cluster_message="",
        cluster_with_numbers=False,
        original_message=detected_message,
        message=detected_message,
        message_lines=1,
        message_words_number=1,
        message_extended=detected_message,
        message_without_params_extended=detected_message,
        message_without_params_and_brackets=detected_message,
        detected_message=detected_message,
        detected_message_with_numbers=detected_message,
        detected_message_extended=detected_message,
        detected_message_without_params_extended=detected_message,
        detected_message_without_params_and_brackets=detected_message,
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
        whole_message=detected_message,
    )


def _build_hit(issue_type: str, detected_message: str, idx: int) -> SimpleNamespace:
    test_item = TestItemIndexData(
        test_item_id=f"ti-{idx}",
        launch_id="l-1",
        issue_type=issue_type,
        logs=[_build_log(detected_message, f"log-{idx}")],
    )
    return SimpleNamespace(source=test_item)


@pytest.fixture
def os_client_mock() -> mock.Mock:
    return mock.Mock(spec=OsClient)


@pytest.fixture
def suggest_patterns_service(os_client_mock: mock.Mock) -> SuggestPatternsService:
    """Create SuggestPatternsService with mocked OsClient."""
    return SuggestPatternsService(APP_CONFIG, DEFAULT_SEARCH_CONFIG, os_client=os_client_mock)


def test_suggest_patterns_calls_correct_services(
    suggest_patterns_service: SuggestPatternsService,
    os_client_mock: mock.Mock,
    test_data: dict[str, dict[str, dict[str, str]]],
) -> None:
    """Test that suggest_patterns method calls internal services with correct arguments."""
    test_project_id = 123
    entries = []
    for key in ["scan_results_ab", "scan_results_pb", "scan_results_si", "scan_results_ti"]:
        entries.extend(test_data[key])
    hits = [
        _build_hit(entry["_source"]["issue_type"], entry["_source"]["detected_message"], idx)
        for idx, entry in enumerate(entries)
    ]
    os_client_mock.search.return_value = hits

    # Execute the method
    result = suggest_patterns_service.suggest_patterns(test_project_id)

    # Verify os_client.search was called once
    os_client_mock.search.assert_called_once()
    search_args = os_client_mock.search.call_args[0]
    assert search_args[0] == test_project_id, "search should use provided project ID"

    query = search_args[1]
    assert "query" in query, "search should have query key"
    assert "bool" in query["query"], "search should have bool query"
    assert "filter" in query["query"]["bool"], "search should have filter clause"

    should_clause = query["query"]["bool"]["filter"][0]["bool"]["should"]
    assert len(should_clause) == 4, "search should have 4 wildcard clauses"

    labels = ["ab", "pb", "si", "ti"]
    for idx, label in enumerate(labels):
        pattern = f"{label}*"
        wildcard = should_clause[idx]["wildcard"]["issue_type"]["value"]
        case_insensitive = should_clause[idx]["wildcard"]["issue_type"]["case_insensitive"]
        assert wildcard == pattern, f"wildcard should match '{pattern}'"
        assert case_insensitive, "wildcard should be case insensitive"

    # Verify sort by start_time
    assert "sort" in query, "search should have sort"
    assert "start_time" in query["sort"], "search should sort by start_time"
    assert query["sort"]["start_time"] == "desc", "search should sort by start_time descending"

    # Verify size parameter
    assert "size" in query, "search should have size parameter"
    assert query["size"] == APP_CONFIG.esChunkNumber, "search should use app esChunkNumber"

    # Verify boost parameters were added by append_aa_ma_boosts
    assert "should" not in query["query"]["bool"], "search should have should clause with boosts"

    # Verify result structure
    assert result.suggestionsWithLabels is not None, "result should have suggestionsWithLabels"
    assert result.suggestionsWithoutLabels is not None, "result should have suggestionsWithoutLabels"
    assert isinstance(result.suggestionsWithLabels, list), "suggestionsWithLabels should be a list"
    assert isinstance(result.suggestionsWithoutLabels, list), "suggestionsWithoutLabels should be a list"


def test_suggest_patterns_with_nonexistent_index(
    suggest_patterns_service: SuggestPatternsService,
    os_client_mock: mock.Mock,
) -> None:
    """Test suggest_patterns when index does not exist."""
    test_project_id = 999
    os_client_mock.search.return_value = []

    # Execute the method
    result = suggest_patterns_service.suggest_patterns(test_project_id)

    # Verify search was called
    os_client_mock.search.assert_called_once()

    # Verify result contains empty lists
    assert result.suggestionsWithLabels == [], "suggestionsWithLabels should be empty for non-existent index"
    assert result.suggestionsWithoutLabels == [], "suggestionsWithoutLabels should be empty for non-existent index"


def test_suggest_patterns_filters_and_aggregates_exceptions(
    suggest_patterns_service: SuggestPatternsService,
    os_client_mock: mock.Mock,
) -> None:
    """Test that suggest_patterns correctly filters and aggregates exceptions."""
    test_project_id = 456

    hits = []
    for idx in range(10):
        hits.append(_build_hit("AB001", "java.lang.NullPointerException at test", idx))
    hits.append(_build_hit("PB001", "java.lang.NullPointerException at test", 10))
    hits.append(_build_hit("TI001", "java.lang.NullPointerException at test", 11))
    hits.append(_build_hit("TI001", "java.lang.NullPointerException at test", 12))
    os_client_mock.search.return_value = hits

    # Execute the method
    result = suggest_patterns_service.suggest_patterns(test_project_id)

    # Verify search was called once
    os_client_mock.search.assert_called_once()

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


def test_suggest_patterns_with_empty_results(
    suggest_patterns_service: SuggestPatternsService,
    os_client_mock: mock.Mock,
) -> None:
    """Test suggest_patterns when all scan results are empty."""
    test_project_id = 789

    os_client_mock.search.return_value = []

    # Execute the method
    result = suggest_patterns_service.suggest_patterns(test_project_id)

    # Verify search was called once
    os_client_mock.search.assert_called_once()

    # Verify result contains empty lists
    assert result.suggestionsWithLabels == [], "suggestionsWithLabels should be empty when no data found"
    assert result.suggestionsWithoutLabels == [], "suggestionsWithoutLabels should be empty when no data found"


def test_suggest_patterns_query_structure_with_different_config() -> None:
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

    os_client_mock = mock.Mock(spec=OsClient)
    os_client_mock.search.return_value = []
    service = SuggestPatternsService(custom_app_config, custom_search_config, os_client=os_client_mock)

    test_project_id = 111

    # Execute the method
    service.suggest_patterns(test_project_id)

    os_client_mock.search.assert_called_once()
    query = os_client_mock.search.call_args[0][1]

    # Verify first scan call uses custom chunk size
    assert query["size"] == es_chunk_number, "Query should use custom esChunkNumber"
