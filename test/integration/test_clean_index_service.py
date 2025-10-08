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
from app.commons.model.launch_objects import CleanIndex, CleanIndexStrIds
from app.service.clean_index_service import CleanIndexService
from app.service.suggest_info_service import SuggestInfoService
from test import APP_CONFIG


@pytest.fixture
def mocked_opensearch_client() -> OpenSearch:
    """Create a mocked OpenSearch client instance"""
    mock_client = mock.Mock(OpenSearch)
    mock_client.indices = mock.Mock(IndicesClient)

    # Mock indices.get for index_exists checks
    mock_client.indices.get.return_value = {"index": "exists"}

    # Configure methods for delete_logs (bulk operations)
    mock_client.bulk.return_value = {"took": 100, "errors": False, "items": []}

    # Configure methods for delete_by_query operations
    mock_client.delete_by_query.return_value = {"deleted": 5}

    # Configure methods for search operations (scan helper)
    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {"_id": "launch_1"},
                {"_id": "launch_2"},
            ]
        }
    }

    return mock_client


@pytest.fixture
def mocked_suggest_info_service() -> SuggestInfoService:
    """Create a mocked SuggestInfoService instance"""
    mock_service = mock.Mock(SuggestInfoService)
    mock_service.clean_suggest_info_logs.return_value = None
    mock_service.clean_suggest_info_logs_by_test_item.return_value = 2
    mock_service.clean_suggest_info_logs_by_launch_id.return_value = 3
    return mock_service


@pytest.fixture
def clean_index_service(
    mocked_opensearch_client: OpenSearch, mocked_suggest_info_service: SuggestInfoService
) -> CleanIndexService:
    """Create CleanIndexService with real EsClient and mocked dependencies"""
    # Create real EsClient with mocked OpenSearch client
    es_client = EsClient(APP_CONFIG, es_client=mocked_opensearch_client)

    # Create CleanIndexService with mocked suggest_info_service
    service = CleanIndexService(
        APP_CONFIG,
        es_client=es_client,
        suggest_info_service=mocked_suggest_info_service,
    )

    return service


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_delete_logs_calls_correct_services(
    mock_scan,
    mock_bulk,
    clean_index_service: CleanIndexService,
    mocked_opensearch_client: OpenSearch,
    mocked_suggest_info_service: SuggestInfoService,
) -> None:
    """Test that delete_logs method calls internal services with correct arguments."""
    # Configure mock bulk to return success
    mock_bulk.return_value = (5, [])  # (success_count, errors)
    # Configure mock scan to return test items
    mock_scan.return_value = iter([{"_source": {"test_item": "item_1"}}, {"_source": {"test_item": "item_2"}}])

    test_project_id = 123
    test_log_ids = [1, 2, 3, 4, 5]
    clean_index = CleanIndex(ids=test_log_ids, project=test_project_id)

    result = clean_index_service.delete_logs(clean_index)

    assert result == 5, "delete_logs should return the success count from bulk operation"

    # Verify es_client checked index exists
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"
    mocked_opensearch_client.indices.get.assert_called_once()
    call_args = mocked_opensearch_client.indices.get.call_args_list
    assert call_args[0][1]["index"] == expected_index_name

    # Verify opensearchpy.helpers.bulk was called
    mock_bulk.assert_called_once()

    # Verify suggest_info_service.clean_suggest_info_logs was called with correct arguments
    mocked_suggest_info_service.clean_suggest_info_logs.assert_called_once()
    clean_index_arg = mocked_suggest_info_service.clean_suggest_info_logs.call_args[0][0]
    assert isinstance(clean_index_arg, CleanIndexStrIds)
    assert clean_index_arg.project == test_project_id
    assert clean_index_arg.ids == [str(log_id) for log_id in test_log_ids]


# noinspection PyUnresolvedReferences
def test_delete_test_items_calls_correct_services(
    clean_index_service: CleanIndexService,
    mocked_opensearch_client: OpenSearch,
    mocked_suggest_info_service: SuggestInfoService,
) -> None:
    """Test that delete_test_items method calls internal services with correct arguments."""
    test_project_id = 456
    test_item_ids = [10, 20, 30]
    remove_items_info = {
        "project": test_project_id,
        "itemsToDelete": test_item_ids,
    }

    result = clean_index_service.delete_test_items(remove_items_info)

    assert result == 5, "delete_test_items should return the number of deleted logs"

    # Verify es_client checked index exists
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"
    mocked_opensearch_client.indices.get.assert_called_once()
    call_args = mocked_opensearch_client.indices.get.call_args_list
    assert call_args[0][1]["index"] == expected_index_name

    # Verify es_client.delete_by_query was called with correct index name
    mocked_opensearch_client.delete_by_query.assert_called_once()
    call_args = mocked_opensearch_client.delete_by_query.call_args
    assert call_args[1]["index"] == expected_index_name

    # Verify suggest_info_service.clean_suggest_info_logs_by_test_item was called
    mocked_suggest_info_service.clean_suggest_info_logs_by_test_item.assert_called_once_with(remove_items_info)


# noinspection PyUnresolvedReferences
def test_delete_launches_calls_correct_services(
    clean_index_service: CleanIndexService,
    mocked_opensearch_client: OpenSearch,
    mocked_suggest_info_service: SuggestInfoService,
) -> None:
    """Test that delete_launches method calls internal services with correct arguments."""
    test_project_id = 789
    test_launch_ids = [100, 200]
    launch_remove_info = {
        "project": test_project_id,
        "launch_ids": test_launch_ids,
    }

    result = clean_index_service.delete_launches(launch_remove_info)

    assert result == 5, "delete_launches should return the number of deleted logs"

    # Verify es_client checked index exists
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"
    mocked_opensearch_client.indices.get.assert_called_once()
    call_args = mocked_opensearch_client.indices.get.call_args_list
    assert call_args[0][1]["index"] == expected_index_name

    # Verify es_client.delete_by_query was called with correct index name
    mocked_opensearch_client.delete_by_query.assert_called_once()
    call_args = mocked_opensearch_client.delete_by_query.call_args
    assert call_args[1]["index"] == expected_index_name

    # Verify suggest_info_service.clean_suggest_info_logs_by_launch_id was called
    mocked_suggest_info_service.clean_suggest_info_logs_by_launch_id.assert_called_once_with(launch_remove_info)


# noinspection PyUnresolvedReferences
def test_remove_by_launch_start_time_calls_correct_services(
    clean_index_service: CleanIndexService,
    mocked_opensearch_client: OpenSearch,
    mocked_suggest_info_service: SuggestInfoService,
) -> None:
    """Test that remove_by_launch_start_time method calls internal services in correct order."""
    test_project_id = 321
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    remove_by_launch_start_time_info = {
        "project": test_project_id,
        "interval_start_date": start_date,
        "interval_end_date": end_date,
    }

    result = clean_index_service.remove_by_launch_start_time(remove_by_launch_start_time_info)

    assert result == 5, "remove_by_launch_start_time should return the number of deleted logs"

    # Verify es_client methods were called with correct index name
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"

    # Verify search was called first (for get_launch_ids_by_start_time_range)
    mocked_opensearch_client.search.assert_called_once()
    search_call_args = mocked_opensearch_client.search.call_args
    assert search_call_args[1]["index"] == expected_index_name

    # Verify delete_by_query was called (for remove_by_launch_start_time_range)
    mocked_opensearch_client.delete_by_query.assert_called_once()
    delete_call_args = mocked_opensearch_client.delete_by_query.call_args
    assert delete_call_args[1]["index"] == expected_index_name

    # Verify suggest_info_service.clean_suggest_info_logs_by_launch_id was called
    mocked_suggest_info_service.clean_suggest_info_logs_by_launch_id.assert_called_once()
    launch_remove_info_arg = mocked_suggest_info_service.clean_suggest_info_logs_by_launch_id.call_args[0][0]
    assert launch_remove_info_arg["project"] == test_project_id
    assert "launch_ids" in launch_remove_info_arg


# noinspection PyUnresolvedReferences
def test_remove_by_log_time_calls_correct_services(
    clean_index_service: CleanIndexService,
    mocked_opensearch_client: OpenSearch,
    mocked_suggest_info_service: SuggestInfoService,
) -> None:
    """Test that remove_by_log_time method calls internal services in correct order."""
    test_project_id = 654
    start_date = "2023-06-01"
    end_date = "2023-06-30"
    remove_by_log_time_info = {
        "project": test_project_id,
        "interval_start_date": start_date,
        "interval_end_date": end_date,
    }

    result = clean_index_service.remove_by_log_time(remove_by_log_time_info)

    assert result == 5, "remove_by_log_time should return the number of deleted logs"

    # Verify es_client methods were called with correct index name
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"

    # Verify search was called first (for get_log_ids_by_log_time_range)
    mocked_opensearch_client.search.assert_called_once()
    search_call_args = mocked_opensearch_client.search.call_args
    assert search_call_args[1]["index"] == expected_index_name

    # Verify delete_by_query was called (for remove_by_log_time_range)
    mocked_opensearch_client.delete_by_query.assert_called_once()
    delete_call_args = mocked_opensearch_client.delete_by_query.call_args
    assert delete_call_args[1]["index"] == expected_index_name

    # Verify suggest_info_service.clean_suggest_info_logs was called
    mocked_suggest_info_service.clean_suggest_info_logs.assert_called_once()
    clean_index_arg = mocked_suggest_info_service.clean_suggest_info_logs.call_args[0][0]
    assert isinstance(clean_index_arg, CleanIndexStrIds)
    assert clean_index_arg.project == test_project_id


# noinspection PyUnresolvedReferences
@mock.patch("opensearchpy.helpers.bulk")
@mock.patch("opensearchpy.helpers.scan")
def test_delete_logs_with_empty_list(
    mock_scan,
    mock_bulk,
    clean_index_service: CleanIndexService,
    mocked_opensearch_client: OpenSearch,
    mocked_suggest_info_service: SuggestInfoService,
) -> None:
    """Test delete_logs with empty log IDs list"""
    # Configure mock bulk to return success
    mock_bulk.return_value = (0, [])
    mock_scan.return_value = iter([])

    test_project_id = 999
    clean_index = CleanIndex(ids=[], project=test_project_id)

    result = clean_index_service.delete_logs(clean_index)

    assert result == 0
    mocked_suggest_info_service.clean_suggest_info_logs.assert_called_once()
    clean_index_arg = mocked_suggest_info_service.clean_suggest_info_logs.call_args[0][0]
    assert clean_index_arg.ids == []


def test_delete_test_items_with_different_project(
    clean_index_service: CleanIndexService,
    mocked_opensearch_client: OpenSearch,
    mocked_suggest_info_service: SuggestInfoService,
) -> None:
    """Test delete_test_items with different project ID"""
    test_project_id = 111
    remove_items_info = {
        "project": test_project_id,
        "itemsToDelete": [1, 2, 3],
    }

    result = clean_index_service.delete_test_items(remove_items_info)

    assert result == 5
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"
    call_args = mocked_opensearch_client.delete_by_query.call_args
    assert call_args[1]["index"] == expected_index_name
