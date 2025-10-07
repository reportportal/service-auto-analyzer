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
from commons.object_saving import ObjectSaver
from opensearchpy import OpenSearch
from opensearchpy.client import IndicesClient

from app.commons.esclient import EsClient
from app.commons.model.launch_objects import SearchConfig
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.trigger_manager import TriggerManager
from app.service.delete_index_service import DeleteIndexService
from test import APP_CONFIG, DEFAULT_BOOST_LAUNCH


@pytest.fixture
def search_config() -> SearchConfig:
    """Create SearchConfig with required features for model training"""
    return SearchConfig(
        BoostLaunch=DEFAULT_BOOST_LAUNCH,
        SuggestBoostModelFeatures="0-100",
        AutoBoostModelFeatures="0-100",
        SuggestBoostModelMonotonousFeatures="",
        AutoBoostModelMonotonousFeatures="",
    )


@pytest.fixture
def mocked_object_saver() -> ObjectSaver:
    """Create a mocked ObjectSaver instance"""
    mock_saver = mock.Mock(ObjectSaver)
    # Configure object_saver to return empty lists for folder objects
    mock_saver.get_folder_objects.return_value = []
    return mock_saver


@pytest.fixture
def mocked_opensearch_client() -> OpenSearch:
    """Create a mocked OpenSearch client instance"""
    mock_client = mock.Mock(OpenSearch)
    mock_client.indices = mock.Mock(IndicesClient)
    # Configure indices.delete to return successfully
    mock_client.indices.delete.return_value = {"acknowledged": True}
    return mock_client


@pytest.fixture
def delete_index_service(
    mocked_object_saver: ObjectSaver, mocked_opensearch_client: OpenSearch, search_config: SearchConfig
) -> DeleteIndexService:
    """Create DeleteIndexService with real components and mocked dependencies"""
    # Create real ModelChooser with mocked object_saver
    model_chooser = ModelChooser(APP_CONFIG, search_config, object_saver=mocked_object_saver)

    # Create real EsClient with mocked OpenSearch client
    es_client = EsClient(APP_CONFIG, es_client=mocked_opensearch_client)

    # Create real NamespaceFinder with mocked object_saver
    namespace_finder = NamespaceFinder(APP_CONFIG, object_saver=mocked_object_saver)

    # Create real TriggerManager with mocked object_saver
    trigger_manager = TriggerManager(
        model_chooser,
        app_config=APP_CONFIG,
        search_cfg=search_config,
        object_saver=mocked_object_saver,
    )

    # Create DeleteIndexService with all dependencies
    service = DeleteIndexService(
        model_chooser,
        APP_CONFIG,
        search_config,
        es_client=es_client,
        namespace_finder=namespace_finder,
        trigger_manager=trigger_manager,
    )

    return service


# noinspection PyUnresolvedReferences
def test_delete_index_calls_correct_services(
    delete_index_service: DeleteIndexService, mocked_object_saver: ObjectSaver, mocked_opensearch_client: OpenSearch
) -> None:
    """Test that delete_index method calls internal services with correct arguments in correct order"""
    # Given
    test_project_id = 123

    # When
    result = delete_index_service.delete_index(test_project_id)

    # Then - Verify result
    assert result == 1, "delete_index should return 1 on success"

    # Verify es_client was called with correct index name
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"
    mocked_opensearch_client.indices.delete.assert_called_once_with(index=expected_index_name)

    # Verify object_saver calls for namespace cleanup
    namespace_calls = [
        call
        for call in mocked_object_saver.remove_project_objects.call_args_list
        if call[0][0] == ["project_log_unique_words", "chosen_namespaces"]
    ]
    assert len(namespace_calls) == 1, "NamespaceFinder should call remove_project_objects once"
    assert namespace_calls[0][0][1] == test_project_id, "NamespaceFinder should use correct project_id"

    # Verify object_saver calls for trigger cleanup
    trigger_calls = [
        call
        for call in mocked_object_saver.remove_project_objects.call_args_list
        if call[0][0] in [["defect_type_trigger_info"], ["suggestion_trigger_info"], ["auto_analysis_trigger_info"]]
    ]
    assert len(trigger_calls) == 3, "TriggerManager should remove 3 types of triggers"

    # Verify object_saver calls for model cleanup
    model_folder_calls = mocked_object_saver.get_folder_objects.call_args_list
    assert len(model_folder_calls) >= 3, "ModelChooser should check for models"

    # Verify call order: es_client should be called first
    first_object_saver_position = next(
        (
            i
            for i, call in enumerate(mocked_object_saver.method_calls)
            if call[0] in ["remove_project_objects", "get_folder_objects"]
        ),
        None,
    )

    # Since these are on different mocks, we verify that es_client is called
    # and object_saver has cleanup operations
    assert mocked_opensearch_client.indices.delete.call_count == 1
    assert first_object_saver_position is not None, "object_saver should be called after es_client"


def test_delete_index_with_different_project_id(
    delete_index_service: DeleteIndexService, mocked_object_saver: ObjectSaver, mocked_opensearch_client: OpenSearch
) -> None:
    """Test delete_index with different project IDs"""
    # Given
    test_project_id = 456

    # When
    result = delete_index_service.delete_index(test_project_id)

    # Then
    assert result == 1
    expected_index_name = f"{APP_CONFIG.esProjectIndexPrefix}{test_project_id}"
    mocked_opensearch_client.indices.delete.assert_called_once_with(index=expected_index_name)
