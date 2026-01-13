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

from app.commons.model.launch_objects import (
    DeleteLaunchesRequest,
    DeleteLogsRequest,
    DeleteTestItemsRequest,
    RemoveByDatesRequest,
    SearchConfig,
)
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.object_saving import ObjectSaver
from app.commons.os_client import OsClient
from app.commons.trigger_manager import TriggerManager
from app.service.clean_index_service import CleanIndexService
from test import APP_CONFIG, DEFAULT_BOOST_LAUNCH


def get_folder_objects(folder: str, __: str | int | None = None) -> list[str]:
    model_name = folder.removesuffix("_model/")
    return [f"{model_name}_1"]


@pytest.fixture
def mocked_os_client() -> OsClient:
    """Create a mocked OsClient instance"""
    client = mock.Mock(spec=OsClient)
    client.delete_logs_by_ids.return_value = 5
    client.delete_test_items.return_value = 5
    client.delete_by_launch_ids.return_value = 5
    client.delete_by_launch_start_time_range.return_value = 5
    client.delete_by_log_time_range.return_value = 5
    client.delete_index.return_value = True
    return client


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
    mock_saver.get_folder_objects.side_effect = get_folder_objects
    mock_saver.remove_folder_objects.return_value = 1
    return mock_saver


@pytest.fixture
def model_chooser(search_config: SearchConfig, mocked_object_saver: ObjectSaver) -> ModelChooser:
    return ModelChooser(APP_CONFIG, search_config, object_saver=mocked_object_saver)


@pytest.fixture
def clean_index_service(
    model_chooser: ModelChooser, search_config: SearchConfig, mocked_os_client: OsClient
) -> CleanIndexService:
    """Create CleanIndexService with mocked OsClient"""
    return CleanIndexService(model_chooser, APP_CONFIG, search_config, os_client=mocked_os_client)


@pytest.fixture
def delete_clean_index_service(
    mocked_object_saver: ObjectSaver,
    mocked_os_client: OsClient,
    search_config: SearchConfig,
    model_chooser: ModelChooser,
) -> CleanIndexService:
    """Create CleanIndexService configured for delete_index testing with mocked dependencies"""
    namespace_finder = NamespaceFinder(APP_CONFIG, object_saver=mocked_object_saver)
    trigger_manager = TriggerManager(
        model_chooser,
        app_config=APP_CONFIG,
        search_cfg=search_config,
        object_saver=mocked_object_saver,
    )

    return CleanIndexService(
        model_chooser,
        APP_CONFIG,
        search_cfg=search_config,
        os_client=mocked_os_client,
        namespace_finder=namespace_finder,
        trigger_manager=trigger_manager,
    )


def test_delete_logs_calls_correct_services(
    clean_index_service: CleanIndexService,
    mocked_os_client: OsClient,
) -> None:
    """Test that delete_logs method calls OsClient with normalized IDs."""
    test_project_id = 123
    test_log_ids = [1, 2, 3, 4, 5]
    clean_index = DeleteLogsRequest(ids=test_log_ids, project=test_project_id)

    result = clean_index_service.delete_logs(clean_index)

    assert result == 5, "delete_logs should return the success count from OsClient"
    mocked_os_client.delete_logs_by_ids.assert_called_once_with(
        test_project_id, [str(log_id) for log_id in test_log_ids]
    )


# noinspection PyUnresolvedReferences
def test_delete_test_items_calls_correct_services(
    clean_index_service: CleanIndexService,
    mocked_os_client: OsClient,
) -> None:
    """Test that delete_test_items method calls internal services with correct arguments."""
    test_project_id = 456
    test_item_ids = [10, 20, 30]
    remove_items_info = DeleteTestItemsRequest(project=test_project_id, itemsToDelete=test_item_ids)

    result = clean_index_service.delete_test_items(remove_items_info)

    assert result == 5, "delete_test_items should return the number of deleted logs"
    mocked_os_client.delete_test_items.assert_called_once_with(
        test_project_id, [str(test_item_id) for test_item_id in test_item_ids]
    )


# noinspection PyUnresolvedReferences
def test_delete_launches_calls_correct_services(
    clean_index_service: CleanIndexService,
    mocked_os_client: OsClient,
) -> None:
    """Test that delete_launches method calls internal services with correct arguments."""
    test_project_id = 789
    test_launch_ids = [100, 200]
    launch_remove_info = DeleteLaunchesRequest(project=test_project_id, launch_ids=test_launch_ids)

    result = clean_index_service.delete_launches(launch_remove_info)

    assert result == 5, "delete_launches should return the number of deleted logs"
    mocked_os_client.delete_by_launch_ids.assert_called_once_with(
        test_project_id, [str(launch_id) for launch_id in test_launch_ids]
    )


# noinspection PyUnresolvedReferences
def test_remove_by_launch_start_time_calls_correct_services(
    clean_index_service: CleanIndexService,
    mocked_os_client: OsClient,
) -> None:
    """Test that remove_by_launch_start_time method calls internal services in correct order."""
    test_project_id = 321
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    remove_by_launch_start_time_info = RemoveByDatesRequest(
        project=test_project_id,
        interval_start_date=start_date,
        interval_end_date=end_date,
    )

    result = clean_index_service.remove_by_launch_start_time(remove_by_launch_start_time_info)

    assert result == 5, "remove_by_launch_start_time should return the number of deleted logs"
    mocked_os_client.delete_by_launch_start_time_range.assert_called_once_with(test_project_id, start_date, end_date)


# noinspection PyUnresolvedReferences
def test_remove_by_log_time_calls_correct_services(
    clean_index_service: CleanIndexService,
    mocked_os_client: OsClient,
) -> None:
    """Test that remove_by_log_time method calls internal services in correct order."""
    test_project_id = 654
    start_date = "2023-06-01"
    end_date = "2023-06-30"
    remove_by_log_time_info = RemoveByDatesRequest(
        project=test_project_id,
        interval_start_date=start_date,
        interval_end_date=end_date,
    )

    result = clean_index_service.remove_by_log_time(remove_by_log_time_info)

    assert result == 5, "remove_by_log_time should return the number of deleted logs"
    mocked_os_client.delete_by_log_time_range.assert_called_once_with(test_project_id, start_date, end_date)


def test_delete_logs_with_empty_list(
    clean_index_service: CleanIndexService,
    mocked_os_client: OsClient,
) -> None:
    """Test delete_logs with empty log IDs list"""
    mocked_os_client.delete_logs_by_ids.return_value = 0

    test_project_id = 999
    clean_index = DeleteLogsRequest(ids=[], project=test_project_id)

    result = clean_index_service.delete_logs(clean_index)

    assert result == 0
    mocked_os_client.delete_logs_by_ids.assert_called_once_with(test_project_id, [])


def test_delete_test_items_with_different_project(
    clean_index_service: CleanIndexService,
    mocked_os_client: OsClient,
) -> None:
    """Test delete_test_items with different project ID"""
    test_project_id = 111
    remove_items_info = DeleteTestItemsRequest(project=test_project_id, itemsToDelete=[1, 2, 3])

    result = clean_index_service.delete_test_items(remove_items_info)

    assert result == 5
    mocked_os_client.delete_test_items.assert_called_once_with(test_project_id, ["1", "2", "3"])


def test_delete_index_calls_correct_services(
    delete_clean_index_service: CleanIndexService, mocked_object_saver: ObjectSaver, mocked_os_client: OsClient
) -> None:
    """Test that delete_index method calls internal services with correct arguments."""
    test_project_id = 123

    result = delete_clean_index_service.delete_index(test_project_id)

    assert result == 1, "delete_index should return 1 on success"
    mocked_os_client.delete_index.assert_called_once_with(test_project_id)

    remove_call_args = mocked_object_saver.remove_project_objects.call_args_list

    namespace_calls = [
        call for call in remove_call_args if call[0][0] == ["project_log_unique_words", "chosen_namespaces"]
    ]
    assert len(namespace_calls) == 1, "NamespaceFinder should call remove_project_objects once"
    assert namespace_calls[0][0][1] == test_project_id, "NamespaceFinder should use correct project_id"

    trigger_calls = {
        call[0][0][0]
        for call in remove_call_args
        if call[0][0] in [["defect_type_trigger_info"], ["suggestion_trigger_info"], ["auto_analysis_trigger_info"]]
    }
    assert len(trigger_calls) == 3, "TriggerManager should remove 3 types of triggers"

    model_folder_calls = {
        call[0][0]
        for call in mocked_object_saver.get_folder_objects.call_args_list
        if call[0][0] in {"auto_analysis_model/", "suggestion_model/", "defect_type_model/"}
    }
    assert len(model_folder_calls) == 3, "ModelChooser should check for models"

    remove_folder_objects_calls = {
        call[0][0]
        for call in mocked_object_saver.remove_folder_objects.call_args_list
        if call[0][0] in {"auto_analysis_1", "suggestion_1", "defect_type_1"}
    }
    assert len(remove_folder_objects_calls) == 3, "ModelChooser should remove custom models"


def test_delete_index_with_different_project_id(
    delete_clean_index_service: CleanIndexService, mocked_os_client: OsClient
) -> None:
    """Test delete_index with different project IDs"""
    test_project_id = 456
    result = delete_clean_index_service.delete_index(test_project_id)

    assert result == 1
    mocked_os_client.delete_index.assert_called_once_with(test_project_id)
