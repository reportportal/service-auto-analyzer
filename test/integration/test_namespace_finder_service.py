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

from app.commons.model.launch_objects import Launch
from app.commons.namespace_finder import CHOSEN_NAMESPACES_OBJECT, UNIQUE_WORDS_OBJECT, NamespaceFinder
from app.commons.object_saving import ObjectSaver
from app.service.namespace_finder_service import NamespaceFinderService
from app.utils.utils import read_json_file
from test import APP_CONFIG


@pytest.fixture
def test_launches() -> list[Launch]:
    """Load test data with launches containing stacktraces."""
    data = read_json_file("test_res", "namespace_finder_test_data.json", to_json=True)
    return [Launch(**launch_data) for launch_data in data["launches"]]


@pytest.fixture
def mocked_object_saver() -> ObjectSaver:
    """Create a mocked ObjectSaver instance."""
    mock_saver = mock.Mock(ObjectSaver)
    # Configure does_object_exists to return False initially (no existing words)
    mock_saver.does_object_exists.return_value = False
    # Configure get_project_object to return empty dict when no existing data
    mock_saver.get_project_object.return_value = {}
    return mock_saver


@pytest.fixture
def namespace_finder_service(mocked_object_saver: ObjectSaver) -> NamespaceFinderService:
    """Create NamespaceFinderService with real NamespaceFinder and mocked object_saver."""
    # Create real NamespaceFinder with mocked object_saver
    namespace_finder = NamespaceFinder(APP_CONFIG, object_saver=mocked_object_saver)

    # Create NamespaceFinderService with the namespace_finder
    service = NamespaceFinderService(APP_CONFIG, namespace_finder=namespace_finder)

    return service


# noinspection PyUnresolvedReferences
def test_update_chosen_namespaces_calls_correct_services(
    namespace_finder_service: NamespaceFinderService, mocked_object_saver: ObjectSaver, test_launches: list[Launch]
) -> None:
    """Test that update_chosen_namespaces method calls internal services with correct arguments."""
    # Expected project ID from test data
    expected_project_id = 123

    # Execute the method
    namespace_finder_service.update_chosen_namespaces(test_launches)

    # Verify the call sequence to object_saver
    mocked_object_saver.does_object_exists.assert_called_once()

    # Verify does_object_exists was called with correct parameters
    does_exist_call = mocked_object_saver.does_object_exists.call_args_list[0]
    assert does_exist_call[0][0] == UNIQUE_WORDS_OBJECT, f"does_object_exists should check for '{UNIQUE_WORDS_OBJECT}'"
    assert (
        does_exist_call[0][1] == expected_project_id
    ), f"does_object_exists should use project_id={expected_project_id}"

    # Verify put_project_object was called twice (once for words, once for namespaces)
    assert (
        mocked_object_saver.put_project_object.call_count == 2
    ), "object_saver.put_project_object should be called twice: once for words, once for namespaces"

    # Verify first put_project_object call (saving unique words)
    first_put_call = mocked_object_saver.put_project_object.call_args_list[0]
    saved_words = first_put_call[0][0]
    assert isinstance(saved_words, dict), "First put_project_object should save a dictionary of words"
    assert first_put_call[0][1] == UNIQUE_WORDS_OBJECT, f"First put_project_object should save '{UNIQUE_WORDS_OBJECT}'"
    assert (
        first_put_call[0][2] == expected_project_id
    ), f"First put_project_object should use project_id={expected_project_id}"
    assert first_put_call[1]["using_json"] is True, "First put_project_object should use JSON serialization"

    # Verify saved words contain expected stacktrace components
    assert any(
        "com.example.service" in word for word in saved_words
    ), "Saved words should contain stacktrace packages like 'com.example.service'"
    assert any(
        "org.hibernate" in word or "org.springframework" in word for word in saved_words
    ), "Saved words should contain library packages"

    # Verify second put_project_object call (saving chosen namespaces)
    second_put_call = mocked_object_saver.put_project_object.call_args_list[1]
    saved_namespaces = second_put_call[0][0]
    assert isinstance(saved_namespaces, dict), "Second put_project_object should save a dictionary of namespaces"
    assert (
        second_put_call[0][1] == CHOSEN_NAMESPACES_OBJECT
    ), f"Second put_project_object should save '{CHOSEN_NAMESPACES_OBJECT}'"
    assert (
        second_put_call[0][2] == expected_project_id
    ), f"Second put_project_object should use project_id={expected_project_id}"
    assert second_put_call[1]["using_json"] is True, "Second put_project_object should use JSON serialization"


# noinspection PyUnresolvedReferences
def test_update_chosen_namespaces_with_existing_words(test_launches: list[Launch]) -> None:
    """Test update_chosen_namespaces when project already has existing unique words."""
    expected_project_id = 123

    # Create a fresh mock for this test
    mock_saver = mock.Mock(ObjectSaver)

    # Configure mock to return True for existing words
    existing_words = {
        "org.junit.runner.JUnitCore.run": 1,
        "org.junit.runner.Runner.run": 1,
        "org.junit.framework.TestCase.run": 1,
    }
    mock_saver.does_object_exists.return_value = True
    mock_saver.get_project_object.return_value = existing_words

    # Create namespace_finder and service
    namespace_finder = NamespaceFinder(APP_CONFIG, object_saver=mock_saver)
    service = NamespaceFinderService(APP_CONFIG, namespace_finder=namespace_finder)

    # Execute the method
    service.update_chosen_namespaces(test_launches)

    # Verify get_project_object was called to load existing words
    mock_saver.get_project_object.assert_called_once()

    get_call = mock_saver.get_project_object.call_args_list[0]
    assert get_call[0][0] == UNIQUE_WORDS_OBJECT, f"get_project_object should load '{UNIQUE_WORDS_OBJECT}'"
    assert get_call[0][1] == expected_project_id, f"get_project_object should use project_id={expected_project_id}"
    assert get_call[1]["using_json"] is True, "get_project_object should use JSON deserialization"

    # Verify that saved words include both existing and new words
    first_put_call = mock_saver.put_project_object.call_args_list[0]
    saved_words = first_put_call[0][0]

    # Check that existing words are preserved
    for existing_word in existing_words:
        assert existing_word in saved_words, f"Existing word '{existing_word}' should be preserved"

    # Check that new words from test launches are added
    assert any("com.example.service" in word for word in saved_words), "New words from test launches should be added"


# noinspection PyUnresolvedReferences
def test_update_chosen_namespaces_with_multiple_projects(test_launches: list[Launch]) -> None:
    """Test that update_chosen_namespaces only processes one project at a time."""
    # Create a fresh mock for this test
    mock_saver = mock.Mock(ObjectSaver)
    mock_saver.does_object_exists.return_value = False
    mock_saver.get_project_object.return_value = {}

    # Create launches with different project IDs
    launch1 = test_launches[0]
    launch2 = Launch(**test_launches[1].dict())
    launch2.project = 456  # Different project ID

    # Create namespace_finder and service
    namespace_finder = NamespaceFinder(APP_CONFIG, object_saver=mock_saver)
    service = NamespaceFinderService(APP_CONFIG, namespace_finder=namespace_finder)

    # Execute with launches from different projects
    service.update_chosen_namespaces([launch1, launch2])

    # Verify that only one project was processed (the last one in the list)
    # This is based on the implementation where prepare_log_words returns the last project
    mock_saver.does_object_exists.assert_called_once()
    does_exist_call = mock_saver.does_object_exists.call_args_list[0]
    assert does_exist_call[0][1] == 456, "Should use the project ID from the last launch"


# noinspection PyUnresolvedReferences
def test_update_chosen_namespaces_with_empty_launches(
    namespace_finder_service: NamespaceFinderService, mocked_object_saver: ObjectSaver
) -> None:
    """Test update_chosen_namespaces with empty launches list."""
    # Execute with empty list
    namespace_finder_service.update_chosen_namespaces([])

    # Verify that no object_saver methods were called
    mocked_object_saver.does_object_exists.assert_not_called()
    mocked_object_saver.get_project_object.assert_not_called()
    mocked_object_saver.put_project_object.assert_not_called()
