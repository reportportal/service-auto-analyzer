#  Copyright 2023 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from unittest import mock

import pytest

from app.commons import esclient
from app.commons.model.launch_objects import Launch, Log, TestItem
from test import DEFAULT_ES_CONFIG

TEST_PROJECT_ID = 2


def create_test_es_client():
    es_mock = mock.Mock()
    es_mock.search.return_value = {"hits": {"hits": []}}
    return esclient.EsClient(DEFAULT_ES_CONFIG, es_mock)


def create_test_launch_one_item():
    logs = [Log(logId=37135, logLevel=40000, message="Environment variable 'SAUCELABS_USER' does not exist.")]
    test_items = [
        TestItem(
            testItemId=2190,
            isAutoAnalyzed=False,
            testCaseHash=-2120975783,
            testItemName="Example page test",
            logs=logs,
        )
    ]
    return Launch(launchId=10, project=TEST_PROJECT_ID, launchName="Test Launch", launchNumber=7, testItems=test_items)


def create_test_launch_two_items():
    launch = create_test_launch_one_item()
    test_items = launch.testItems
    logs = [Log(logId=37136, logLevel=40000, message="Environment variable 'SAUCELABS_USER' does not exist.")]
    test_items.append(
        TestItem(
            testItemId=2191,
            isAutoAnalyzed=False,
            testCaseHash=-2120975784,
            testItemName="Example page test",
            logs=logs,
        )
    )
    return launch


def create_test_launch_two_items_one_indexed_log():
    launch = create_test_launch_one_item()
    test_items = launch.testItems
    logs = [Log(logId=37136, logLevel=30000, message="Environment variable 'SAUCELABS_USER' does not exist.")]
    test_items.append(
        TestItem(
            testItemId=2191,
            isAutoAnalyzed=False,
            testCaseHash=-2120975784,
            testItemName="Example page test",
            logs=logs,
        )
    )
    return launch


@pytest.mark.parametrize(
    "launch, expected_launch",
    [
        (create_test_launch_one_item(), create_test_launch_one_item()),
        (create_test_launch_two_items(), create_test_launch_two_items()),
    ],
)
def test_to_launch_test_item_list(launch, expected_launch):
    es_client = create_test_es_client()
    items = es_client._to_launch_test_item_list([launch])
    assert len(items) == len(expected_launch.testItems)
    i = 0
    for actual_launch, item in items:
        assert len(actual_launch.testItems) == 0
        assert item.testItemId == expected_launch.testItems[i].testItemId
        i += 1


@pytest.mark.parametrize(
    "launch, expected_length",
    [
        (create_test_launch_one_item(), 1),
        (create_test_launch_two_items(), 2),
        (create_test_launch_two_items_one_indexed_log(), 1),
    ],
)
def test_to_index_bodies(launch, expected_length):
    es_client = create_test_es_client()
    test_item_ids, bodies = es_client._to_index_bodies(
        str(TEST_PROJECT_ID), es_client._to_launch_test_item_list([launch])
    )
    assert len(test_item_ids) == expected_length
    assert len(bodies) == expected_length


@pytest.mark.parametrize(
    "host, use_ssl, es_user, es_password, expected",
    [
        # 1. No protocol specified in the host param
        ("elastic_host", False, "", "", "http://elastic_host"),  # NOSONAR
        ("elastic_host", True, "", "", "https://elastic_host"),  # NOSONAR
        # 2. http protocol specified in the host parameter
        ("http://elastic_host", False, "", "", "http://elastic_host"),  # NOSONAR
        ("http://elastic_host", True, "", "", "http://elastic_host"),  # NOSONAR
        # 3. No protocol, but basic HTTP credentials are present in host parameter
        ("username:password@elastic_host", False, "", "", "http://username:password@elastic_host"),  # NOSONAR
        ("username:password@elastic_host", True, "", "", "https://username:password@elastic_host"),  # NOSONAR
        # 4. Protocol and credentials are present in host parameter -> same URL, no changes
        ("http://username:password@elastic_host", False, "", "", "http://username:password@elastic_host"),  # NOSONAR
        ("http://username:password@elastic_host", True, "", "", "http://username:password@elastic_host"),  # NOSONAR
        # 5. Protocol and credentials are present in host parameter, but also HTTP credentials are in app_config
        (
            "http://username:password@elastic_host",  # NOSONAR
            False,
            "user2",
            "pass2",
            "http://username:password@elastic_host",  # NOSONAR
        ),
        (
            "http://username:password@elastic_host",  # NOSONAR
            True,
            "user2",
            "pass2",
            "http://username:password@elastic_host",  # NOSONAR
        ),
        # 6. No protocol, no credentials in host, but HTTP credentials are specified in app_config
        ("elastic_host", False, "username", "password", "http://username:password@elastic_host"),  # NOSONAR
        ("elastic_host", True, "username", "password", "https://username:password@elastic_host"),  # NOSONAR
    ],
)
def test_get_base_url_variations(host, use_ssl, es_user, es_password, expected):
    es_mock = mock.Mock()
    config = DEFAULT_ES_CONFIG.copy(
        update={
            "esHost": host,
            "esUseSsl": use_ssl,
            "esUser": es_user,
            "esPassword": es_password,
        }
    )
    client = esclient.EsClient(config, es_mock)
    # Accessing name-mangled private method
    base_url = client._EsClient__get_base_url()
    assert base_url == expected
