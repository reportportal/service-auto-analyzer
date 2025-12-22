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
from test import DEFAULT_ES_CONFIG

TEST_PROJECT_ID = 2


def create_test_es_client():
    es_mock = mock.Mock()
    es_mock.search.return_value = {"hits": {"hits": []}}
    return esclient.EsClient(DEFAULT_ES_CONFIG, es_client=es_mock)


@pytest.mark.parametrize(
    "host, use_ssl, expected",
    [
        # 1. No protocol specified in the host param
        ("elastic_host", False, "http://elastic_host"),  # NOSONAR
        ("elastic_host", True, "https://elastic_host"),  # NOSONAR
        # 2. http protocol specified in the host parameter
        ("http://elastic_host", False, "http://elastic_host"),  # NOSONAR
        ("http://elastic_host", True, "http://elastic_host"),  # NOSONAR
        # 3. No protocol, but basic HTTP credentials are present in host parameter
        ("username:password@elastic_host", False, "http://username:password@elastic_host"),  # NOSONAR
        ("username:password@elastic_host", True, "https://username:password@elastic_host"),  # NOSONAR
        # 4. Protocol and credentials are present in host parameter -> same URL, no changes
        ("http://username:password@elastic_host", False, "http://username:password@elastic_host"),  # NOSONAR
        ("http://username:password@elastic_host", True, "http://username:password@elastic_host"),  # NOSONAR
    ],
)
def test_get_base_url_variations(host, use_ssl, expected):
    es_mock = mock.Mock()
    config = DEFAULT_ES_CONFIG.model_copy(
        update={
            "esHost": host,
            "esUseSsl": use_ssl,
        }
    )
    client = esclient.EsClient(config, es_client=es_mock)
    # Accessing name-mangled private method
    base_url = client._EsClient__get_base_url()
    assert base_url == expected
