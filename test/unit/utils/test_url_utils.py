#  Copyright 2026 EPAM Systems
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

import pytest

from app.utils import url_utils


@pytest.mark.parametrize(
    "url, expected_url",
    [
        (
            "amqp://user:password@10.68.56.88:5672/analyzer?heartbeat=30",  # NOSONAR
            "amqp://10.68.56.88:5672/analyzer?heartbeat=30",  # NOSONAR
        ),
        ("amqps://rpuser:fkkf0+4pUn@192.68.56.88:5672", "amqps://192.68.56.88:5672"),  # NOSONAR
        ("https://test123:aa-bb_cc@msgbroker.example.com/", "https://msgbroker.example.com/"),  # NOSONAR
        ("https://test123:aa%20bb%40cc@msgbroker.example.com/", "https://msgbroker.example.com/"),  # NOSONAR
    ],
)
def test_remove_credentials_from_url(url, expected_url):
    assert url_utils.remove_credentials_from_url(url) == expected_url
