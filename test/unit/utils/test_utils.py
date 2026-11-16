#  Copyright 2026 EPAM Systems
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

import pytest

from app.utils.utils import parse_string_map, put_if_set


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, {}),
        ("", {}),
        ("  ", {}),
        ("{}", {}),
        ('{"endpoint": "http://localhost:9000"}', {"endpoint": "http://localhost:9000"}),
        ('{"enable_virtual_host_style": true, "retries": 3}', {"enable_virtual_host_style": "true", "retries": "3"}),
    ],
)
def test_parse_string_map(value, expected):
    assert parse_string_map(value) == expected


@pytest.mark.parametrize("value", ["not a json", '["a", "b"]', '"string"', "{"])
def test_parse_string_map_invalid(value):
    with pytest.raises(ValueError):
        parse_string_map(value)


@pytest.mark.parametrize("value", [None, "", 0, [], {}])
def test_put_if_set_skips_unset_value(value):
    dictionary = {"existing": "value"}

    put_if_set(dictionary, "key", value)

    assert dictionary == {"existing": "value"}


@pytest.mark.parametrize("value", ["value", 1, ["item"]])
def test_put_if_set_puts_set_value(value):
    dictionary = {"existing": "value"}

    put_if_set(dictionary, "key", value)

    assert dictionary == {"existing": "value", "key": value}


def test_put_if_set_overrides_existing_value():
    dictionary = {"key": "old"}

    put_if_set(dictionary, "key", "new")

    assert dictionary == {"key": "new"}
