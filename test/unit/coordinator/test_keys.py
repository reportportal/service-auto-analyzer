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

import json

import pytest

from app.coordinator import keys


def _raw(body) -> bytes:
    return json.dumps(body).encode("utf-8")


@pytest.mark.parametrize(
    ("routing_key", "body", "expected"),
    [
        ("index", [{"launchId": 1, "project": 3}], "3"),
        ("analyze", [{"launchId": 1, "project": "3"}], "3"),
        ("namespace_finder", [{"launchId": 1, "project": 3}], "3"),
        ("defect_update", {"project": 4, "itemsToUpdate": {}}, "4"),
        ("delete", 5, "5"),
        ("suggest_patterns", "5", "5"),
        ("clean", {"project": 6, "ids": []}, "6"),
        ("item_remove", {"project": 7, "itemsToDelete": []}, "7"),
        ("launch_remove", {"project": "7", "launch_ids": []}, "7"),
        ("remove_by_launch_start_time", {"project": 8}, "8"),
        ("remove_by_log_time", {"project": 8}, "8"),
        ("search", {"projectId": 9, "launchId": 1}, "9"),
        ("suggest", {"project": 10, "testItemId": 1}, "10"),
        ("cluster", {"project": 11, "launch": {}}, "11"),
        ("train_models", {"project": 12, "model_type": "suggestion"}, "12"),
    ],
)
def test_lane_key_per_route(routing_key, body, expected):
    assert keys.lane_key(routing_key, body) == expected


@pytest.mark.parametrize(
    ("routing_key", "body"),
    [
        ("index_suggest_info", [{"project": 1}]),
        ("remove_suggest_info", 1),
        ("update_suggest_info", {"project": 1}),
        ("remove_models", {"project": 1}),
        ("get_model_info", {"project": 1}),
        ("noop_echo", {"project": 1}),
        ("unknown_route", {"project": 1}),
    ],
)
def test_lane_key_none_for_routes_without_ordering(routing_key, body):
    assert keys.lane_key(routing_key, body) is None


@pytest.mark.parametrize(
    ("routing_key", "body"),
    [
        ("index", []),
        ("index", [{"launchId": 1}]),
        ("search", {"project": 1}),
        ("suggest", {"projectId": 1}),
        ("delete", None),
        ("delete", {"project": 1}),
        ("clean", [1, 2]),
        ("clean", {"project": ""}),
    ],
)
def test_lane_key_none_without_project(routing_key, body):
    assert keys.lane_key(routing_key, body) is None


def test_int_and_str_project_share_lane():
    assert keys.lane_key("delete", 15) == keys.lane_key("delete", "15")


def test_split_single_project_keeps_raw_body():
    raw = b'[{"launchId": 1, "project": 1}, {"launchId": 2, "project": "1"}]'

    parts = keys.split_by_project("index", raw)

    assert parts == [("1", raw)]
    assert parts[0][1] is raw


def test_split_several_projects_preserves_order():
    body = [
        {"launchId": 1, "project": 1},
        {"launchId": 2, "project": 2},
        {"launchId": 3, "project": 1},
        {"launchId": 4, "project": 3},
    ]

    parts = keys.split_by_project("analyze", _raw(body))

    assert [lane for lane, _ in parts] == ["1", "2", "3"]
    assert [json.loads(raw) for _, raw in parts] == [
        [{"launchId": 1, "project": 1}, {"launchId": 3, "project": 1}],
        [{"launchId": 2, "project": 2}],
        [{"launchId": 4, "project": 3}],
    ]


def test_split_launch_without_project_goes_to_free_part():
    body = [{"launchId": 1, "project": 1}, {"launchId": 2}]

    parts = keys.split_by_project("namespace_finder", _raw(body))

    assert [lane for lane, _ in parts] == ["1", None]
    assert json.loads(parts[1][1]) == [{"launchId": 2}]


@pytest.mark.parametrize(
    ("routing_key", "body", "expected_lane"),
    [
        ("suggest", {"project": 1}, "1"),
        ("delete", 2, "2"),
        ("noop_echo", "value", None),
        ("index", {"not": "a list"}, None),
    ],
)
def test_split_other_routes_one_part(routing_key, body, expected_lane):
    raw = _raw(body)

    assert keys.split_by_project(routing_key, raw) == [(expected_lane, raw)]


def test_split_malformed_json_raises():
    with pytest.raises(json.JSONDecodeError):
        keys.split_by_project("suggest", b"{not json")
