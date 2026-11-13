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

from typing import Any

import pytest

from app.commons.model.db import Hit
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.similarity_filter import (
    SHORTENED_MESSAGE_FIELDS,
    UNSHORTENED_MESSAGE_FIELDS,
    SimilarityFilter,
    choose_message_fields,
)

REQUEST_MESSAGE = "AssertionError: expected status code 200 but was 500 for login request"
SIMILAR_MESSAGE = "AssertionError: expected status code 200 but was 500 for logout request"
UNRELATED_MESSAGE = "Skipped: no way of currently testing this SPECIALNUMBER"


def build_log(log_id: str, message: str, **kwargs: Any) -> LogData:
    fields = {field: message for field in SHORTENED_MESSAGE_FIELDS}
    fields.update(kwargs)
    return LogData(log_id=log_id, log_level=40000, **fields)


def build_request_item(*messages: str) -> TestItemIndexData:
    return TestItemIndexData(
        test_item_id="100",
        launch_id="1",
        logs=[build_log(str(idx), message) for idx, message in enumerate(messages)],
    )


def build_hit(test_item_id: str, inner_hits: dict[str, list[LogData]], **extra: Any) -> Hit[TestItemIndexData]:
    raw_inner_hits: dict[str, Any] = {
        name: {"hits": {"max_score": 1.0, "hits": [{"_id": log.log_id, "_source": log.model_dump()} for log in logs]}}
        for name, logs in inner_hits.items()
    }
    raw_inner_hits.update(extra)
    return Hit[TestItemIndexData].from_dict(
        {
            "_id": test_item_id,
            "_score": 1.0,
            "_source": {"test_item_id": test_item_id, "launch_id": "2", "issue_type": "pb001"},
            "inner_hits": raw_inner_hits,
        }
    )


def get_inner_log_ids(hit: Hit[TestItemIndexData], name: str) -> list[str]:
    return [raw_hit["_id"] for raw_hit in hit.inner_hits[name]["hits"]["hits"]]


def build_filter(min_similarity: float = 0.4, all_fields: bool = False) -> SimilarityFilter:
    return SimilarityFilter(list(SHORTENED_MESSAGE_FIELDS), min_similarity, all_fields=all_fields)


@pytest.mark.parametrize(
    "number_of_log_lines, expected_fields", [(-1, UNSHORTENED_MESSAGE_FIELDS), (2, SHORTENED_MESSAGE_FIELDS)]
)
def test_choose_message_fields(number_of_log_lines: int, expected_fields: list[str]):
    assert choose_message_fields(number_of_log_lines) == expected_fields


def test_unrelated_found_items_are_removed():
    request_item = build_request_item(REQUEST_MESSAGE)
    hits = [
        build_hit("1", {"log_0": [build_log("10", UNRELATED_MESSAGE)]}),
        build_hit("2", {"log_0": [build_log("20", SIMILAR_MESSAGE)]}),
        build_hit("3", {"log_0": [build_log("30", UNRELATED_MESSAGE)]}),
    ]

    filtered_request, filtered_hits = build_filter().filter((request_item, hits))

    assert filtered_request is request_item
    assert [hit.source.test_item_id for hit in filtered_hits] == ["2"]


def test_found_items_without_log_matches_are_removed():
    request_item = build_request_item(REQUEST_MESSAGE)
    hits = [build_hit("1", {}), build_hit("2", {"log_5": [build_log("20", REQUEST_MESSAGE)]})]

    _, filtered_hits = build_filter().filter((request_item, hits))

    assert filtered_hits == []


def test_only_similar_found_logs_are_left_in_inner_hits():
    request_item = build_request_item(REQUEST_MESSAGE, "Connection refused by the database host")
    other_inner_hits = {"hits": {"hits": [{"_id": "99"}]}}
    hit = build_hit(
        "1",
        {
            "log_0": [build_log("10", UNRELATED_MESSAGE), build_log("11", SIMILAR_MESSAGE)],
            "log_1": [build_log("12", UNRELATED_MESSAGE)],
        },
        other=other_inner_hits,
    )

    _, filtered_hits = build_filter().filter((request_item, [hit]))

    assert len(filtered_hits) == 1
    filtered_hit = filtered_hits[0]
    assert set(filtered_hit.inner_hits.keys()) == {"log_0", "other"}
    assert get_inner_log_ids(filtered_hit, "log_0") == ["11"]
    assert filtered_hit.inner_hits["log_0"]["hits"]["max_score"] == 1.0
    assert filtered_hit.inner_hits["other"] == other_inner_hits
    assert get_inner_log_ids(hit, "log_0") == ["10", "11"]
    assert get_inner_log_ids(hit, "log_1") == ["12"]


@pytest.mark.parametrize("all_fields, expected_ids", [(False, ["1"]), (True, [])])
def test_similarity_by_any_or_all_fields(all_fields: bool, expected_ids: list[str]):
    request_item = build_request_item(REQUEST_MESSAGE)
    found_log = build_log("10", REQUEST_MESSAGE, message_without_params_and_brackets=UNRELATED_MESSAGE)
    hits = [build_hit("1", {"log_0": [found_log]})]

    _, filtered_hits = build_filter(all_fields=all_fields).filter((request_item, hits))

    assert [hit.source.test_item_id for hit in filtered_hits] == expected_ids


def test_fields_empty_in_both_logs_are_not_compared():
    request_item = build_request_item(REQUEST_MESSAGE)
    request_item.logs[0].found_tests_and_methods = ""
    hits = [build_hit("1", {"log_0": [build_log("10", REQUEST_MESSAGE)]})]
    similarity_filter = SimilarityFilter(SHORTENED_MESSAGE_FIELDS + ["found_tests_and_methods"], 1.0, all_fields=True)

    _, filtered_hits = similarity_filter.filter((request_item, hits))

    assert [hit.source.test_item_id for hit in filtered_hits] == ["1"]


def test_field_empty_in_one_log_is_not_similar():
    request_item = build_request_item(REQUEST_MESSAGE)
    request_item.logs[0].found_tests_and_methods = "com.example.LoginTest.testLogin"
    hits = [build_hit("1", {"log_0": [build_log("10", REQUEST_MESSAGE)]})]
    similarity_filter = SimilarityFilter(SHORTENED_MESSAGE_FIELDS + ["found_tests_and_methods"], 0.8, all_fields=True)

    _, filtered_hits = similarity_filter.filter((request_item, hits))

    assert filtered_hits == []


def test_logs_with_nothing_to_compare_are_not_similar():
    similarity_filter = build_filter()

    assert not similarity_filter.is_similar(build_log("1", ""), build_log("2", ""))


@pytest.mark.parametrize(
    "required_log_indices, expected_ids",
    [(None, ["1", "2"]), ([], ["1", "2"]), ([0, 1], ["2"]), ([1], ["2"])],
)
def test_required_logs(required_log_indices: list[int] | None, expected_ids: list[str]):
    database_message = "Connection refused by the database host db01"
    request_item = build_request_item(REQUEST_MESSAGE, database_message)
    hits = [
        build_hit(
            "1",
            {"log_0": [build_log("10", SIMILAR_MESSAGE)], "log_1": [build_log("11", UNRELATED_MESSAGE)]},
        ),
        build_hit(
            "2",
            {"log_0": [build_log("20", SIMILAR_MESSAGE)], "log_1": [build_log("21", database_message)]},
        ),
    ]

    _, filtered_hits = build_filter().filter((request_item, hits), required_log_indices)

    assert [hit.source.test_item_id for hit in filtered_hits] == expected_ids


def test_threshold_is_inclusive():
    request_item = build_request_item(REQUEST_MESSAGE)
    hits = [build_hit("1", {"log_0": [build_log("10", REQUEST_MESSAGE)]})]

    _, filtered_hits = build_filter(min_similarity=1.0, all_fields=True).filter((request_item, hits))

    assert [hit.source.test_item_id for hit in filtered_hits] == ["1"]
