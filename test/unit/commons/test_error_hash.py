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

"""Unit tests for error_hash module."""

from typing import Optional

import pytest

from app.commons import error_hash
from app.commons.error_hash import ERROR_HASH_MAX_LINES, compute_error_hash
from app.commons.model.launch_objects import AnalyzerConf, Launch, Log, TestItem
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.request_factory import prepare_test_items


def build_log(
    detected_message: str = "Unable to create user",
    found_exceptions: str = "",
    potential_status_codes: str = "",
    message: Optional[str] = None,
    log_id: str = "1",
) -> LogData:
    return LogData(
        log_id=log_id,
        log_level=40000,
        message=message,
        detected_message=detected_message,
        found_exceptions=found_exceptions,
        potential_status_codes=potential_status_codes,
    )


def prepare_item(
    message: str, issue_type: str = "ti001", is_auto_analyzed: bool = False, launch_id: int = 1
) -> TestItemIndexData:
    launch = Launch(
        launchId=launch_id,
        project=1,
        launchName="Launch",
        analyzerConfig=AnalyzerConf(numberOfLogLines=-1),
        testItems=[
            TestItem(
                testItemId=launch_id * 10,
                isAutoAnalyzed=is_auto_analyzed,
                issueType=issue_type,
                testCaseHash=7,
                logs=[Log(logId=launch_id * 100, logLevel=40000, message=message)],
            )
        ],
    )
    return prepare_test_items(launch)[0]


def test_hash_is_deterministic() -> None:
    logs = [build_log(found_exceptions="java.lang.AssertionError", potential_status_codes="201 500")]

    first = compute_error_hash(logs)

    assert first is not None
    assert compute_error_hash(logs) == first


def test_hash_is_none_without_message_lines_and_exceptions() -> None:
    assert compute_error_hash([]) is None
    assert compute_error_hash([build_log(detected_message=" \n ", potential_status_codes="500")]) is None


def test_hash_is_not_none_with_exceptions_only() -> None:
    assert compute_error_hash([build_log(detected_message="", found_exceptions="java.lang.IllegalStateException")])


def test_status_code_order_matters() -> None:
    assert compute_error_hash([build_log(potential_status_codes="400 401")]) != compute_error_hash(
        [build_log(potential_status_codes="401 400")]
    )


def test_log_without_status_codes_differs_from_log_with_codes() -> None:
    assert compute_error_hash([build_log()]) != compute_error_hash([build_log(potential_status_codes="500")])


def test_status_codes_are_compared_per_log() -> None:
    codes_in_first_log = [build_log(potential_status_codes="500"), build_log(log_id="2")]
    codes_in_second_log = [build_log(), build_log(log_id="2", potential_status_codes="500")]

    assert compute_error_hash(codes_in_first_log) != compute_error_hash(codes_in_second_log)


def test_different_exceptions_differ() -> None:
    assert compute_error_hash([build_log(found_exceptions="java.lang.NullPointerException")]) != compute_error_hash(
        [build_log(found_exceptions="java.lang.IllegalStateException")]
    )


def test_different_messages_differ() -> None:
    assert compute_error_hash([build_log("Unable to create user")]) != compute_error_hash(
        [build_log("Unable to delete user")]
    )


def test_whitespace_is_normalized() -> None:
    assert compute_error_hash([build_log("Unable  to create\tuser ")]) == compute_error_hash(
        [build_log("Unable to create user")]
    )


def test_message_is_used_when_detected_message_is_empty() -> None:
    assert compute_error_hash([build_log(detected_message="", message="Unable to create user")]) == compute_error_hash(
        [build_log()]
    )


def test_lines_after_the_limit_are_ignored() -> None:
    lines = [f"line {chr(ord('a') + i % 26)}{i // 26}" for i in range(ERROR_HASH_MAX_LINES)]

    assert compute_error_hash([build_log("\n".join(lines + ["extra line"]))]) == compute_error_hash(
        [build_log("\n".join(lines))]
    )


def test_hash_depends_on_version(monkeypatch: pytest.MonkeyPatch) -> None:
    logs = [build_log()]
    current = compute_error_hash(logs)

    monkeypatch.setattr(error_hash, "ERROR_HASH_VERSION", "v0")

    assert compute_error_hash(logs) != current


def test_prepared_items_differing_only_in_numbers_share_hash() -> None:
    first = prepare_item("Order 81723 for user alice failed: timeout after 3000 ms")
    second = prepare_item("Order 99120 for user alice failed: timeout after 5000 ms")

    assert first.error_hash is not None
    assert first.error_hash == second.error_hash


def test_prepared_items_with_swapped_status_codes_differ() -> None:
    first = prepare_item("expected response to have status code 200 but got 400")
    second = prepare_item("expected response to have status code 400 but got 200")

    assert first.logs is not None and second.logs is not None
    assert first.logs[0].detected_message == second.logs[0].detected_message
    assert first.error_hash != second.error_hash


def test_index_time_and_analysis_time_preparation_give_the_same_hash() -> None:
    message = (
        "java.lang.AssertionError: 1 expectation failed.\nExpected status code <201> but was <500>.\n"
        "\tat com.example.UserTest.create(UserTest.java:42)"
    )
    indexed = prepare_item(message, issue_type="pb001", is_auto_analyzed=True, launch_id=1)
    analyzed = prepare_item(message, issue_type="ti001", is_auto_analyzed=False, launch_id=2)

    assert indexed.error_hash is not None
    assert indexed.error_hash == analyzed.error_hash
    assert "error_hash" in indexed.to_index_dict()
