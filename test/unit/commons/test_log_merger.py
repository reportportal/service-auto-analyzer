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

from app.commons import log_merger
from app.commons.model.launch_objects import Launch, Log, TestItem
from app.commons.model.test_item_index import LogData, MergedLogData
from app.commons.request_factory import prepare_test_items

ERROR = 40000
FATAL = 50000


def build_log(log_id: str, log_level: int = ERROR, message_lines: int = 1, **kwargs) -> LogData:
    return LogData(log_id=log_id, log_level=log_level, message_lines=message_lines, **kwargs)


def get_ids(logs: list[LogData]) -> list[str]:
    return [log.log_id for log in logs]


@pytest.mark.parametrize(
    "message_lines, expected",
    [(None, False), (0, True), (1, True), (2, True), (3, False)],
)
def test_is_small_log(message_lines, expected):
    assert log_merger.is_small_log(build_log("1", message_lines=message_lines)) is expected


def test_merge_small_logs_of_one_level_into_one_group():
    logs = [build_log("1"), build_log("2"), build_log("3", message_lines=2)]

    big_logs, small_log_groups = log_merger.merge_small_logs(logs)

    assert big_logs == []
    assert [get_ids(group) for group in small_log_groups] == [["1", "2", "3"]]


def test_merge_small_logs_of_different_levels_into_separate_groups():
    logs = [build_log("1", FATAL), build_log("2", ERROR), build_log("3", FATAL)]

    big_logs, small_log_groups = log_merger.merge_small_logs(logs)

    assert big_logs == []
    assert [get_ids(group) for group in small_log_groups] == [["1", "3"], ["2"]]


def test_merge_small_logs_drops_small_logs_of_level_with_big_log():
    logs = [build_log("1"), build_log("2", message_lines=3), build_log("3")]

    big_logs, small_log_groups = log_merger.merge_small_logs(logs)

    assert get_ids(big_logs) == ["2"]
    assert small_log_groups == []


def test_merge_small_logs_keeps_small_logs_of_level_without_big_log():
    logs = [
        build_log("1", ERROR, message_lines=5),
        build_log("2", FATAL),
        build_log("3", ERROR),
        build_log("4", FATAL),
    ]

    big_logs, small_log_groups = log_merger.merge_small_logs(logs)

    assert get_ids(big_logs) == ["1"]
    assert [get_ids(group) for group in small_log_groups] == [["2", "4"]]


def test_merge_small_logs_keeps_all_big_logs():
    logs = [build_log("1", ERROR, message_lines=3), build_log("2", FATAL, message_lines=4)]

    big_logs, small_log_groups = log_merger.merge_small_logs(logs)

    assert get_ids(big_logs) == ["1", "2"]
    assert small_log_groups == []


@pytest.mark.parametrize(
    "message, expected_small",
    [
        ("Small error log", True),
        ("First line of the error\nSecond line of the error", True),
        ("Error: " + "something went wrong " * 9, True),
        ("Error: " + "something went wrong " * 10, False),
        ("First line of the error\nSecond line of the error\nThird line of the error", False),
        ("First line\nSecond line\nThird line", True),
        ("Error:\n1\n2\n3", True),
    ],
)
def test_log_size_is_calculated_with_line_length(message, expected_small):
    launch = Launch(
        launchId=1,
        project=1,
        testItems=[
            TestItem(
                testItemId=2,
                isAutoAnalyzed=False,
                issueType="ti001",
                logs=[Log(logId=3, logLevel=ERROR, message=message)],
            )
        ],
    )

    prepared_log = prepare_test_items(launch)[0].logs[0]

    assert log_merger.is_small_log(prepared_log) is expected_small


def test_merged_log_data_from_logs():
    logs = [
        build_log(
            "1",
            FATAL,
            whole_message="first message",
            found_exceptions="java.lang.IllegalStateException",
            potential_status_codes="401",
        ),
        build_log("2", FATAL, whole_message="second message", found_exceptions="", potential_status_codes=None),
        build_log(
            "3",
            FATAL,
            whole_message="third message",
            found_exceptions="java.lang.NullPointerException",
            potential_status_codes="500 400",
        ),
    ]

    merged = MergedLogData.from_logs(logs)

    assert merged.log_ids == ["1", "2", "3"]
    assert merged.log_level == FATAL
    assert merged.whole_message == "first message\nsecond message\nthird message"
    assert merged.found_exceptions == "java.lang.IllegalStateException java.lang.NullPointerException"
    assert merged.potential_status_codes == "401 500 400"
    assert merged.cluster_id == ""
    assert merged.cluster_message == ""
    assert merged.cluster_with_numbers is False
