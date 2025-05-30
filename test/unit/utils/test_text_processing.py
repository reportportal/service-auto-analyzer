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

from app.utils import text_processing, utils
from test import read_file, read_file_lines


def test_delete_empty_lines():
    log = utils.read_file("test_res/test_logs", "reportportal-api.txt")
    expected = utils.read_file("test_res/test_logs", "reportportal-api-no-empty-lines.txt")

    assert text_processing.delete_empty_lines(log) == expected.rstrip("\n")


def test_filter_empty_lines():
    log = read_file_lines("test_res/test_logs", "reportportal-api.txt")
    expected = read_file_lines("test_res/test_logs", "reportportal-api-no-empty-lines.txt")

    assert text_processing.filter_empty_lines(log) == expected


def test_remove_starting_datetime():
    log = read_file_lines("test_res/test_logs", "log_line_timestamps.txt")
    expected_log = read_file_lines("test_res/test_logs", "log_line_no_timestamp.txt")
    for i, line in enumerate(log):
        assert text_processing.remove_starting_datetime(line) == expected_log[i]


def test_remove_starting_log_level():
    log = read_file_lines("test_res/test_logs", "log_line_no_timestamp.txt")
    expected_log = read_file_lines("test_res/test_logs", "log_line_no_log_level.txt")
    for i, line in enumerate(log):
        assert text_processing.remove_starting_log_level(line) == expected_log[i]


def test_remove_starting_thread_id():
    log = read_file_lines("test_res/test_logs", "log_line_no_log_level.txt")
    expected_log = read_file_lines("test_res/test_logs", "log_line_no_thread_id.txt")
    for i, line in enumerate(log):
        assert text_processing.remove_starting_thread_id(line) == expected_log[i]


def test_remove_starting_thread_namer():
    log = read_file_lines("test_res/test_logs", "log_line_no_thread_id.txt")
    expected_log = read_file_lines("test_res/test_logs", "log_line_no_thread_name.txt")
    for i, line in enumerate(log):
        assert text_processing.remove_starting_thread_name(line) == expected_log[i]


@pytest.mark.parametrize(
    "test_file, expected_file",
    [
        ("stacktraces/log_stacktrace_generated.txt", "stacktraces/log_stacktrace_prepared.txt"),
        ("stacktraces/log_stacktrace_generated_2.txt", "stacktraces/log_stacktrace_prepared_2.txt"),
        ("stacktraces/log_stacktrace_generated_3.txt", "stacktraces/log_stacktrace_prepared_3.txt"),
        ("log_locator_with_attribute.txt", "log_locator_with_attribute_prepared.txt"),
    ],
)
def test_remove_generated_parts(test_file, expected_file):
    log = read_file("test_res/test_logs", test_file)
    expected_log = read_file("test_res/test_logs", expected_file)
    assert text_processing.remove_generated_parts(log) == expected_log


def test_clean_from_brackets():
    log = read_file_lines("test_res/test_logs", "brackets_test.txt")
    expected_log = read_file_lines("test_res/test_logs", "brackets_test_results.txt")
    for i, line in enumerate(log):
        assert text_processing.clean_from_brackets(line) == expected_log[i]


@pytest.mark.parametrize(
    "message, expected_message",
    [
        ("\t \r\n ", "\n "),
        ("\r\n", "\n"),
        ("\n", "\n"),
        ("\u00a0\u00a0\u00a0\n", "\n"),
        ("\u00a0\r\n", "\n"),
    ],
)
def test_unify_line_endings(message, expected_message):
    assert text_processing.unify_line_endings(message) == expected_message


@pytest.mark.parametrize(
    "message, expected_message",
    [
        ("\t \r\n ", " \r\n"),
        ("\r\n", "\r\n"),
        ("\n", "\n"),
        ("\u00a0\u00a0\u00a0\n", "\n"),
        ("\u00a0\r\n", " \r\n"),
        ("\u00a0\u2000\u2001", " "),
        ("\u202f\u205f\u3000", " "),
        ("a\u202f\u205f\u3000b", "a b"),
        ("\u00a0\u00a0\u00a0\n\u00a0\u00a0\u00a0", "\n"),
    ],
)
def test_unify_spaces(message, expected_message):
    assert text_processing.unify_spaces(message) == expected_message


def test_remove_markdown_mode():
    log = read_file("test_res/test_logs/markdown", "markdown_at_log.txt")
    expected_log = read_file("test_res/test_logs/markdown", "markdown_at_log_prepared.txt")
    assert text_processing.remove_markdown_mode(log) == expected_log


@pytest.mark.parametrize(
    "test_file, expected_file",
    [
        ("separators/markdown_separator_log.txt", "separators/markdown_separator_log_prepared.txt"),
        ("separators/step_separator_log.txt", "separators/step_separator_log_prepared.txt"),
        ("separators/step_separator_equality_log.txt", "separators/step_separator_log_prepared.txt"),
        ("separators/step_separator_underscore_log.txt", "separators/step_separator_log_prepared.txt"),
        ("separators/fancy_separator_log.txt", "separators/fancy_separator_log_prepared.txt"),
    ],
)
def test_replace_code_separators(test_file, expected_file):
    log = read_file("test_res/test_logs", test_file)
    expected_log = read_file("test_res/test_logs", expected_file)
    assert text_processing.replace_code_separators(log) == expected_log


def test_remove_webdriver_auxiliary_info():
    log = read_file_lines("test_res/test_logs/webdriver", "webdriver_oneliners.txt")[0:-1]
    expected_log = read_file_lines("test_res/test_logs/webdriver", "webdriver_oneliners_prepared.txt")
    for i, line in enumerate(log):
        assert text_processing.remove_webdriver_auxiliary_info(line) == expected_log[i]


@pytest.mark.parametrize(
    "test_file, expected_file",
    [
        (
            "stacktraces/webdriver_selenide_stacktrace.txt",
            "stacktraces/webdriver_selenide_stacktrace_no_webdriver.txt",
        ),
    ],
)
def test_remove_webdriver_auxiliary_info_big(test_file, expected_file):
    log = read_file("test_res/test_logs", test_file)
    expected_log = read_file("test_res/test_logs", expected_file)
    assert text_processing.remove_webdriver_auxiliary_info(log) == expected_log


def test_find_test_methods_in_text():
    logs = json.loads(read_file("test_res/fixtures", "example_logs.json"))
    for log in logs:
        assert text_processing.find_test_methods_in_text(log["log"]) == set(log["expected_test_methods"])


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
    assert text_processing.remove_credentials_from_url(url) == expected_url
