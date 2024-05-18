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

import pytest

from app.utils import utils, text_processing
from test import read_file_lines, read_file


def test_delete_empty_lines():
    log = utils.read_file('test_res/test_logs', 'reportportal-api.txt')
    expected = utils.read_file('test_res/test_logs', 'reportportal-api-no-empty-lines.txt')

    assert text_processing.delete_empty_lines(log) == expected


def test_filter_empty_lines():
    log = read_file_lines('test_res/test_logs', 'reportportal-api.txt')
    expected = read_file_lines('test_res/test_logs', 'reportportal-api-no-empty-lines.txt')

    assert text_processing.filter_empty_lines(log) == expected


def test_remove_starting_datetime():
    log = read_file_lines('test_res/test_logs', 'log_line_timestamps.txt')
    expected_log = read_file_lines('test_res/test_logs', 'log_line_no_timestamp.txt')
    for i, line in enumerate(log):
        assert text_processing.remove_starting_datetime(line) == expected_log[i]


def test_remove_starting_log_level():
    log = read_file_lines('test_res/test_logs', 'log_line_no_timestamp.txt')
    expected_log = read_file_lines('test_res/test_logs', 'log_line_no_log_level.txt')
    for i, line in enumerate(log):
        assert text_processing.remove_starting_log_level(line) == expected_log[i]


def test_remove_starting_thread_id():
    log = read_file_lines('test_res/test_logs', 'log_line_no_log_level.txt')
    expected_log = read_file_lines('test_res/test_logs', 'log_line_no_thread_id.txt')
    for i, line in enumerate(log):
        assert text_processing.remove_starting_thread_id(line) == expected_log[i]


def test_remove_starting_thread_namer():
    log = read_file_lines('test_res/test_logs', 'log_line_no_thread_id.txt')
    expected_log = read_file_lines('test_res/test_logs', 'log_line_no_thread_name.txt')
    for i, line in enumerate(log):
        assert text_processing.remove_starting_thread_name(line) == expected_log[i]


@pytest.mark.parametrize(
    'test_file, expected_file',
    [
        ('stacktraces/log_stacktrace_generated.txt', 'stacktraces/log_stacktrace_prepared.txt'),
        ('stacktraces/log_stacktrace_generated_2.txt', 'stacktraces/log_stacktrace_prepared_2.txt'),
        ('stacktraces/log_stacktrace_generated_3.txt', 'stacktraces/log_stacktrace_prepared_3.txt'),
        ('log_locator_with_attribute.txt', 'log_locator_with_attribute_prepared.txt')
    ]
)
def test_remove_generated_parts(test_file, expected_file):
    log = read_file('test_res/test_logs', test_file)
    expected_log = read_file('test_res/test_logs', expected_file)
    assert text_processing.remove_generated_parts(log) == expected_log


def test_clean_from_brackets():
    log = read_file_lines('test_res/test_logs', 'brackets_test.txt')
    expected_log = read_file_lines('test_res/test_logs', 'brackets_test_results.txt')
    for i, line in enumerate(log):
        assert text_processing.clean_from_brackets(line) == expected_log[i]


@pytest.mark.parametrize(
    'message, expected_message',
    [
        ('\t \r\n ', '\n '),
        ('\r\n', '\n'),
        ('\n', '\n'),
        ('\u00A0\u00A0\u00A0\n', '\n'),
        ('\u00A0\r\n', '\n'),
    ]
)
def test_unify_line_endings(message, expected_message):
    assert text_processing.unify_line_endings(message) == expected_message


@pytest.mark.parametrize(
    'message, expected_message',
    [
        ('\t \r\n ', ' '),
        ('\r\n', ' '),
        ('\n', ' '),
        ('\u00A0\u00A0\u00A0\n', ' '),
        ('\u00A0\r\n', ' '),
        ('\u00A0\u2000\u2001', ' '),
        ('\u202F\u205F\u3000', ' '),
        ('a\u202F\u205F\u3000b', 'a b'),
    ]
)
def test_unify_spaces(message, expected_message):
    assert text_processing.unify_spaces(message) == expected_message
