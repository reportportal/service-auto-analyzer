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

from test import read_file_lines

from app.utils import utils, text_processing


def test_delete_empty_lines():
    log = utils.read_file('test_res/test_logs', 'reportportal-api.txt')
    expected = utils.read_file('test_res/test_logs', 'reportportal-api-no-empty-lines.txt')

    assert text_processing.delete_empty_lines(log) == expected


def test_filter_empty_lines():
    log = read_file_lines('test_res/test_logs', 'reportportal-api.txt')
    expected = read_file_lines('test_res/test_logs', 'reportportal-api-no-empty-lines.txt')

    assert text_processing.filter_empty_lines(log) == expected


def test_remove_starting_datetime():
    log = read_file_lines('test_res/test_logs', 'log_line.txt')
    expected_log = read_file_lines('test_res/test_logs', 'log_line_no_timestamp_current.txt')
    for i, line in enumerate(log):
        assert text_processing.remove_starting_datetime(line) == expected_log[i]
