#  Copyright 2024 EPAM Systems
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

from test import read_file_lines, read_file
from app.utils import log_preparation


# def test_remove_starting_thread_name():
#     log = read_file_lines('test_res/test_logs', 'log_line_timestamps.txt')
#     expected_log = read_file_lines('test_res/test_logs', 'log_line_prepared.txt')
#     for i, line in enumerate(log):
#         assert log_preparation.basic_prepare(line) == expected_log[i].strip()


def test_separators_log_prepare():
    log = read_file('test_res/test_logs/separators', 'mixed_markdown_separators.txt')
    expected_log = read_file('test_res/test_logs/separators', 'mixed_markdown_separators_prepared.txt')
    assert log_preparation.basic_prepare(log) == expected_log.strip()
