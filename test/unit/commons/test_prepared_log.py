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

import json
from test import read_file

import pytest

from app.commons.prepared_log import PreparedLogMessage


@pytest.mark.parametrize(
    "test_file, expected_file",
    [
        (
            "stacktraces/log_stacktrace_js.txt",
            "stacktraces/log_stacktrace_js_exception_message_no_params_and_brackets.txt",
        ),
    ],
)
def test_exception_message_no_params_and_brackets(test_file, expected_file):
    log = read_file("test_res/test_logs", test_file)
    expected_log = read_file("test_res/test_logs", expected_file)
    assert PreparedLogMessage(log, -1).exception_message_no_params == expected_log.strip()


@pytest.mark.parametrize(
    "test_file, expected_file",
    [
        ("messages/error_base64.txt", "messages/result_error_no_data.json"),
        ("messages/error_with_json.txt", "messages/result_error_no_data.json"),
        ("messages/error_with_url_1.txt", "messages/result_error_with_url_1.json"),
        ("messages/error_with_url_2.txt", "messages/result_error_with_url_2.json"),
        ("messages/error_with_url_3.txt", "messages/result_error_with_url_3.json"),
        ("messages/error_with_url_4.txt", "messages/result_error_with_url_4.json"),
    ],
)
def test_extract_urls(test_file, expected_file):
    log = read_file("test_res/test_logs", test_file)
    expected_log = json.loads(read_file("test_res/test_logs", expected_file))
    prepared_log = PreparedLogMessage("", -1)
    prepared_log._exception_message = log
    assert prepared_log.exception_message_urls == " ".join(expected_log)


@pytest.mark.parametrize(
    "test_file, expected_file",
    [
        ("messages/error_base64.txt", "messages/result_error_base64_path_1.txt"),
        ("messages/error_with_json.txt", "messages/result_error_no_data.json"),
        ("messages/error_with_url_1.txt", "messages/result_error_no_data.json"),
        ("messages/error_with_url_2.txt", "messages/result_error_no_data.json"),
        ("messages/error_with_url_3.txt", "messages/result_error_no_data.json"),
        ("messages/error_with_url_4.txt", "messages/result_error_no_data.json"),
    ],
)
def test_extract_paths(test_file, expected_file):
    log = read_file("test_res/test_logs", test_file)
    expected_log = json.loads(read_file("test_res/test_logs", expected_file))
    prepared_log = PreparedLogMessage("", -1)
    prepared_log._exception_message = log
    assert prepared_log.exception_message_paths == " ".join(expected_log)
