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

import pytest

from app.commons.prepared_log import PreparedLogMessage
from test import read_file


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
        ("messages/error_base64.txt", "messages/result_error_no_data.json"),
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


SYNTHETIC_TESTS = [
    "Failed to load file from C:\\Users\\username\\Documents\\project\\file.txt",
    "Error reading C:\\Program Files\\App\\logs\\error.log and D:\\Backup\\logs\\app.log",
    "Could not open /home/user/projects/app/config.yml",
    "Could not open /home/user/John\\ Doe/app/config.yml",
    "Error reading files /var/log/app.log and /opt/app/data/file.json",
    "Failed to fetch https://example.com/api/v1/data and access /etc/app/config.json",
    "Error loading configuration from C:\\config\\app.ini and data from https://api.example.org/v2/users",
    "Failed to access http://localhost:8080/api/v1/path/to/resource and https://192.168.1.1/admin",
    "Application could not read C:\\Users\\Admin\\config.ini or /etc/app/settings.json",
    "Error in C:\\Program Files (x86)\\My App\\data files\\log-2023-01.txt",
    'Error in C:\\"Program Files (x86)"\\"My App"\\"data files"\\log-2023-01.txt',
    "Error in directory /home/user/http_modules/config but not in https://example.com/path",
]

SYNTHETIC_TESTS_URLS = [
    [],
    [],
    [],
    [],
    [],
    ["https://example.com/api/v1/data"],
    ["https://api.example.org/v2/users"],
    # Mute Sonar about hardcoding IP addresses. because this is test data
    ["http://localhost:8080/api/v1/path/to/resource", "https://192.168.1.1/admin"],  # NOSONAR
    [],
    [],
    [],
    ["https://example.com/path"],
]

SYNTHETIC_TESTS_PATHS = [
    ["C:\\Users\\username\\Documents\\project\\file.txt"],
    ["C:\\Program Files\\App\\logs\\error.log", "D:\\Backup\\logs\\app.log"],
    ["/home/user/projects/app/config.yml"],
    ["/home/user/John\\ Doe/app/config.yml"],
    ["/var/log/app.log", "/opt/app/data/file.json"],
    ["/etc/app/config.json"],
    ["C:\\config\\app.ini"],
    [],
    ["C:\\Users\\Admin\\config.ini", "/etc/app/settings.json"],
    ["C:\\Program Files (x86)\\My App\\data files\\log-2023-01.txt"],
    ['C:\\"Program Files (x86)"\\"My App"\\"data files"\\log-2023-01.txt'],
    ["/home/user/http_modules/config"],
]

SYNTHETIC_URL_TEST_CASES = zip(SYNTHETIC_TESTS, SYNTHETIC_TESTS_URLS)
SYNTHETIC_PATH_TEST_CASES = zip(SYNTHETIC_TESTS, SYNTHETIC_TESTS_PATHS)


@pytest.mark.parametrize(
    "message, expected_urls",
    SYNTHETIC_URL_TEST_CASES,
)
def test_extract_urls_synthetic(message, expected_urls):
    prepared_log = PreparedLogMessage("", -1)
    prepared_log._exception_message = message

    assert prepared_log.exception_message_urls == " ".join(expected_urls)


@pytest.mark.parametrize(
    "message, expected_paths",
    SYNTHETIC_PATH_TEST_CASES,
)
def test_extract_paths_synthetic(message, expected_paths):
    prepared_log = PreparedLogMessage("", -1)
    prepared_log._exception_message = message

    assert prepared_log.exception_message_paths == " ".join(expected_paths)


@pytest.mark.parametrize(
    "test_file, expected_status_codes",
    [
        ["status_codes/expected_but_was.txt", "409 404"],
        ["status_codes/status_code_in_json.txt", "500"],
        ["status_codes/graphql_error.txt", "401"],
        ["status_codes/status_code_in_text_01.txt", "500"],
        ["status_codes/status_code_in_text_02.txt", "404"],
    ],
)
def test_extract_status_codes(test_file, expected_status_codes):
    log = read_file("test_res/test_logs", test_file)
    prepared_log = PreparedLogMessage(log, -1)
    assert prepared_log.exception_message_potential_status_codes == expected_status_codes
