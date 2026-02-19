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

from app.utils import log_preparation, text_processing, utils
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
    logs = json.loads(read_file("test_res/test_logs", "example_logs.json"))
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


@pytest.mark.parametrize(
    "base_text, other_texts, expected_scores",
    [
        # Test identical texts
        (
            "org.openqa.selenium.TimeoutException: ErrorCodec.decode",
            ["org.openqa.selenium.TimeoutException: ErrorCodec.decode"],
            [(1.0, False)],
        ),
        # Test completely different exceptions
        (
            "org.openqa.selenium.TimeoutException",
            ["java.lang.NullPointerException"],
            [(0.11, False)],  # Should be low similarity
        ),
        # Test same exception type, different details
        (
            "org.openqa.selenium.TimeoutException: ErrorCodec.decode",
            ["org.openqa.selenium.TimeoutException: Different error"],
            [(0.71, False)],  # Should be moderate similarity (approximate)
        ),
        # Test empty strings
        ("", [""], [(0.0, True)]),
        # Test one empty string
        ("some text", [""], [(0.0, False)]),
        # Test multiple texts at once
        (
            "org.openqa.selenium.TimeoutException",
            [
                "org.openqa.selenium.TimeoutException",
                "java.lang.NullPointerException",
                "org.openqa.selenium.WebDriverException",
            ],
            [(1.0, False), (0.11, False), (0.60, False)],  # Expected approximate similarities
        ),
    ],
)
def test_calculate_text_similarity_basic_cases(base_text, other_texts, expected_scores):
    """Test basic text similarity calculation cases"""
    actual_scores = text_processing.calculate_text_similarity(base_text, other_texts)

    assert len(actual_scores) == len(expected_scores)

    for actual, expected in zip(actual_scores, expected_scores):
        assert actual.similarity == pytest.approx(expected[0], abs=0.01)
        assert actual.both_empty == expected[1]


@pytest.mark.parametrize(
    "text, expected_preprocessing_contains",
    [
        (
            "org.openqa.selenium.TimeoutException: ErrorCodec.decode(ErrorCodec.java:167)",
            ["timeout", "exception", "error", "codec", "decode"],
        ),
        (
            'AppiumBy.iOSNsPredicate: label CONTAINS[c] "Continue"',
            ["appium", "predicate", "label", "contains", "continue"],
        ),
        (
            "CamelCaseExample_with_snake_case",
            ["camel", "case", "example", "snake"],
        ),
        (
            "XMLHttpRequest.send()",
            ["xml", "http", "request", "send"],
        ),
        (
            "build_info: version: '4.33.0', revision: '2c6aaad03a'",
            ["build", "info", "version", "revision", "4", "33", "0"],
        ),
    ],
)
def test_preprocess_text_for_similarity(text, expected_preprocessing_contains):
    """Test text preprocessing for similarity calculation"""
    processed = " ".join(text_processing.preprocess_text_for_similarity(text))

    # Check that the processed text is lowercase
    assert processed == processed.lower()

    # Check that expected words are present in the processed text
    tokens = set(processed.split())
    for expected_word in expected_preprocessing_contains:
        assert expected_word in tokens

    # Check that no punctuation remains (except spaces)
    assert not any(char in processed for char in ".,;:!?()[]{}\"'")


@pytest.mark.parametrize(
    "base_text, other_texts",
    [
        # Test with numbers that should be normalized
        (
            "Error code 404: File not found at line 123",
            ["Error code 500: File not found at line 456"],
        ),
        # Test with URLs that should be normalized
        (
            "Failed to connect to https://api.example.com/v1/users",
            ["Failed to connect to https://api.example.com/v2/products"],
        ),
        # Test with file paths that should be normalized
        (
            "File not found: /path/to/file.java:150",
            ["File not found: /different/path/file.java:200"],
        ),
        # Test with special characters and mixed case
        (
            "MyClass.methodName() threw NullPointerException",
            ["MyClass.methodName() threw IllegalArgumentException"],
        ),
    ],
)
def test_calculate_text_similarity_edge_cases(base_text, other_texts):
    """Test text similarity calculation with edge cases"""
    actual_scores = text_processing.calculate_text_similarity(base_text, other_texts)

    # Basic validation
    assert len(actual_scores) == len(other_texts)

    for score in actual_scores:
        assert 0.0 <= score.similarity <= 1.0
        # For these edge cases, similarity should be moderate to high
        # since they have similar structure but different details
        assert score.similarity > 0.3  # Should have some similarity due to common words


def test_calculate_text_similarity_no_other_texts():
    """Test calculate_text_similarity with no other texts provided"""
    result = text_processing.calculate_text_similarity("base text", [])
    assert result == []


def test_calculate_text_similarity_appium_exceptions():
    """Test text similarity with appium exception files if available"""
    ios_exception = log_preparation.clean_message(
        utils.read_file("test_res/test_logs/stacktraces", "appium_ios_exception.txt")
    )
    android_exception = log_preparation.clean_message(
        utils.read_file("test_res/test_logs/stacktraces", "appium_android_exception.txt")
    )

    # Test overall similarity
    similarity_scores = text_processing.calculate_text_similarity(ios_exception, [android_exception])
    assert len(similarity_scores) == 1
    assert 0.0 <= similarity_scores[0].similarity <= 1.0

    # Test first line similarity
    ios_first_line = ios_exception.split("\n")[0]
    android_first_line = android_exception.split("\n")[0]
    first_line_scores = text_processing.calculate_text_similarity(ios_first_line, [android_first_line])
    assert len(first_line_scores) == 1
    assert 0.0 <= first_line_scores[0].similarity <= 1.0


@pytest.mark.parametrize(
    "base_text, other_texts, expected_length",
    [
        ("base", ["base text1"], 1),
        ("base", ["base text1", "base text2"], 2),
        ("base", ["base text1", "base text2", "base text3", "base text4 text5"], 4),
        ("base", [], 0),
    ],
)
def test_calculate_text_similarity_multiple_texts(base_text, other_texts, expected_length):
    """Test that calculate_text_similarity handles multiple texts correctly"""
    scores = text_processing.calculate_text_similarity(base_text, other_texts)
    assert len(scores) == expected_length

    for score in scores:
        assert 0.0 <= score.similarity <= 1.0


REQUEST_TEXT = (
    "new-string [TestNG-tests-SPECIALNUMBER] ERROR c.e.t.r.q.w.c.FailureLoggingListener - Test "
    "createExternalSystemUnableInteractWithExternalSystem has been broken with exception org.testng.TestException :\n"
    "Incorrect Error Type. Expected :  UNABLE_INTERACT_WITH_EXTERNAL_SYSTEM, but was 'PROJECT_NOT_FOUND'."
)

HISTORY_TEXTS = [
    "new-string [TestNG-tests-SPECIALNUMBER] ERROR c.e.t.r.q.w.c.FailureLoggingListener - Test "
    "createExternalSystemUnableInteractWithExternalSystem has been fail with exception org.testng.TestException :\n"
    "Incorrect Error Type. Expected :  UNABLE_INTERACT_WITH_EXTERNAL_SYSTEM, but was 'PROJECT_NOT_FOUND'.,"
    "new-string [TestNG-tests-SPECIALNUMBER] ERROR c.e.t.r.q.w.c.FailureLoggingListener - Test "
    "createExternalSystemUnableInteractWithExternalSystem has been failed with exception org.testng.TestException :\n"
    "Incorrect Error Type. Expected :  UNABLE_INTERACT_WITH_EXTERNAL_SYSTEM, but was 'PROJECT_NOT_FOUND'.",
    "new-string [TestNG-tests-SPECIALNUMBER] ERROR c.e.t.r.q.w.c.FailureLoggingListener - Test "
    "createExternalSystemUnableInteractWithExternalSystem has been failure with exception org.testng.TestException :\n"
    "Incorrect Error Type. Expected :  UNABLE_INTERACT_WITH_EXTERNAL_SYSTEM, but was 'PROJECT_NOT_FOUND'.",
    "new-string [TestNG-tests-SPECIALNUMBER] ERROR c.e.t.r.q.w.c.FailureLoggingListener - Test "
    "createExternalSystemUnableInteractWithExternalSystem has been break with exception org.testng.TestException :\n"
    "Incorrect Error Type. Expected :  UNABLE_INTERACT_WITH_EXTERNAL_SYSTEM, but was 'PROJECT_NOT_FOUND'.",
    "new-string [TestNG-tests-SPECIALNUMBER] ERROR c.e.t.r.q.w.c.FailureLoggingListener - Test "
    "createExternalSystemUnableInteractWithExternalSystem has been ended with exception org.testng.TestException :\n"
    "Incorrect Error Type. Expected :  UNABLE_INTERACT_WITH_EXTERNAL_SYSTEM, but was 'PROJECT_NOT_FOUND'.",
]


def test_calculate_text_similarity_script_scenarios():
    """Users complain that if they "slightly" change log message analyzer stops to classify test items because of IDF.

    This test ensures that IDF is off for `calculate_text_similarity` function, which is used in classification.
    """
    previous_similarities = None
    for n in range(1, len(HISTORY_TEXTS) + 1):
        result = text_processing.calculate_text_similarity(REQUEST_TEXT, HISTORY_TEXTS[:n])
        assert len(result) == n
        if previous_similarities:
            for i, r in enumerate(previous_similarities):
                assert result[i].similarity == pytest.approx(r.similarity, abs=0.01)
        previous_similarities = result


@pytest.mark.parametrize(
    "test_texts, expected_result",
    [
        (["a", "a", "a"], [2]),
        (["a", "b", "a"], [1, 2]),
        (["a", "b", "b"], [0, 2]),
        (["a", "b", "a", "b"], [2, 3]),
        (["a", "b", "a", "b", "a"], [3, 4]),
        (["a", "b", "a", "b", "a", "c"], [3, 4, 5]),
    ],
)
def test_find_last_unique_texts(test_texts, expected_result):
    logs_left = text_processing.find_last_unique_texts(0.95, test_texts)
    assert logs_left == expected_result
