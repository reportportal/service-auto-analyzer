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

from app.utils import text_processing


def basic_prepare(message: str) -> str:
    cleaned_message = message.strip()
    # Sometimes log level goes first
    cleaned_message = text_processing.remove_starting_log_level(cleaned_message)
    cleaned_message = text_processing.remove_starting_datetime(cleaned_message)
    cleaned_message = text_processing.remove_starting_log_level(cleaned_message)
    cleaned_message = text_processing.remove_starting_thread_id(cleaned_message)
    cleaned_message = text_processing.remove_starting_thread_name(cleaned_message)
    # Sometimes log level goes after thread name
    cleaned_message = text_processing.remove_starting_log_level(cleaned_message)

    # This should go right after starting garbage clean-up
    cleaned_message = text_processing.unify_line_endings(cleaned_message)
    cleaned_message = text_processing.remove_markdown_mode(cleaned_message)
    cleaned_message = text_processing.delete_empty_lines(cleaned_message)
    return cleaned_message


def clean_message(basic_message: str) -> str:
    cleaned_message = text_processing.replace_code_separators(basic_message)
    cleaned_message = text_processing.remove_webdriver_auxiliary_info(cleaned_message)
    cleaned_message = text_processing.replace_tabs_for_newlines(cleaned_message)
    cleaned_message = text_processing.fix_big_encoded_urls(cleaned_message)
    cleaned_message = text_processing.remove_generated_parts(cleaned_message)
    cleaned_message = text_processing.remove_guid_uuids_from_text(cleaned_message)
    cleaned_message = text_processing.remove_access_tokens(cleaned_message)
    cleaned_message = text_processing.clean_html(cleaned_message)
    cleaned_message = text_processing.delete_empty_lines(cleaned_message)
    cleaned_message = text_processing.leave_only_unique_lines(cleaned_message)
    return cleaned_message


def prepare_message(message: str, number_of_lines: int, test_and_methods: set[str]) -> str:
    cleaned_message = text_processing.first_lines(message, number_of_lines)
    cleaned_message = text_processing.replace_text_pieces(cleaned_message, test_and_methods)
    return cleaned_message


def prepare_message_no_numbers(message: str, number_of_lines: int, test_and_methods: set[str]) -> str:
    cleaned_message = prepare_message(message, number_of_lines, test_and_methods)
    cleaned_message = text_processing.delete_empty_lines(text_processing.remove_numbers(cleaned_message))
    return cleaned_message


def prepare_message_no_params(message: str) -> str:
    cleaned_message = text_processing.clean_from_params(message)
    return cleaned_message


def prepare_exception_message_and_stacktrace(message: str) -> tuple[str, str]:
    exception_message, stacktrace = text_processing.detect_log_description_and_stacktrace(message)
    stacktrace = text_processing.clean_from_brackets(stacktrace)
    stacktrace = text_processing.remove_numbers(stacktrace)
    return exception_message, stacktrace


def prepare_exception_message_no_params(exception_message: str) -> str:
    cleaned_message = text_processing.clean_from_params(exception_message)
    cleaned_message = text_processing.unify_spaces(cleaned_message)
    return cleaned_message


def prepare_exception_message_no_params_no_numbers(exception_message: str) -> str:
    cleaned_message = text_processing.remove_numbers(exception_message)
    cleaned_message = text_processing.clean_from_params(cleaned_message)
    cleaned_message = text_processing.unify_spaces(cleaned_message)
    return cleaned_message
