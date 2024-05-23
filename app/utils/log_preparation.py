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

    cleaned_message = text_processing.replace_tabs_for_newlines(cleaned_message)
    cleaned_message = text_processing.fix_big_encoded_urls(cleaned_message)
    cleaned_message = text_processing.remove_generated_parts(cleaned_message)
    cleaned_message = text_processing.remove_guid_uuids_from_text(cleaned_message)
    cleaned_message = text_processing.remove_access_tokens(cleaned_message)
    cleaned_message = text_processing.clean_html(cleaned_message)
    cleaned_message = text_processing.delete_empty_lines(cleaned_message)
    cleaned_message = text_processing.leave_only_unique_lines(cleaned_message)
    return cleaned_message


def prepare_message(clean_message: str, number_of_lines: int, test_and_methods: set[str]) -> str:
    message = text_processing.first_lines(clean_message, number_of_lines)
    message = text_processing.replace_text_pieces(message, test_and_methods)
    message = text_processing.delete_empty_lines(text_processing.remove_numbers(message))
    return message


def prepare_message_without_params(message: str) -> str:
    message_without_params = text_processing.clean_from_urls(message)
    message_without_params = text_processing.clean_from_paths(message_without_params)
    message_without_params = text_processing.clean_from_params(message_without_params)
    message_without_params = text_processing.remove_starting_datetime(message_without_params)
    message_without_params = text_processing.remove_numbers(message_without_params)
    return message_without_params


def prepare_exception_message_no_urls_paths(exception_message_no_paths: str, test_and_methods: set[str]) -> str:
    detected_message_without_params = text_processing.replace_text_pieces(exception_message_no_paths, test_and_methods)
    return detected_message_without_params


def prepare_exception_message_without_params(prepared_exception_message_no_urls_paths: str) -> str:
    detected_message_without_params = text_processing.clean_from_params(prepared_exception_message_no_urls_paths)
    detected_message_without_params = text_processing.remove_numbers(detected_message_without_params)
    return detected_message_without_params


def prepare_exception_message_and_stacktrace(clean_message: str) -> tuple[str, str]:
    exception_message, stacktrace = text_processing.detect_log_description_and_stacktrace(clean_message)
    stacktrace = text_processing.remove_numbers(stacktrace)
    return exception_message, stacktrace


def prepare_exception_message_no_params(exception_message_no_urls_paths: str) -> str:
    return text_processing.remove_numbers(text_processing.clean_from_params(exception_message_no_urls_paths))
