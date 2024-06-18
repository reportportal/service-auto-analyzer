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

from app.utils.log_preparation import (basic_prepare, prepare_message, prepare_message_without_params,
                                       prepare_exception_message_no_urls_paths, prepare_exception_message_no_params,
                                       prepare_exception_message_and_stacktrace)
from app.utils import text_processing


class PreparedLogMessage:

    original_message: str
    number_of_lines: int
    _clean_message: str = None
    _test_and_methods: set[str] = None
    _message: str = None
    _message_no_params: str = None
    _message_no_params_and_brackets: str = None
    _raw_exception_message: str = None
    _exception_message: str = None
    _stacktrace: str = None
    _exception_message_urls: str = None
    _exception_message_no_urls: str = None
    _exception_message_paths: str = None
    _exception_message_no_paths: str = None
    _exception_message_potential_status_codes: str = None
    _exception_message_no_urls_paths: str = None
    _exception_message_params: str = None
    _exception_message_no_params: str = None
    _exception_message_no_params_and_brackets: str = None
    _exception_message_no_numbers: str = None
    _exception_message_numbers: str = None
    _exception_found: str = None
    _exception_found_extended: str = None
    _test_and_methods_extended: str = None
    _stacktrace_paths: str = None
    _stacktrace_no_paths: str = None
    _stacktrace_no_paths_extended: str = None

    def __init__(self, message: str, number_of_lines: int):
        self.original_message = message
        self.number_of_lines = number_of_lines

    def __str__(self):
        return self.original_message

    @property
    def clean_message(self) -> str:
        if not self._clean_message:
            self._clean_message = basic_prepare(self.original_message)
        return self._clean_message

    @property
    def test_and_methods(self) -> set[str]:
        if not self._test_and_methods:
            self._test_and_methods = text_processing.find_test_methods_in_text(self.clean_message)
        return self._test_and_methods

    @property
    def message(self) -> str:
        if not self._message:
            self._message = prepare_message(self.clean_message, self.number_of_lines, self.test_and_methods)
        return self._message

    @property
    def message_no_params(self) -> str:
        if not self._message_no_params:
            self._message_no_params = prepare_message_without_params(self.message)
        return self._message_no_params

    @property
    def message_no_params_and_brackets(self) -> str:
        if not self._message_no_params_and_brackets:
            self._message_no_params_and_brackets = text_processing.clean_brackets(
                self.message_no_params)
        return self._message_no_params_and_brackets

    @property
    def raw_exception_message(self) -> str:
        if not self._raw_exception_message:
            self._raw_exception_message, self._stacktrace = prepare_exception_message_and_stacktrace(
                self.clean_message)
        return self._raw_exception_message

    @property
    def exception_message(self) -> str:
        if not self._exception_message:
            self._exception_message = text_processing.replace_text_pieces(
                self.raw_exception_message, self.test_and_methods)
        return self._exception_message

    @property
    def stacktrace(self) -> str:
        if not self._stacktrace:
            self._raw_exception_message, self._stacktrace = prepare_exception_message_and_stacktrace(
                self.clean_message)
        return self._stacktrace

    @property
    def exception_message_urls(self) -> str:
        if not self._exception_message_urls:
            self._exception_message_urls = " ".join(text_processing.extract_urls(self.raw_exception_message))
        return self._exception_message_urls

    @property
    def exception_message_no_urls(self) -> str:
        if not self._exception_message_no_urls:
            self._exception_message_no_urls = text_processing.clean_from_urls(self.raw_exception_message)
        return self._exception_message_no_urls

    @property
    def exception_message_paths(self) -> str:
        if not self._exception_message_paths:
            self._exception_message_paths = " ".join(text_processing.extract_paths(self.exception_message_no_urls))
        return self._exception_message_paths

    @property
    def exception_message_no_paths(self) -> str:
        if not self._exception_message_no_paths:
            self._exception_message_no_paths = text_processing.clean_from_paths(self.exception_message_no_urls)
        return self._exception_message_no_paths

    @property
    def exception_message_potential_status_codes(self) -> str:
        if not self._exception_message_potential_status_codes:
            self._exception_message_potential_status_codes = " ".join(
                text_processing.get_potential_status_codes(self.exception_message_no_paths))
        return self._exception_message_potential_status_codes

    @property
    def exception_message_no_urls_paths(self) -> str:
        if not self._exception_message_no_urls_paths:
            self._exception_message_no_urls_paths = prepare_exception_message_no_urls_paths(
                self.exception_message_no_paths, self.test_and_methods)
        return self._exception_message_no_urls_paths

    @property
    def exception_message_params(self) -> str:
        if not self._exception_message_params:
            self._exception_message_params = " ".join(text_processing.extract_message_params(
                self.exception_message_no_urls_paths))
        return self._exception_message_params

    @property
    def exception_message_no_params(self) -> str:
        if not self._exception_message_no_params:
            self._exception_message_no_params = prepare_exception_message_no_params(
                self.exception_message_no_urls_paths)
        return self._exception_message_no_params

    @property
    def exception_message_no_numbers(self) -> str:
        if not self._exception_message_no_numbers:
            self._exception_message_no_numbers = text_processing.remove_numbers(self.exception_message)
        return self._exception_message_no_numbers

    @property
    def exception_message_no_params_and_brackets(self) -> str:
        if not self._exception_message_no_params_and_brackets:
            self._exception_message_no_params_and_brackets = text_processing.clean_brackets(
                self.exception_message_no_params)
        return self._exception_message_no_params_and_brackets

    @property
    def exception_message_numbers(self) -> str:
        if not self._exception_message_numbers:
            self._exception_message_numbers = text_processing.find_only_numbers(self.exception_message)
        return self._exception_message_numbers

    @property
    def exception_found(self) -> str:
        if not self._exception_found:
            self._exception_found = text_processing.get_found_exceptions(self.exception_message_no_numbers)
        return self._exception_found

    @property
    def exception_found_extended(self) -> str:
        if not self._exception_found_extended:
            self._exception_found_extended = text_processing.enrich_found_exceptions(self.exception_found)
        return self._exception_found_extended

    # TODO: This is used in training only, subject to remove
    @property
    def stacktrace_paths(self) -> str:
        if not self._stacktrace_paths:
            self._stacktrace_paths = " ".join(text_processing.extract_paths(self.stacktrace))
        return self._stacktrace_paths

    # TODO: This is used in training only, subject to remove
    @property
    def stacktrace_no_paths(self) -> str:
        if not self._stacktrace_no_paths:
            self._stacktrace_no_paths = text_processing.clean_from_paths(self.stacktrace)
        return self._stacktrace_no_paths

    # TODO: This is used in training only, subject to remove
    @property
    def stacktrace_no_paths_extended(self) -> str:
        if not self._stacktrace_no_paths_extended:
            self._stacktrace_no_paths_extended = text_processing.enrich_text_with_method_and_classes(
                self.stacktrace_no_paths)
        return self._stacktrace_no_paths_extended

    @property
    def test_and_methods_extended(self) -> str:
        if not self._test_and_methods_extended:
            self._test_and_methods_extended = text_processing.enrich_text_with_method_and_classes(
                " ".join(self.test_and_methods))
        return self._test_and_methods_extended
