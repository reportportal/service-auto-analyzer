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
from typing import Optional

from typing_extensions import override

from app.utils import text_processing
from app.utils.log_preparation import (basic_prepare, clean_message, prepare_message, prepare_message_no_numbers,
                                       prepare_message_no_params, prepare_exception_message_no_params_no_numbers,
                                       prepare_exception_message_no_params,
                                       prepare_exception_message_and_stacktrace)


class PreparedLogMessage:
    original_message: str
    number_of_lines: int
    _basic_message: Optional[str] = None
    _clean_message: Optional[str] = None
    _test_and_methods: Optional[set[str]] = None
    _message: Optional[str] = None
    _message_no_params: Optional[str] = None
    _exception_message: Optional[str] = None
    _stacktrace: Optional[str] = None
    _exception_message_urls_list: Optional[list[str]] = None
    _exception_message_urls: Optional[str] = None
    _exception_message_no_urls: Optional[str] = None
    _exception_message_paths: Optional[str] = None
    _exception_message_potential_status_codes: Optional[str] = None
    _exception_message_params: Optional[str] = None
    _exception_message_no_params: Optional[str] = None
    _exception_message_no_numbers: Optional[str] = None
    _exception_message_numbers: Optional[str] = None
    _exception_found: Optional[str] = None
    _exception_found_extended: Optional[str] = None
    _test_and_methods_extended: Optional[str] = None
    _stacktrace_paths: Optional[str] = None
    _stacktrace_no_paths: Optional[str] = None
    _stacktrace_no_paths_extended: Optional[str] = None

    def __init__(self, message: str, number_of_lines: int):
        self.original_message = message
        self.number_of_lines = number_of_lines

    def __str__(self):
        return self.original_message

    @property
    def basic_message(self) -> str:
        if not self._basic_message:
            self._basic_message = basic_prepare(self.original_message)
        return self._basic_message

    @property
    def clean_message(self) -> str:
        if not self._clean_message:
            self._clean_message = clean_message(self.basic_message)
        return self._clean_message

    @property
    def test_and_methods(self) -> set[str]:
        if not self._test_and_methods:
            self._test_and_methods = text_processing.find_test_methods_in_text(self.clean_message)
        return self._test_and_methods

    @property
    def message(self) -> str:
        if not self._message:
            self._message = prepare_message_no_numbers(self.clean_message, self.number_of_lines, self.test_and_methods)
        return self._message

    @property
    def message_no_params(self) -> str:
        if not self._message_no_params:
            self._message_no_params = prepare_message_no_params(self.message)
        return self._message_no_params

    @property
    def exception_message(self) -> str:
        if not self._exception_message:
            self._exception_message, self._stacktrace = prepare_exception_message_and_stacktrace(self.clean_message)
        return self._exception_message

    @property
    def stacktrace(self) -> str:
        if not self._stacktrace:
            self._exception_message, self._stacktrace = prepare_exception_message_and_stacktrace(self.clean_message)
        return self._stacktrace

    @property
    def exception_message_urls_list(self) -> list[str]:
        if not self._exception_message_urls_list:
            self._exception_message_urls_list = text_processing.extract_urls(self.exception_message)
        return self._exception_message_urls_list

    @property
    def exception_message_urls(self) -> str:
        if not self._exception_message_urls:
            self._exception_message_urls = " ".join(self.exception_message_urls_list)
        return self._exception_message_urls

    @property
    def exception_message_no_urls(self) -> str:
        if not self._exception_message_no_urls:
            self._exception_message_no_urls = text_processing.remove_urls(
                self.exception_message, self.exception_message_urls_list)
        return self._exception_message_no_urls

    @property
    def exception_message_paths(self) -> str:
        if not self._exception_message_paths:
            self._exception_message_paths = " ".join(text_processing.extract_paths(self.exception_message_no_urls))
        return self._exception_message_paths

    @property
    def exception_message_potential_status_codes(self) -> str:
        if not self._exception_message_potential_status_codes:
            self._exception_message_potential_status_codes = " ".join(
                text_processing.get_potential_status_codes(self.exception_message))
        return self._exception_message_potential_status_codes

    @property
    def exception_message_params(self) -> str:
        if not self._exception_message_params:
            self._exception_message_params = " ".join(text_processing.extract_message_params(
                self.exception_message))
        return self._exception_message_params

    @property
    def exception_message_no_params(self) -> str:
        if not self._exception_message_no_params:
            self._exception_message_no_params = prepare_exception_message_no_params_no_numbers(
                self.exception_message)
        return self._exception_message_no_params

    @property
    def exception_message_no_numbers(self) -> str:
        if not self._exception_message_no_numbers:
            self._exception_message_no_numbers = text_processing.remove_numbers(self.exception_message)
        return self._exception_message_no_numbers

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

    @property
    def test_and_methods_extended(self) -> str:
        if not self._test_and_methods_extended:
            self._test_and_methods_extended = text_processing.enrich_text_with_method_and_classes(
                " ".join(self.test_and_methods))
        return self._test_and_methods_extended


class PreparedLogMessageClustering(PreparedLogMessage):

    def __init__(self, message: str, number_of_lines: int) -> None:
        super().__init__(message, number_of_lines)

    @override
    @property
    def clean_message(self) -> str:
        return self.basic_message

    @override
    @property
    def message(self) -> str:
        if not self._message:
            self._message = prepare_message(self.clean_message, self.number_of_lines, self.test_and_methods)
        return self._message

    @property
    def exception_message_no_params(self) -> str:
        if not self._exception_message_no_params:
            self._exception_message_no_params = prepare_exception_message_no_params(self.exception_message)
        return self._exception_message_no_params
