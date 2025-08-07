"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import unittest
from unittest.mock import MagicMock

from app.service import SuggestPatternsService
from app.utils import utils
from test import APP_CONFIG
from test.mock_service import TestService
from test.service import (
    get_common_expected_patterns_with_labels,
    get_common_expected_patterns_without_labels,
    get_common_query_data,
    get_index_found_call,
    get_index_not_found_call,
)


class TestSearchService(TestService):

    @utils.ignore_warnings
    def test_suggest_patterns(self):
        """Test suggest patterns"""

        # Common test data
        common_query_data = get_common_query_data()
        common_expected_with_labels = get_common_expected_patterns_with_labels()
        common_expected_without_labels = get_common_expected_patterns_without_labels()

        tests = [
            # Test case 0: Index not found
            {
                "test_calls": [get_index_not_found_call("1")],
                "rq": 1,
                "query_data": [],
                "expected_count_with_labels": [],
                "expected_count_without_labels": [],
            },
            # Test case 1: Index found but no query data
            {
                "test_calls": [get_index_found_call("1")],
                "rq": 1,
                "query_data": [],
                "expected_count_with_labels": [],
                "expected_count_without_labels": [],
            },
            # Test case 2: Index found with app config but no query data
            {
                "test_calls": [get_index_found_call("rp_1")],
                "app_config": APP_CONFIG,
                "rq": 1,
                "query_data": [],
                "expected_count_with_labels": [],
                "expected_count_without_labels": [],
            },
            # Test case 3: Index found with query data
            {
                "test_calls": [get_index_found_call("1")],
                "rq": 1,
                "query_data": common_query_data,
                "expected_count_with_labels": common_expected_with_labels,
                "expected_count_without_labels": common_expected_without_labels,
            },
            # Test case 4: Index found with app config and query data
            {
                "test_calls": [get_index_found_call("rp_1")],
                "app_config": APP_CONFIG,
                "rq": 1,
                "query_data": common_query_data,
                "expected_count_with_labels": common_expected_with_labels,
                "expected_count_without_labels": common_expected_without_labels,
            },
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                search_service = SuggestPatternsService(
                    app_config=app_config, search_cfg=self.get_default_search_config()
                )
                search_service.query_data = MagicMock(return_value=test["query_data"])

                response = search_service.suggest_patterns(test["rq"])
                assert len(response.suggestionsWithLabels) == len(test["expected_count_with_labels"])

                for real_resp, expected_resp in zip(
                    response.suggestionsWithLabels, test["expected_count_with_labels"]
                ):
                    assert real_resp == expected_resp

                assert len(response.suggestionsWithLabels) == len(test["expected_count_with_labels"])

                for real_resp, expected_resp in zip(
                    response.suggestionsWithoutLabels, test["expected_count_without_labels"]
                ):
                    assert real_resp == expected_resp

                TestSearchService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f"Error in the test case number: {idx}").with_traceback(err.__traceback__)


if __name__ == "__main__":
    unittest.main()
