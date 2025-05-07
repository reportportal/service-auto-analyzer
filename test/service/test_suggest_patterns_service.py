"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import unittest
from http import HTTPStatus
from unittest.mock import MagicMock

import httpretty

from app.commons.model import launch_objects
from app.service import SuggestPatternsService
from app.utils import utils
from test import APP_CONFIG
from test.mock_service import TestService


class TestSearchService(TestService):

    @utils.ignore_warnings
    def test_suggest_patterns(self):
        """Test suggest patterns"""
        tests = [
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/1",
                        "status": HTTPStatus.NOT_FOUND,
                    },
                ],
                "rq": 1,
                "query_data": [],
                "expected_count_with_labels": [],
                "expected_count_without_labels": [],
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/1",
                        "status": HTTPStatus.OK,
                    },
                ],
                "rq": 1,
                "query_data": [],
                "expected_count_with_labels": [],
                "expected_count_without_labels": [],
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/rp_1",
                        "status": HTTPStatus.OK,
                    },
                ],
                "app_config": APP_CONFIG,
                "rq": 1,
                "query_data": [],
                "expected_count_with_labels": [],
                "expected_count_without_labels": [],
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/1",
                        "status": HTTPStatus.OK,
                    },
                ],
                "rq": 1,
                "query_data": [
                    ("assertionError notFoundError", "ab001"),
                    ("assertionError ifElseError", "pb001"),
                    ("assertionError commonError", "ab001"),
                    ("assertionError commonError", "ab001"),
                    ("assertionError", "ab001"),
                    ("assertionError commonError", "ab001"),
                    ("assertionError commonError", "ti001"),
                ],
                "expected_count_with_labels": [
                    launch_objects.SuggestPatternLabel(
                        pattern="assertionError", totalCount=24, percentTestItemsWithLabel=0.83, label="ab001"
                    ),
                    launch_objects.SuggestPatternLabel(
                        pattern="commonError", totalCount=12, percentTestItemsWithLabel=1.0, label="ab001"
                    ),
                ],
                "expected_count_without_labels": [
                    launch_objects.SuggestPatternLabel(
                        pattern="assertionError", totalCount=28, percentTestItemsWithLabel=0.0, label=""
                    ),
                    launch_objects.SuggestPatternLabel(
                        pattern="commonError", totalCount=16, percentTestItemsWithLabel=0.0, label=""
                    ),
                ],
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/rp_1",
                        "status": HTTPStatus.OK,
                    },
                ],
                "app_config": APP_CONFIG,
                "rq": 1,
                "query_data": [
                    ("assertionError notFoundError", "ab001"),
                    ("assertionError ifElseError", "pb001"),
                    ("assertionError commonError", "ab001"),
                    ("assertionError commonError", "ab001"),
                    ("assertionError", "ab001"),
                    ("assertionError commonError", "ab001"),
                    ("assertionError commonError", "ti001"),
                ],
                "expected_count_with_labels": [
                    launch_objects.SuggestPatternLabel(
                        pattern="assertionError", totalCount=24, percentTestItemsWithLabel=0.83, label="ab001"
                    ),
                    launch_objects.SuggestPatternLabel(
                        pattern="commonError", totalCount=12, percentTestItemsWithLabel=1.0, label="ab001"
                    ),
                ],
                "expected_count_without_labels": [
                    launch_objects.SuggestPatternLabel(
                        pattern="assertionError", totalCount=28, percentTestItemsWithLabel=0.0, label=""
                    ),
                    launch_objects.SuggestPatternLabel(
                        pattern="commonError", totalCount=16, percentTestItemsWithLabel=0.0, label=""
                    ),
                ],
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
