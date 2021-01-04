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
import sure # noqa
import httpretty
from unittest.mock import MagicMock
import commons.launch_objects as launch_objects
from service.suggest_patterns_service import SuggestPatternsService
from test.test_service import TestService
from utils import utils


class TestSearchService(TestService):

    @utils.ignore_warnings
    def test_suggest_patterns(self):
        """Test suggest patterns"""
        tests = [
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    },
                                   ],
                "rq":             1,
                "query_data":     [],
                "expected_count_with_labels": [],
                "expected_count_without_labels": []
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   ],
                "rq":             1,
                "query_data":     [],
                "expected_count_with_labels": [],
                "expected_count_without_labels": []
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   ],
                "rq":             1,
                "query_data":     [("assertionError notFoundError", "ab001"),
                                   ("assertionError ifElseError", "pb001"),
                                   ("assertionError commonError", "ab001"),
                                   ("assertionError commonError", "ab001"),
                                   ("assertionError", "ab001"),
                                   ("assertionError commonError", "ab001"),
                                   ("assertionError commonError", "ti001")],
                "expected_count_with_labels": [
                    launch_objects.SuggestPatternLabel(
                        pattern='assertionError', totalCount=24,
                        percentTestItemsWithLabel=0.83, label='ab001'),
                    launch_objects.SuggestPatternLabel(
                        pattern='commonError', totalCount=12,
                        percentTestItemsWithLabel=1.0, label='ab001')],
                "expected_count_without_labels": [
                    launch_objects.SuggestPatternLabel(
                        pattern='assertionError', totalCount=28, percentTestItemsWithLabel=0.0, label=''),
                    launch_objects.SuggestPatternLabel(
                        pattern='commonError', totalCount=16, percentTestItemsWithLabel=0.0, label='')]
            },
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])

                search_service = SuggestPatternsService(app_config=self.app_config,
                                                        search_cfg=self.get_default_search_config())
                search_service.query_data = MagicMock(return_value=test["query_data"])

                response = search_service.suggest_patterns(test["rq"])
                response.suggestionsWithLabels.should.have.length_of(
                    len(test["expected_count_with_labels"]))

                for real_resp, expected_resp in zip(
                        response.suggestionsWithLabels, test["expected_count_with_labels"]):
                    real_resp.should.equal(expected_resp)

                response.suggestionsWithoutLabels.should.have.length_of(
                    len(test["expected_count_without_labels"]))

                for real_resp, expected_resp in zip(
                        response.suggestionsWithoutLabels, test["expected_count_without_labels"]):
                    real_resp.should.equal(expected_resp)

                TestSearchService.shutdown_server(test["test_calls"])


if __name__ == '__main__':
    unittest.main()
