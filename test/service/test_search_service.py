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

import json
import unittest
from unittest.mock import MagicMock

from app.commons.model import launch_objects
from app.service import SearchService
from app.utils import utils
from test import APP_CONFIG, get_fixture
from test.mock_service import TestService
from test.service import (
    get_basic_search_test_calls,
    get_extended_search_test_calls,
    get_index_found_call,
    get_search_log_info,
    get_search_logs_for_search_service,
)


class TestSearchService(TestService):

    @utils.ignore_warnings
    def test_search_logs(self):
        """Test search logs"""

        # Common search logs objects
        basic_search_logs = get_search_logs_for_search_service()
        search_logs_with_app_config = get_search_logs_for_search_service()
        empty_message_search_logs = get_search_logs_for_search_service(log_messages=[""])
        non_empty_message_search_logs = get_search_logs_for_search_service(log_messages=["error occurred once"])
        status_code_search_logs = get_search_logs_for_search_service(
            log_messages=["error occurred once status code: 500 but got 200"]
        )
        search_logs_with_analyzer_config = get_search_logs_for_search_service(
            log_messages=["error occurred once"],
            analyzer_config=launch_objects.AnalyzerConf(allMessagesShouldMatch=True),
        )

        tests = [
            # Test case 0: Basic test with no hits
            {
                "test_calls": get_basic_search_test_calls("1", self.search_logs_rq, self.no_hits_search_rs),
                "rq": basic_search_logs,
                "expected_count": 0,
            },
            # Test case 1: Basic test with no hits and app config
            {
                "test_calls": get_basic_search_test_calls("rp_1", self.search_logs_rq, self.no_hits_search_rs),
                "rq": search_logs_with_app_config,
                "app_config": APP_CONFIG,
                "expected_count": 0,
            },
            # Test case 2: Empty log messages
            {
                "test_calls": [get_index_found_call("1")],
                "rq": empty_message_search_logs,
                "expected_count": 0,
            },
            # Test case 3: One hit with search logs
            {
                "test_calls": get_basic_search_test_calls(
                    "1", self.search_logs_rq, self.one_hit_search_rs_search_logs
                ),
                "rq": basic_search_logs,
                "expected_count": 0,
            },
            # Test case 4: Two hits with not found and search logs
            {
                "test_calls": get_extended_search_test_calls(
                    "1",
                    self.search_logs_rq_not_found,
                    self.two_hits_search_rs_search_logs,
                    self.search_not_merged_logs_by_test_item,
                    self.two_hits_search_rs_search_logs,
                ),
                "rq": non_empty_message_search_logs,
                "expected_count": 1,
            },
            # Test case 5: Two hits with status codes
            {
                "test_calls": get_extended_search_test_calls(
                    "1",
                    self.search_logs_rq_with_status_codes,
                    self.two_hits_search_rs_search_logs_with_status_codes,
                    self.search_not_merged_logs_by_test_item,
                    self.two_hits_search_rs_search_logs_with_status_codes,
                ),
                "rq": status_code_search_logs,
                "expected_count": 1,
                "response": [get_search_log_info(2, 1, 95)],
            },
            # Test case 6: Two hits with not found and app config
            {
                "test_calls": get_extended_search_test_calls(
                    "rp_1",
                    self.search_logs_rq_not_found,
                    self.two_hits_search_rs_search_logs,
                    self.search_not_merged_logs_by_test_item,
                    self.two_hits_search_rs_search_logs,
                ),
                "rq": non_empty_message_search_logs,
                "app_config": APP_CONFIG,
                "expected_count": 1,
                "response": [get_search_log_info(1, 1, 100)],
            },
            # Test case 7: Two hits with analyzer config
            {
                "test_calls": get_extended_search_test_calls(
                    "1",
                    self.search_logs_rq_not_found,
                    self.two_hits_search_rs_search_logs,
                    self.search_not_merged_logs_by_test_item,
                    self.two_hits_search_rs_search_logs,
                ),
                "rq": search_logs_with_analyzer_config,
                "expected_count": 1,
                "response": [get_search_log_info(1, 1, 100)],
            },
        ]

        for idx, test in enumerate(tests):
            print(f"Running test case idx: {idx}")
            self._start_server(test["test_calls"])
            app_config = self.app_config
            if "app_config" in test:
                app_config = test["app_config"]
            search_service = SearchService(app_config=app_config, search_cfg=self.get_default_search_config())

            search_service.es_client.es_client.scroll = MagicMock(
                return_value=json.loads(get_fixture(self.no_hits_search_rs))
            )

            response = search_service.search_logs(test["rq"])
            assert len(response) == test["expected_count"]
            if "response" in test:
                assert response == test["response"]

            TestSearchService.shutdown_server(test["test_calls"])


if __name__ == "__main__":
    unittest.main()
