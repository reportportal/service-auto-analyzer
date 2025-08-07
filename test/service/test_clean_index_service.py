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

import json
import unittest
from unittest.mock import MagicMock

from app.commons.model import launch_objects
from app.service.clean_index_service import CleanIndexService
from app.utils import utils
from test import APP_CONFIG, get_fixture
from test.mock_service import TestService
from test.service import (
    get_bulk_call,
    get_index_found_call,
    get_index_not_found_call,
    get_search_for_logs_call_with_parameters,
)


class TestCleanIndexService(TestService):

    @utils.ignore_warnings
    def test_clean_index(self):
        """Test cleaning index logs"""
        common_index_steps: list[dict] = [
            get_index_found_call("1"),
            get_search_for_logs_call_with_parameters(
                "1", get_fixture(self.search_not_merged_logs_for_delete), get_fixture(self.one_hit_search_rs)
            ),
            get_bulk_call(None, get_fixture(self.delete_logs_rs)),
            get_search_for_logs_call_with_parameters(
                "1", get_fixture(self.search_merged_logs), get_fixture(self.one_hit_search_rs)
            ),
            get_bulk_call(None, get_fixture(self.delete_logs_rs)),
            get_search_for_logs_call_with_parameters(
                "1", get_fixture(self.search_not_merged_logs), get_fixture(self.one_hit_search_rs)
            ),
            get_bulk_call(get_fixture(self.index_logs_rq), get_fixture(self.index_logs_rs)),
        ]

        clean_index_test = common_index_steps.copy()
        clean_index_test.append(get_index_not_found_call("1_suggest"))

        clean_index_once_again_test = common_index_steps.copy()
        clean_index_once_again_test.extend(
            [
                get_index_found_call("1_suggest"),
                get_search_for_logs_call_with_parameters(
                    "1_suggest",
                    get_fixture(self.search_suggest_info_ids_query),
                    get_fixture(self.one_hit_search_suggest_info_rs),
                ),
                get_bulk_call(get_fixture(self.delete_suggest_logs_rq), get_fixture(self.delete_logs_rs)),
            ]
        )
        tests = [
            # Test case 0: Index not found
            {
                "test_calls": [
                    get_index_not_found_call("2"),
                    get_index_not_found_call("2_suggest"),
                ],
                "rq": launch_objects.CleanIndex(ids=[1], project=2),
                "expected_count": 0,
            },
            # Test case 1: Index not found with index prefix
            {
                "test_calls": [
                    get_index_not_found_call("rp_2"),
                    get_index_not_found_call("rp_2_suggest"),
                ],
                "rq": launch_objects.CleanIndex(ids=[1], project=2),
                "app_config": APP_CONFIG,
                "expected_count": 0,
            },
            # Test case 2: Clean index
            {
                "test_calls": clean_index_test,
                "rq": launch_objects.CleanIndex(ids=[1], project=1),
                "expected_count": 1,
            },
            {
                "test_calls": clean_index_once_again_test,
                "rq": launch_objects.CleanIndex(ids=[1], project=1),
                "expected_count": 1,
            },
            {
                "test_calls": [
                    get_index_found_call("rp_1"),
                    get_search_for_logs_call_with_parameters(
                        "rp_1",
                        get_fixture(self.search_not_merged_logs_for_delete),
                        get_fixture(self.one_hit_search_rs),
                    ),
                    get_bulk_call(None, get_fixture(self.delete_logs_rs)),
                    get_search_for_logs_call_with_parameters(
                        "rp_1", get_fixture(self.search_merged_logs), get_fixture(self.one_hit_search_rs)
                    ),
                    get_bulk_call(None, get_fixture(self.delete_logs_rs)),
                    get_search_for_logs_call_with_parameters(
                        "rp_1", get_fixture(self.search_not_merged_logs), get_fixture(self.one_hit_search_rs)
                    ),
                    get_bulk_call(get_fixture(self.index_logs_rq), get_fixture(self.index_logs_rs)),
                    get_index_found_call("rp_1_suggest"),
                    get_search_for_logs_call_with_parameters(
                        "rp_1_suggest",
                        get_fixture(self.search_suggest_info_ids_query),
                        get_fixture(self.one_hit_search_suggest_info_rs),
                    ),
                    get_bulk_call(
                        get_fixture(self.delete_suggest_logs_rq_with_prefix), get_fixture(self.delete_logs_rs)
                    ),
                ],
                "rq": launch_objects.CleanIndex(ids=[1], project=1),
                "app_config": APP_CONFIG,
                "expected_count": 1,
            },
        ]

        for idx, test in enumerate(tests):
            print(f"Test case number: {idx}")
            self._start_server(test["test_calls"])
            app_config = self.app_config
            if "app_config" in test:
                app_config = test["app_config"]
            _clean_index_service = CleanIndexService(app_config=app_config)
            _clean_index_service.es_client.es_client.scroll = MagicMock(
                return_value=json.loads(get_fixture(self.no_hits_search_rs))
            )
            _clean_index_service.suggest_info_service.es_client.es_client.scroll = MagicMock(
                return_value=json.loads(get_fixture(self.no_hits_search_rs))
            )

            response = _clean_index_service.delete_logs(test["rq"])

            assert test["expected_count"] == response

            TestCleanIndexService.shutdown_server(test["test_calls"])


if __name__ == "__main__":
    unittest.main()
