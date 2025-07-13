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

from app.commons import object_saving
from app.commons.model import launch_objects
from app.machine_learning.models.boosting_decision_maker import BoostingDecisionMaker
from app.service import SuggestService
from app.utils import utils
from test import APP_CONFIG, get_fixture
from test.mock_service import TestService
from test.service import (
    get_basic_suggest_test_calls,
    get_basic_test_item_info,
    get_index_found_call,
    get_index_not_found_call,
    get_search_for_logs_call_no_parameters,
    get_search_for_logs_call_with_parameters,
    get_suggest_analysis_result,
    get_test_item_info_with_empty_logs,
    get_test_item_info_with_logs,
)


class TestSuggestService(TestService):

    @utils.ignore_warnings
    def test_suggest_items(self):
        """Test suggesting test items"""

        # Common test data
        basic_suggest_test_calls = get_basic_suggest_test_calls("1", self.index_logs_rs)
        basic_test_item_info = get_basic_test_item_info()
        test_item_info_with_logs = get_test_item_info_with_logs()
        test_item_info_with_empty_logs = get_test_item_info_with_empty_logs()

        # Test item info from fixtures
        test_item_info_w_logs = launch_objects.TestItemInfo(
            **get_fixture(self.suggest_test_item_info_w_logs, to_json=True)
        )
        test_item_info_w_merged_logs = launch_objects.TestItemInfo(
            **get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)
        )
        test_item_info_cluster = launch_objects.TestItemInfo(
            **get_fixture(self.suggest_test_item_info_cluster, to_json=True)
        )

        # Common msearch results
        no_hits_msearch = [
            get_fixture(self.no_hits_search_rs, to_json=True),
            get_fixture(self.no_hits_search_rs, to_json=True),
            get_fixture(self.no_hits_search_rs, to_json=True),
        ]
        mixed_hits_msearch = [
            get_fixture(self.no_hits_search_rs, to_json=True),
            get_fixture(self.one_hit_search_rs, to_json=True),
            get_fixture(self.one_hit_search_rs, to_json=True),
        ]

        # Custom msearch results
        one_hit_msearch = [
            get_fixture(self.one_hit_search_rs, to_json=True),
            get_fixture(self.one_hit_search_rs, to_json=True),
            get_fixture(self.one_hit_search_rs, to_json=True),
        ]

        two_hits_msearch = [
            get_fixture(self.one_hit_search_rs, to_json=True),
            get_fixture(self.two_hits_search_rs, to_json=True),
            get_fixture(self.two_hits_search_rs, to_json=True),
        ]

        three_hits_msearch = [
            get_fixture(self.two_hits_search_rs, to_json=True),
            get_fixture(self.three_hits_search_rs, to_json=True),
            get_fixture(self.no_hits_search_rs, to_json=True),
        ]

        three_hits_with_duplicate_msearch = [
            get_fixture(self.two_hits_search_rs, to_json=True),
            get_fixture(self.three_hits_search_rs_with_duplicate, to_json=True),
            get_fixture(self.no_hits_search_rs, to_json=True),
        ]

        merged_logs_msearch = [
            get_fixture(self.one_hit_search_rs_merged, to_json=True),
            get_fixture(self.one_hit_search_rs_merged, to_json=True),
            get_fixture(self.one_hit_search_rs_merged, to_json=True),
        ]

        merged_logs_wrong_msearch = [
            get_fixture(self.one_hit_search_rs_merged_wrong, to_json=True),
            get_fixture(self.one_hit_search_rs_merged_wrong, to_json=True),
            get_fixture(self.one_hit_search_rs_merged_wrong, to_json=True),
        ]

        # Cluster-specific test calls
        cluster_test_calls = [
            get_index_found_call("1"),
            get_search_for_logs_call_no_parameters(
                "1", get_fixture(self.search_test_item_cluster), get_fixture(self.three_hits_search_rs)
            ),
            get_search_for_logs_call_with_parameters(
                "1",
                get_fixture(self.search_logs_by_test_item),
                get_fixture(self.three_hits_search_rs),
            ),
        ]

        cluster_small_logs_test_calls = [
            get_index_found_call("1"),
            get_search_for_logs_call_no_parameters(
                "1", get_fixture(self.search_test_item_cluster), get_fixture(self.one_hit_search_rs_small_logs)
            ),
            get_search_for_logs_call_with_parameters(
                "1",
                get_fixture(self.search_logs_by_test_item),
                get_fixture(self.one_hit_search_rs_small_logs),
            ),
        ]

        tests = [
            # Test case 0: Basic test with no results
            {
                "test_calls": basic_suggest_test_calls,
                "test_item_info": basic_test_item_info,
                "expected_result": [],
                "boost_predict": ([], []),
            },
            # Test case 1: Index not found
            {
                "test_calls": [get_index_not_found_call("2")],
                "test_item_info": test_item_info_with_logs,
                "expected_result": [],
                "boost_predict": ([], []),
            },
            # Test case 2: Empty logs
            {
                "test_calls": basic_suggest_test_calls,
                "test_item_info": test_item_info_with_empty_logs,
                "expected_result": [],
                "boost_predict": ([], []),
            },
            # Test case 3: No hits in search
            {
                "test_calls": basic_suggest_test_calls,
                "msearch_results": no_hits_msearch,
                "test_item_info": test_item_info_w_logs,
                "expected_result": [],
                "boost_predict": ([], []),
            },
            # Test case 4: No hits in search (duplicate)
            {
                "test_calls": basic_suggest_test_calls,
                "msearch_results": no_hits_msearch,
                "test_item_info": test_item_info_w_logs,
                "expected_result": [],
                "boost_predict": ([], []),
            },
            # Test case 5: One hit with mixed results
            {
                "test_calls": [get_index_found_call("1")],
                "msearch_results": mixed_hits_msearch,
                "test_item_info": test_item_info_w_logs,
                "expected_result": [get_suggest_analysis_result(match_score=80.0)],
                "boost_predict": ([1], [[0.2, 0.8]]),
            },
            # Test case 6: All one hit results
            {
                "test_calls": [get_index_found_call("1")],
                "msearch_results": one_hit_msearch,
                "test_item_info": test_item_info_w_logs,
                "expected_result": [get_suggest_analysis_result()],
                "boost_predict": ([1], [[0.3, 0.7]]),
            },
            # Test case 7: Two hits results
            {
                "test_calls": [get_index_found_call("1")],
                "msearch_results": two_hits_msearch,
                "test_item_info": test_item_info_w_logs,
                "expected_result": [get_suggest_analysis_result(es_score=15.0)],
                "boost_predict": ([1, 0], [[0.3, 0.7], [0.9, 0.1]]),
            },
            # Test case 8: Two hits with multiple results
            {
                "test_calls": [get_index_found_call("1")],
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "test_item_info": test_item_info_w_logs,
                "expected_result": [
                    get_suggest_analysis_result(es_score=15.0),
                    get_suggest_analysis_result(
                        issue_type="PB001",
                        relevant_item=2,
                        relevant_log_id=2,
                        match_score=45.0,
                        es_position=1,
                        model_feature_values="0.67",
                        result_position=1,
                    ),
                ],
                "boost_predict": ([1, 0], [[0.3, 0.7], [0.55, 0.45]]),
            },
            # Test case 9: Three hits
            {
                "test_calls": [get_index_found_call("1")],
                "msearch_results": three_hits_msearch,
                "test_item_info": test_item_info_w_logs,
                "expected_result": [
                    get_suggest_analysis_result(
                        issue_type="PB001",
                        relevant_item=3,
                        relevant_log_id=3,
                        match_score=80.0,
                        es_position=2,
                        model_feature_values="0.67",
                        result_position=0,
                    ),
                    get_suggest_analysis_result(
                        es_score=15.0,
                        result_position=1,
                    ),
                    get_suggest_analysis_result(
                        issue_type="PB001",
                        relevant_item=2,
                        relevant_log_id=2,
                        match_score=45.0,
                        es_position=1,
                        model_feature_values="0.67",
                        result_position=2,
                    ),
                ],
                "boost_predict": ([1, 0, 1], [[0.3, 0.7], [0.55, 0.45], [0.2, 0.8]]),
            },
            # Test case 10: Three hits with duplicate
            {
                "test_calls": [get_index_found_call("1")],
                "msearch_results": three_hits_with_duplicate_msearch,
                "test_item_info": test_item_info_w_logs,
                "expected_result": [
                    get_suggest_analysis_result(
                        relevant_item=3,
                        relevant_log_id=3,
                        es_score=15.0,
                        result_position=0,
                    ),
                    get_suggest_analysis_result(
                        es_score=15.0,
                        result_position=1,
                    ),
                    get_suggest_analysis_result(
                        issue_type="PB001",
                        relevant_item=2,
                        relevant_log_id=2,
                        es_position=1,
                        model_feature_values="0.67",
                        result_position=2,
                    ),
                ],
                "boost_predict": ([1, 1, 1], [[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]),
            },
            # Test case 11: Merged logs
            {
                "test_calls": [get_index_found_call("1")],
                "msearch_results": merged_logs_msearch,
                "test_item_info": test_item_info_w_merged_logs,
                "expected_result": [
                    get_suggest_analysis_result(
                        match_score=90.0,
                        is_merged_log=True,
                    )
                ],
                "boost_predict": ([1], [[0.1, 0.9]]),
            },
            # Test case 12: Merged logs with app config
            {
                "test_calls": [get_index_found_call("rp_1")],
                "app_config": APP_CONFIG,
                "msearch_results": merged_logs_msearch,
                "test_item_info": test_item_info_w_merged_logs,
                "expected_result": [
                    get_suggest_analysis_result(
                        match_score=90.0,
                        is_merged_log=True,
                    )
                ],
                "boost_predict": ([1], [[0.1, 0.9]]),
            },
            # Test case 13: Merged logs wrong
            {
                "test_calls": basic_suggest_test_calls,
                "msearch_results": merged_logs_wrong_msearch,
                "test_item_info": test_item_info_w_merged_logs,
                "expected_result": [],
                "boost_predict": ([], []),
            },
            # Test case 14: Merged logs wrong with app config
            {
                "test_calls": get_basic_suggest_test_calls("rp_1", self.index_logs_rs),
                "app_config": APP_CONFIG,
                "msearch_results": merged_logs_wrong_msearch,
                "test_item_info": test_item_info_w_merged_logs,
                "expected_result": [],
                "boost_predict": ([], []),
            },
            # Test case 15: Cluster test
            {
                "test_calls": cluster_test_calls,
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "test_item_info": test_item_info_cluster,
                "expected_result": [
                    get_suggest_analysis_result(
                        test_item=1,
                        test_item_log_id=1,
                        es_score=15.0,
                        is_merged_log=False,
                        cluster_id=5349085043832165,
                    ),
                    get_suggest_analysis_result(
                        test_item=1,
                        test_item_log_id=1,
                        issue_type="PB001",
                        relevant_item=2,
                        relevant_log_id=2,
                        match_score=45.0,
                        es_position=1,
                        model_feature_values="0.67",
                        result_position=1,
                        is_merged_log=False,
                        cluster_id=5349085043832165,
                    ),
                ],
                "boost_predict": ([1, 0], [[0.3, 0.7], [0.55, 0.45]]),
            },
            # Test case 16: Cluster small logs test
            {
                "test_calls": cluster_small_logs_test_calls,
                "msearch_results": merged_logs_msearch,
                "test_item_info": test_item_info_cluster,
                "expected_result": [
                    get_suggest_analysis_result(
                        test_item=1,
                        test_item_log_id=1,
                        match_score=75.0,
                        is_merged_log=True,
                        cluster_id=5349085043832165,
                    )
                ],
                "boost_predict": ([1], [[0.25, 0.75]]),
            },
        ]

        for idx, test in enumerate(tests):
            print(f"Running test case idx: {idx}")
            self._start_server(test["test_calls"])
            config = self.get_default_search_config()
            app_config = self.app_config
            if "app_config" in test:
                app_config = test["app_config"]
            suggest_service = SuggestService(self.model_chooser, app_config=app_config, search_cfg=config)
            suggest_service.es_client.es_client.scroll = MagicMock(
                return_value=json.loads(get_fixture(self.no_hits_search_rs))
            )
            if "msearch_results" in test:
                suggest_service.es_client.es_client.msearch = MagicMock(
                    return_value={"responses": test["msearch_results"]}
                )
            _boosting_decision_maker = BoostingDecisionMaker(object_saving.create_filesystem(""), "", features=[0])
            _boosting_decision_maker.predict = MagicMock(return_value=test["boost_predict"])
            suggest_service.model_chooser.choose_model = MagicMock(return_value=_boosting_decision_maker)
            response = suggest_service.suggest_items(test["test_item_info"])

            assert len(response) == len(test["expected_result"])
            for real_resp, expected_resp in zip(response, test["expected_result"]):
                real_resp.processedTime = 10.0
                assert real_resp == expected_resp

            TestSuggestService.shutdown_server(test["test_calls"])


if __name__ == "__main__":
    unittest.main()
