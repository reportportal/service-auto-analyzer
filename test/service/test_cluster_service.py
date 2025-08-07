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

import unittest

from freezegun import freeze_time

from app.service import ClusterService
from app.utils import utils
from test import APP_CONFIG
from test.mock_service import TestService
from test.service import (
    get_basic_cluster_test_calls,
    get_cluster_info,
    get_cluster_result,
    get_index_not_found_call,
    get_launch_from_fixture,
    get_launch_info_for_clustering,
    get_one_search_cluster_calls,
    get_simple_launch_info_for_clustering,
    get_three_search_cluster_calls,
    get_two_search_cluster_calls,
)


class TestClusterService(TestService):

    @freeze_time("2021-10-18 17:00:00")
    @utils.ignore_warnings
    def test_find_clusters(self):
        """Test finding clusters"""

        # Common test data - Launch objects from fixtures
        launch_clustering = get_launch_from_fixture(self.launch_w_items_clustering)
        launch_clustering_different_errors = get_launch_from_fixture(
            self.launch_w_items_clustering_with_different_errors
        )
        launch_small_logs = get_launch_from_fixture(self.launch_w_small_logs_for_clustering)
        launch_small_logs_numbers = get_launch_from_fixture(self.launch_w_small_logs_for_clustering_numbers)

        # Common cluster results
        empty_cluster_result_p1 = get_cluster_result(1, 1)
        empty_cluster_result_p2 = get_cluster_result(2, 1)

        # Complex cluster results with multiple ClusterInfo objects
        two_clusters_result = get_cluster_result(
            2,
            1,
            [
                get_cluster_info(21874152824769751, "error occurred\nerror found\nerror mined", [4, 5], [2, 5]),
                get_cluster_info(44972330576749361, "error occurred\nerror found\nassert query", [9], [6]),
            ],
        )

        single_cluster_result = get_cluster_result(
            2, 1, [get_cluster_info(48859729558090231, "error occurred\nerror found", [4, 5, 9], [2, 5, 6])]
        )

        es_update_result = get_cluster_result(
            2,
            1,
            [
                get_cluster_info(123, "error occurred \n error found \n error mined", [4, 5, 111], [2, 5]),
                get_cluster_info(44972330576749361, "error occurred\nerror found\nassert query", [9], [6]),
            ],
        )

        different_errors_result = get_cluster_result(
            2,
            1,
            [
                get_cluster_info(37711525315085941, "AssertionError error occurred\nerror found", [4], [2]),
                get_cluster_info(
                    83179189436345941, "AssertionError status code 500 error occurred\nerror found", [5], [5]
                ),
                get_cluster_info(90988898127574211, "NoSuchElementException error occurred\nerror found", [9], [6]),
            ],
        )

        small_logs_result = get_cluster_result(
            2,
            1,
            [
                get_cluster_info(
                    60604459849884091, "error occurred twice\nAssertionError error occurred\nerror found", [3, 4], [2]
                ),
                get_cluster_info(58202398056526781, "AssertionError status code 500 error occurred", [5], [5]),
                get_cluster_info(
                    86465058569810291, "NoSuchElementException error occurred\nerror found\nassert query", [9], [6]
                ),
            ],
        )

        small_logs_no_numbers_result = get_cluster_result(
            2,
            1,
            [
                get_cluster_info(
                    60604459849884090, "error occurred twice\nAssertionError error occurred\nerror found", [3, 4], [2]
                ),
                get_cluster_info(
                    3268144301345660, "AssertionError [EXCLUDED NUMBER] status code 500 error occurred", [5], [5]
                ),
                get_cluster_info(
                    86465058569810290, "NoSuchElementException error occurred\nerror found\nassert query", [9], [6]
                ),
            ],
        )

        tests = [
            # Test case 0: Basic index found - empty result
            {
                "test_calls": get_basic_cluster_test_calls("1"),
                "launch_info": get_simple_launch_info_for_clustering(1, 1),
                "expected_result": empty_cluster_result_p1,
            },
            # Test case 1: Index not found - empty result
            {
                "test_calls": [get_index_not_found_call("1")],
                "launch_info": get_simple_launch_info_for_clustering(1, 1),
                "expected_result": empty_cluster_result_p1,
            },
            # Test case 2: Basic index found with app config - empty result
            {
                "test_calls": get_basic_cluster_test_calls("rp_2"),
                "launch_info": get_simple_launch_info_for_clustering(1, 2),
                "app_config": APP_CONFIG,
                "expected_result": empty_cluster_result_p2,
            },
            # Test case 3: Two search operations - not for update
            {
                "test_calls": get_two_search_cluster_calls(
                    "2",
                    self.search_logs_rq_first_group_not_for_update,
                    self.no_hits_search_rs,
                    self.search_logs_rq_second_group_not_for_update,
                    self.no_hits_search_rs,
                    self.cluster_update,
                    self.index_logs_rs,
                ),
                "launch_info": get_launch_info_for_clustering(launch_clustering, 2),
                "expected_result": two_clusters_result,
            },
            # Test case 4: One search operation - 2 lines not for update
            {
                "test_calls": get_one_search_cluster_calls(
                    "2",
                    self.search_logs_rq_first_group_2lines_not_for_update,
                    self.no_hits_search_rs,
                    self.cluster_update_all_the_same,
                    self.index_logs_rs,
                ),
                "launch_info": get_launch_info_for_clustering(launch_clustering, 2, number_of_log_lines=2),
                "expected_result": single_cluster_result,
            },
            # Test case 5: Two search operations - for update
            {
                "test_calls": get_two_search_cluster_calls(
                    "2",
                    self.search_logs_rq_first_group,
                    self.no_hits_search_rs,
                    self.search_logs_rq_second_group,
                    self.no_hits_search_rs,
                    self.cluster_update,
                    self.index_logs_rs,
                ),
                "launch_info": get_launch_info_for_clustering(launch_clustering, 2, for_update=True),
                "expected_result": two_clusters_result,
            },
            # Test case 6: Two search operations with ES hits - for update
            {
                "test_calls": get_two_search_cluster_calls(
                    "2",
                    self.search_logs_rq_first_group,
                    self.one_hit_search_rs_clustering,
                    self.search_logs_rq_second_group,
                    self.one_hit_search_rs_clustering,
                    self.cluster_update_es_update,
                    self.index_logs_rs,
                ),
                "launch_info": get_launch_info_for_clustering(launch_clustering, 2, for_update=True),
                "expected_result": es_update_result,
            },
            # Test case 7: One search operation with ES hits - 2 lines for update
            {
                "test_calls": get_one_search_cluster_calls(
                    "2",
                    self.search_logs_rq_first_group_2lines,
                    self.one_hit_search_rs_clustering,
                    self.cluster_update_all_the_same_es_update,
                    self.index_logs_rs,
                ),
                "launch_info": get_launch_info_for_clustering(
                    launch_clustering, 2, for_update=True, number_of_log_lines=2
                ),
                "expected_result": single_cluster_result,
            },
            # Test case 8: One search operation with app config - 2 lines for update
            {
                "test_calls": get_one_search_cluster_calls(
                    "rp_2",
                    self.search_logs_rq_first_group_2lines,
                    self.one_hit_search_rs_clustering,
                    self.cluster_update_all_the_same_es_update_with_prefix,
                    self.index_logs_rs,
                ),
                "launch_info": get_launch_info_for_clustering(
                    launch_clustering, 2, for_update=True, number_of_log_lines=2
                ),
                "app_config": APP_CONFIG,
                "expected_result": single_cluster_result,
            },
            # Test case 9: Three search operations with different errors - 2 lines not for update
            {
                "test_calls": get_three_search_cluster_calls(
                    "2",
                    self.search_logs_rq_first_group_assertion_error,
                    self.no_hits_search_rs,
                    self.search_logs_rq_first_group_assertion_error_status_code,
                    self.no_hits_search_rs,
                    self.search_logs_rq_first_group_no_such_element,
                    self.no_hits_search_rs,
                    self.cluster_update_all_the_same_es_with_different_errors,
                    self.index_logs_rs,
                ),
                "launch_info": get_launch_info_for_clustering(
                    launch_clustering_different_errors, 2, number_of_log_lines=2
                ),
                "expected_result": different_errors_result,
            },
            # Test case 10: Three search operations with small logs
            {
                "test_calls": get_three_search_cluster_calls(
                    "2",
                    self.search_logs_rq_first_group_small_logs,
                    self.no_hits_search_rs,
                    self.search_logs_rq_second_group_small_logs,
                    self.no_hits_search_rs,
                    self.search_logs_rq_first_group_no_such_element_all_log_lines,
                    self.no_hits_search_rs,
                    self.cluster_update_small_logs,
                    self.index_logs_rs,
                ),
                "launch_info": get_launch_info_for_clustering(launch_small_logs, 2),
                "expected_result": small_logs_result,
            },
            # Test case 11: Three search operations with small logs and clean numbers
            {
                "test_calls": get_three_search_cluster_calls(
                    "2",
                    self.search_logs_rq_first_group_small_logs_no_numbers,
                    self.no_hits_search_rs,
                    self.search_logs_rq_second_group_small_logs_no_numbers,
                    self.no_hits_search_rs,
                    self.search_logs_rq_first_group_no_such_element_no_numbers_all_log_lines,
                    self.no_hits_search_rs,
                    self.cluster_update_small_logs_no_numbers,
                    self.index_logs_rs,
                ),
                "launch_info": get_launch_info_for_clustering(launch_small_logs_numbers, 2, clean_numbers=True),
                "expected_result": small_logs_no_numbers_result,
            },
        ]

        for idx, test in enumerate(tests):
            print(f"Test case number: {idx}")
            self._start_server(test["test_calls"])
            config = self.get_default_search_config()
            app_config = self.app_config
            if "app_config" in test:
                app_config = test["app_config"]
            _cluster_service = ClusterService(app_config=app_config, search_cfg=config)

            response = _cluster_service.find_clusters(test["launch_info"])

            assert len(response.clusters) == len(test["expected_result"].clusters)
            assert response == test["expected_result"]

            TestClusterService.shutdown_server(test["test_calls"])


if __name__ == "__main__":
    unittest.main()
