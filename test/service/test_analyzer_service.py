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
from unittest.mock import MagicMock

from app.commons import object_saving
from app.machine_learning.models.boosting_decision_maker import BoostingDecisionMaker
from app.service import AutoAnalyzerService
from app.utils import utils
from test import APP_CONFIG, get_fixture
from test.mock_service import TestService
from test.service import (
    get_analyzer_config_all_messages_match,
    get_analyzer_index_call,
    get_analyzer_index_not_found_call,
    get_basic_boost_predict_empty,
    get_boost_predict_all_match,
    get_boost_predict_double_ab001,
    get_boost_predict_mixed_ab001,
    get_boost_predict_nd001,
    get_boost_predict_no_nd001,
    get_boost_predict_pb001_complex,
    get_boost_predict_single_ab001,
    get_boost_predict_single_pb001,
    get_launch_objects_from_fixture,
    get_launch_objects_with_analyzer_config,
    get_mixed_hits_msearch_results,
    get_nd_issue_msearch_results,
    get_no_hits_msearch_results,
    get_pb_issue_msearch_results,
    get_three_hits_msearch_results,
    get_two_hits_msearch_results,
)


class TestAutoAnalyzerService(TestService):

    @utils.ignore_warnings
    def test_analyze_logs(self):
        """Test analyzing logs"""

        # Common msearch results using helper functions
        no_hits_msearch = get_no_hits_msearch_results(self.no_hits_search_rs)
        mixed_hits_msearch = get_mixed_hits_msearch_results(self.no_hits_search_rs, self.one_hit_search_rs)
        two_hits_msearch = get_two_hits_msearch_results(
            self.one_hit_search_rs, self.two_hits_search_rs, self.no_hits_search_rs
        )
        three_hits_msearch = get_three_hits_msearch_results(
            self.two_hits_search_rs, self.three_hits_search_rs, self.no_hits_search_rs
        )
        pb_issue_msearch = get_pb_issue_msearch_results(self.no_hits_search_rs, self.three_hits_search_rs)
        nd_issue_msearch = get_nd_issue_msearch_results(
            self.no_hits_search_rs, self.two_hits_search_rs, self.two_hits_with_no_defect
        )
        nd_issue_no_result_msearch = get_nd_issue_msearch_results(
            self.no_hits_search_rs, self.two_hits_search_rs, self.two_hits_with_no_defect
        )
        pb_complex_msearch = get_nd_issue_msearch_results(
            self.no_hits_search_rs, self.three_hits_search_rs, self.three_hits_with_no_defect
        )

        # Custom msearch results for specific cases
        merge_logs_msearch = [
            get_fixture(self.two_hits_search_rs, to_json=True),
            get_fixture(self.two_hits_search_rs, to_json=True),
        ]
        unique_id_msearch = [
            get_fixture(self.no_hits_search_rs, to_json=True),
            get_fixture(self.no_hits_search_rs, to_json=True),
            get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True),
            get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True),
        ]
        all_match_msearch = [
            get_fixture(self.two_hits_search_rs_second_message, to_json=True),
            get_fixture(self.two_hits_search_rs_second_message, to_json=True),
            get_fixture(self.two_hits_search_rs, to_json=True),
            get_fixture(self.two_hits_search_rs, to_json=True),
        ]

        # Analyzer config
        all_messages_match_config = get_analyzer_config_all_messages_match()

        tests = [
            # Test case 0: Launch without test items
            {
                "test_calls": [get_analyzer_index_call("1")],
                "index_rq": get_fixture(self.launch_wo_test_items),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": get_basic_boost_predict_empty(),
            },
            # Test case 1: Launch with test items but no logs
            {
                "test_calls": [get_analyzer_index_call("1")],
                "index_rq": get_fixture(self.launch_w_test_items_wo_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": get_basic_boost_predict_empty(),
            },
            # Test case 2: Launch with test items with empty logs
            {
                "test_calls": [get_analyzer_index_call("2")],
                "index_rq": get_fixture(self.launch_w_test_items_w_empty_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": get_basic_boost_predict_empty(),
            },
            # Test case 3: Launch with test items with logs but no search results
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": no_hits_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": get_basic_boost_predict_empty(),
            },
            # Test case 4: Index not found
            {
                "test_calls": [get_analyzer_index_not_found_call("2")],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": get_basic_boost_predict_empty(),
            },
            # Test case 5: Mixed hits leading to AB001
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": mixed_hits_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict": get_boost_predict_single_ab001(),
            },
            # Test case 6: Two hits leading to AB001
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": two_hits_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict": get_boost_predict_mixed_ab001(),
            },
            # Test case 7: Three hits leading to AB001
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": three_hits_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict": get_boost_predict_double_ab001(),
            },
            # Test case 8: PB001 issue type
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": pb_issue_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "PB001",
                "boost_predict": get_boost_predict_single_pb001(),
            },
            # Test case 9: AB001 with unique ID
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": unique_id_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict": get_boost_predict_mixed_ab001(),
            },
            # Test case 10: Merged logs scenario
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": merge_logs_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs_to_be_merged),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": get_basic_boost_predict_empty(),
            },
            # Test case 11: Unique ID with mixed AB001
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": unique_id_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict": get_boost_predict_mixed_ab001(),
            },
            # Test case 12: With app config
            {
                "test_calls": [get_analyzer_index_call("rp_2")],
                "msearch_results": unique_id_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "app_config": APP_CONFIG,
                "boost_predict": get_boost_predict_mixed_ab001(),
            },
            # Test case 13: ND001 issue type
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": nd_issue_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_id": 34,
                "expected_issue_type": "ND001",
                "boost_predict": get_boost_predict_nd001(),
            },
            # Test case 14: ND001 rejected
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": nd_issue_no_result_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": get_boost_predict_no_nd001(),
            },
            # Test case 15: PB001 complex
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": pb_complex_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_id": 2,
                "expected_issue_type": "PB001",
                "boost_predict": get_boost_predict_pb001_complex(),
            },
            # Test case 16: All messages should match - no results
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": unique_id_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "analyzer_config": all_messages_match_config,
                "boost_predict": get_basic_boost_predict_empty(),
            },
            # Test case 17: All messages should match - with results
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": all_match_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "analyzer_config": all_messages_match_config,
                "boost_predict": get_boost_predict_all_match(),
            },
            # Test case 18: All messages should match - ND001 rejected
            {
                "test_calls": [get_analyzer_index_call("2")],
                "msearch_results": nd_issue_msearch,
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "analyzer_config": all_messages_match_config,
                "boost_predict": get_basic_boost_predict_empty(),
            },
        ]

        for idx, test in enumerate(tests):
            print(f"Running test case idx: {idx}")
            self._start_server(test["test_calls"])
            config = self.get_default_search_config()
            app_config = self.app_config
            if "app_config" in test:
                app_config = test["app_config"]
            analyzer_service = AutoAnalyzerService(self.model_chooser, app_config=app_config, search_cfg=config)
            _boosting_decision_maker = BoostingDecisionMaker(object_saving.create_filesystem(""), "", features=[0])
            _boosting_decision_maker.predict = MagicMock(return_value=test["boost_predict"])
            if "msearch_results" in test:
                analyzer_service.es_client.es_client.msearch = MagicMock(
                    return_value={"responses": test["msearch_results"]}
                )
            analyzer_service.model_chooser.choose_model = MagicMock(return_value=_boosting_decision_maker)

            launches = get_launch_objects_from_fixture(test["index_rq"])
            if "analyzer_config" in test:
                launches = get_launch_objects_with_analyzer_config(test["index_rq"], test["analyzer_config"])
            response = analyzer_service.analyze_logs(launches)

            assert len(response) == test["expected_count"]

            if test["expected_issue_type"] != "":
                assert response[0].issueType == test["expected_issue_type"]

            if "expected_id" in test:
                assert response[0].relevantItem == test["expected_id"]

            TestAutoAnalyzerService.shutdown_server(test["test_calls"])


if __name__ == "__main__":
    unittest.main()
