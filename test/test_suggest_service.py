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
from unittest.mock import MagicMock
import json
from http import HTTPStatus
import logging
import sure # noqa
import httpretty

import commons.launch_objects as launch_objects
from boosting_decision_making.boosting_decision_maker import BoostingDecisionMaker
from service.suggest_service import SuggestService
from utils import utils


class TestSuggestService(unittest.TestCase):
    """Tests analyzer service functionality"""

    ERROR_LOGGING_LEVEL = 40000

    @utils.ignore_warnings
    def setUp(self):
        self.two_indices_rs = "two_indices_rs.json"
        self.index_created_rs = "index_created_rs.json"
        self.index_already_exists_rs = "index_already_exists_rs.json"
        self.index_deleted_rs = "index_deleted_rs.json"
        self.index_not_found_rs = "index_not_found_rs.json"
        self.launch_wo_test_items = "launch_wo_test_items.json"
        self.launch_w_test_items_wo_logs = "launch_w_test_items_wo_logs.json"
        self.launch_w_test_items_w_logs = "launch_w_test_items_w_logs.json"
        self.launch_w_test_items_w_empty_logs = "launch_w_test_items_w_empty_logs.json"
        self.launch_w_test_items_w_logs_to_be_merged =\
            "launch_w_test_items_w_logs_to_be_merged.json"
        self.index_logs_rq = "index_logs_rq.json"
        self.index_logs_rq_big_messages = "index_logs_rq_big_messages.json"
        self.index_logs_rs = "index_logs_rs.json"
        self.search_rq_first = "search_rq_first.json"
        self.search_rq_second = "search_rq_second.json"
        self.search_rq_third = "search_rq_third.json"
        self.search_rq_filtered = "search_rq_filtered.json"
        self.search_rq_another_log = "search_rq_another_log.json"
        self.search_rq_different_logs = "search_rq_different_logs.json"
        self.search_rq_to_be_merged = "search_rq_to_be_merged.json"
        self.no_hits_search_rs = "no_hits_search_rs.json"
        self.one_hit_search_rs = "one_hit_search_rs.json"
        self.one_hit_search_rs_search_logs = "one_hit_search_rs_search_logs.json"
        self.two_hits_search_rs = "two_hits_search_rs.json"
        self.two_hits_search_rs_search_logs = "two_hits_search_rs_search_logs.json"
        self.three_hits_search_rs = "three_hits_search_rs.json"
        self.launch_w_test_items_w_logs_different_log_level =\
            "launch_w_test_items_w_logs_different_log_level.json"
        self.index_logs_rq_different_log_level = "index_logs_rq_different_log_level.json"
        self.index_logs_rq_different_log_level_merged =\
            "index_logs_rq_different_log_level_merged.json"
        self.index_logs_rs_different_log_level = "index_logs_rs_different_log_level.json"
        self.delete_logs_rs = "delete_logs_rs.json"
        self.two_hits_search_with_big_messages_rs = "two_hits_search_with_big_messages_rs.json"
        self.search_not_merged_logs_for_delete = "search_not_merged_logs_for_delete.json"
        self.search_merged_logs = "search_merged_logs.json"
        self.search_not_merged_logs = "search_not_merged_logs.json"
        self.search_logs_rq = "search_logs_rq.json"
        self.search_logs_rq_not_found = "search_logs_rq_not_found.json"
        self.index_logs_rq_merged_logs = "index_logs_rq_merged_logs.json"
        self.suggest_test_item_info_w_logs = "suggest_test_item_info_w_logs.json"
        self.three_hits_search_rs_with_duplicate = "three_hits_search_rs_with_duplicate.json"
        self.one_hit_search_rs_merged = "one_hit_search_rs_merged.json"
        self.search_rq_merged_first = "search_rq_merged_first.json"
        self.search_rq_merged_second = "search_rq_merged_second.json"
        self.search_rq_merged_third = "search_rq_merged_third.json"
        self.suggest_test_item_info_w_merged_logs = "suggest_test_item_info_w_merged_logs.json"
        self.one_hit_search_rs_merged_wrong = "one_hit_search_rs_merged_wrong.json"
        self.three_hits_search_rs_with_one_unique_id = "three_hits_search_rs_with_one_unique_id.json"
        self.launch_w_items_clustering = "launch_w_items_clustering.json"
        self.cluster_update_all_the_same = "cluster_update_all_the_same.json"
        self.search_logs_rq_first_group = "search_logs_rq_first_group.json"
        self.search_logs_rq_second_group = "search_logs_rq_second_group.json"
        self.one_hit_search_rs_clustering = "one_hit_search_rs_clustering.json"
        self.search_logs_rq_first_group_2lines = "search_logs_rq_first_group_2lines.json"
        self.cluster_update_es_update = "cluster_update_es_update.json"
        self.cluster_update_all_the_same_es_update = "cluster_update_all_the_same_es_update.json"
        self.cluster_update = "cluster_update.json"
        self.app_config = {
            "esHost": "http://localhost:9200",
            "esVerifyCerts":     False,
            "esUseSsl":          False,
            "esSslShowWarn":     False,
            "esCAcert":          "",
            "esClientCert":      "",
            "esClientKey":       "",
            "appVersion":        ""
        }
        self.model_settings = utils.read_json_file("", "model_settings.json", to_json=True)
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @utils.ignore_warnings
    def get_default_search_config(self):
        """Get default search config"""
        return {
            "MinShouldMatch": "80%",
            "MinTermFreq":    1,
            "MinDocFreq":     1,
            "BoostAA": -2,
            "BoostLaunch":    2,
            "BoostUniqueID":  2,
            "MaxQueryTerms":  50,
            "SearchLogsMinShouldMatch": "98%",
            "SearchLogsMinSimilarity": 0.9,
            "MinWordLength":  0,
            "BoostModelFolder":
                self.model_settings["BOOST_MODEL_FOLDER"],
            "SimilarityWeightsFolder":
                self.model_settings["SIMILARITY_WEIGHTS_FOLDER"],
            "SuggestBoostModelFolder":
                self.model_settings["SUGGEST_BOOST_MODEL_FOLDER"],
            "GlobalDefectTypeModelFolder":
                self.model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"]
        }

    @utils.ignore_warnings
    def _start_server(self, test_calls):
        httpretty.reset()
        httpretty.enable(allow_net_connect=False)
        for test_info in test_calls:
            if "content_type" in test_info:
                httpretty.register_uri(
                    test_info["method"],
                    self.app_config["esHost"] + test_info["uri"],
                    body=test_info["rs"] if "rs" in test_info else "",
                    status=test_info["status"],
                    content_type=test_info["content_type"],
                )
            else:
                httpretty.register_uri(
                    test_info["method"],
                    self.app_config["esHost"] + test_info["uri"],
                    body=test_info["rs"] if "rs" in test_info else "",
                    status=test_info["status"],
                )

    @staticmethod
    @utils.ignore_warnings
    def shutdown_server(test_calls):
        """Shutdown server and test request calls"""
        httpretty.latest_requests().should.have.length_of(len(test_calls))
        for expected_test_call, test_call in zip(test_calls, httpretty.latest_requests()):
            expected_test_call["method"].should.equal(test_call.method)
            expected_test_call["uri"].should.equal(test_call.path)
            if "rq" in expected_test_call:
                expected_body = expected_test_call["rq"]
                real_body = test_call.parse_request_body(test_call.body)
                if type(expected_body) == str and type(real_body) != str:
                    expected_body = json.loads(expected_body)
                expected_body.should.equal(real_body)
        httpretty.disable()
        httpretty.reset()

    @utils.ignore_warnings
    def test_suggest_items(self):
        """Test suggesting test items"""
        tests = [
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.OK,
                                         }, ],
                "test_item_info":      launch_objects.TestItemInfo(testItemId=1,
                                                                   uniqueId="341",
                                                                   testCaseHash=123,
                                                                   launchId=1,
                                                                   launchName="Launch",
                                                                   project=1,
                                                                   logs=[]),
                "expected_result":     [],
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    }, ],
                "test_item_info": launch_objects.TestItemInfo(testItemId=1,
                                                              uniqueId="341",
                                                              testCaseHash=123,
                                                              launchId=1,
                                                              launchName="Launch",
                                                              project=2,
                                                              logs=[launch_objects.Log(
                                                                    logId=1,
                                                                    message="error found",
                                                                    logLevel=40000)]),
                "expected_result":     [],
                "boost_predict":       ([], [])
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.OK,
                                         }, ],
                "test_item_info":      launch_objects.TestItemInfo(testItemId=1,
                                                                   uniqueId="341",
                                                                   testCaseHash=123,
                                                                   launchId=1,
                                                                   launchName="Launch",
                                                                   project=1,
                                                                   logs=[launch_objects.Log(
                                                                         logId=1,
                                                                         message=" ",
                                                                         logLevel=40000)]),
                "expected_result":     [],
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_first),
                                    "rs":           utils.get_fixture(
                                        self.no_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_second),
                                    "rs":           utils.get_fixture(
                                        self.no_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_third),
                                    "rs":           utils.get_fixture(
                                        self.no_hits_search_rs),
                                    }],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [],
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_first),
                                    "rs":           utils.get_fixture(
                                        self.no_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_second),
                                    "rs":           utils.get_fixture(
                                        self.no_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_third),
                                    "rs":           utils.get_fixture(
                                        self.no_hits_search_rs),
                                    }],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [],
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_first),
                                    "rs":           utils.get_fixture(
                                        self.no_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_second),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_third),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=80.0,
                                                         esScore=10.0,
                                                         esPosition=0,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='1.0',
                                                         modelInfo='',
                                                         resultPosition=0,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80)],
                "boost_predict":       ([1], [[0.2, 0.8]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_first),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_second),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_third),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=70.0,
                                                         esScore=10.0,
                                                         esPosition=0,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='1.0',
                                                         modelInfo='',
                                                         resultPosition=0,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80)],
                "boost_predict":       ([1], [[0.3, 0.7]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_first),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_second),
                                    "rs":           utils.get_fixture(
                                        self.two_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_third),
                                    "rs":           utils.get_fixture(
                                        self.two_hits_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=70.0,
                                                         esScore=15.0,
                                                         esPosition=0,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='1.0',
                                                         modelInfo='',
                                                         resultPosition=0,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80)],
                "boost_predict":       ([1, 0], [[0.3, 0.7], [0.9, 0.1]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_first),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_second),
                                    "rs":           utils.get_fixture(
                                        self.two_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_third),
                                    "rs":           utils.get_fixture(
                                        self.no_hits_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=70.0,
                                                         esScore=15.0,
                                                         esPosition=0,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='1.0',
                                                         modelInfo='',
                                                         resultPosition=0,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80),
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='PB001',
                                                         relevantItem=2,
                                                         relevantLogId=2,
                                                         matchScore=45.0,
                                                         esScore=10.0,
                                                         esPosition=1,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='0.67',
                                                         modelInfo='',
                                                         resultPosition=1,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80)],
                "boost_predict":       ([1, 0], [[0.3, 0.7], [0.55, 0.45]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_first),
                                    "rs":           utils.get_fixture(
                                        self.two_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_second),
                                    "rs":           utils.get_fixture(
                                        self.three_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_third),
                                    "rs":           utils.get_fixture(
                                        self.no_hits_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='PB001',
                                                         relevantItem=3,
                                                         relevantLogId=3,
                                                         matchScore=80.0,
                                                         esScore=10.0,
                                                         esPosition=2,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='0.67',
                                                         modelInfo='',
                                                         resultPosition=0,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80),
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=70.0,
                                                         esScore=15.0,
                                                         esPosition=0,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='1.0',
                                                         modelInfo='',
                                                         resultPosition=1,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80),
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='PB001',
                                                         relevantItem=2,
                                                         relevantLogId=2,
                                                         matchScore=45.0,
                                                         esScore=10.0,
                                                         esPosition=1,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='0.67',
                                                         modelInfo='',
                                                         resultPosition=2,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80)],
                "boost_predict":       ([1, 0, 1], [[0.3, 0.7], [0.55, 0.45], [0.2, 0.8]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_first),
                                    "rs":           utils.get_fixture(
                                        self.two_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_second),
                                    "rs":           utils.get_fixture(
                                        self.three_hits_search_rs_with_duplicate),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_third),
                                    "rs":           utils.get_fixture(
                                        self.no_hits_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='AB001',
                                                         relevantItem=3,
                                                         relevantLogId=3,
                                                         matchScore=70.0,
                                                         esScore=15.0,
                                                         esPosition=0,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='1.0',
                                                         modelInfo='',
                                                         resultPosition=0,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80),
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=70.0,
                                                         esScore=15.0,
                                                         esPosition=0,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='1.0',
                                                         modelInfo='',
                                                         resultPosition=1,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80),
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=178,
                                                         issueType='PB001',
                                                         relevantItem=2,
                                                         relevantLogId=2,
                                                         matchScore=70.0,
                                                         esScore=10.0,
                                                         esPosition=1,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='0.67',
                                                         modelInfo='',
                                                         resultPosition=2,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80)],
                "boost_predict":       ([1, 1, 1], [[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_first),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_second),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_third),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         testItemLogId=-1,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=90.0,
                                                         esScore=10.0,
                                                         esPosition=0,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='1.0',
                                                         modelInfo='',
                                                         resultPosition=0,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80)],
                "boost_predict":       ([1], [[0.1, 0.9]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_first),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged_wrong),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_second),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged_wrong),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_third),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged_wrong),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)),
                "expected_result":     [],
                "boost_predict":       ([], [])
            },
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                config = self.get_default_search_config()
                suggest_service = SuggestService(app_config=self.app_config,
                                                 search_cfg=config)
                _boosting_decision_maker = BoostingDecisionMaker()
                _boosting_decision_maker.get_feature_ids = MagicMock(return_value=[0])
                _boosting_decision_maker.predict = MagicMock(return_value=test["boost_predict"])
                suggest_service.suggest_decision_maker = _boosting_decision_maker
                response = suggest_service.suggest_items(test["test_item_info"])

                response.should.have.length_of(len(test["expected_result"]))
                for real_resp, expected_resp in zip(response, test["expected_result"]):
                    real_resp.should.equal(expected_resp)

                TestSuggestService.shutdown_server(test["test_calls"])


if __name__ == '__main__':
    unittest.main()
