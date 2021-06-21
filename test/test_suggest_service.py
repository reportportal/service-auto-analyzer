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
import json
from unittest.mock import MagicMock
from http import HTTPStatus
import sure # noqa
import httpretty

import commons.launch_objects as launch_objects
from boosting_decision_making.boosting_decision_maker import BoostingDecisionMaker
from service.suggest_service import SuggestService
from test.test_service import TestService
from utils import utils


class TestSuggestService(TestService):

    @utils.ignore_warnings
    def test_suggest_items(self):
        """Test suggesting test items"""
        tests = [
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.OK,
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/rp_suggestions_info_metrics",
                                         "status":         HTTPStatus.OK
                                         },
                                        {"method":         httpretty.PUT,
                                         "uri":            "/rp_suggestions_info_metrics/_mapping",
                                         "status":         HTTPStatus.OK,
                                         "rs":             utils.get_fixture(self.index_created_rs),
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=true",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rs":             utils.get_fixture(self.index_logs_rs),
                                         }],
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
                                    }],
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
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/rp_suggestions_info_metrics",
                                         "status":         HTTPStatus.OK
                                         },
                                        {"method":         httpretty.PUT,
                                         "uri":            "/rp_suggestions_info_metrics/_mapping",
                                         "status":         HTTPStatus.OK,
                                         "rs":             utils.get_fixture(self.index_created_rs),
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=true",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rs":             utils.get_fixture(self.index_logs_rs),
                                         }],
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
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_suggestions_info_metrics",
                                    "status":         HTTPStatus.OK
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/rp_suggestions_info_metrics/_mapping",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
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
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_suggestions_info_metrics",
                                    "status":         HTTPStatus.OK
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/rp_suggestions_info_metrics/_mapping",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
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
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest")],
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
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest")],
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
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest")],
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
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest"),
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest")],
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
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest"),
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest"),
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest")],
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
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest"),
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest"),
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
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
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest")],
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
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         isMergedLog=True,
                                                         matchScore=90.0,
                                                         esScore=10.0,
                                                         esPosition=0,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='1.0',
                                                         modelInfo='',
                                                         resultPosition=0,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest")],
                "boost_predict":       ([1], [[0.1, 0.9]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/rp_1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_first),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/rp_1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_second),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/rp_1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_third),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged),
                                    }, ],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts":     False,
                    "esUseSsl":          False,
                    "esSslShowWarn":     False,
                    "turnOffSslVerification": True,
                    "esCAcert":          "",
                    "esClientCert":      "",
                    "esClientKey":       "",
                    "appVersion":        "",
                    "minioRegion":       "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber":     1000,
                    "binaryStoreType":   "minio",
                    "minioHost":         "",
                    "minioAccessKey":    "",
                    "minioSecretKey":    "",
                    "esProjectIndexPrefix": "rp_"
                },
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(project=1,
                                                         testItem=123,
                                                         testItemLogId=178,
                                                         launchId=145,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         isMergedLog=True,
                                                         matchScore=90.0,
                                                         esScore=10.0,
                                                         esPosition=0,
                                                         modelFeatureNames='0',
                                                         modelFeatureValues='1.0',
                                                         modelInfo='',
                                                         resultPosition=0,
                                                         usedLogLines=-1,
                                                         minShouldMatch=80,
                                                         processedTime=10.0,
                                                         methodName="suggest")],
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
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_suggestions_info_metrics",
                                    "status":         HTTPStatus.OK
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/rp_suggestions_info_metrics/_mapping",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
                                    }],
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)),
                "expected_result":     [],
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/rp_1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_first),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged_wrong),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/rp_1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_second),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged_wrong),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/rp_1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           utils.get_fixture(self.search_rq_merged_third),
                                    "rs":           utils.get_fixture(
                                        self.one_hit_search_rs_merged_wrong),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_suggestions_info_metrics",
                                    "status":         HTTPStatus.OK
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/rp_suggestions_info_metrics/_mapping",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
                                    }],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts":     False,
                    "esUseSsl":          False,
                    "esSslShowWarn":     False,
                    "turnOffSslVerification": True,
                    "esCAcert":          "",
                    "esClientCert":      "",
                    "esClientKey":       "",
                    "appVersion":        "",
                    "minioRegion":       "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber":     1000,
                    "binaryStoreType":   "minio",
                    "minioHost":         "",
                    "minioAccessKey":    "",
                    "minioSecretKey":    "",
                    "esProjectIndexPrefix": "rp_"
                },
                "test_item_info":      launch_objects.TestItemInfo(
                    **utils.get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)),
                "expected_result":     [],
                "boost_predict":       ([], [])
            }
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                config = self.get_default_search_config()
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_service = SuggestService(app_config=app_config,
                                                 search_cfg=config)
                _boosting_decision_maker = BoostingDecisionMaker()
                _boosting_decision_maker.get_feature_ids = MagicMock(return_value=[0])
                _boosting_decision_maker.predict = MagicMock(return_value=test["boost_predict"])
                suggest_service.model_chooser.choose_model = MagicMock(
                    return_value=_boosting_decision_maker)
                response = suggest_service.suggest_items(test["test_item_info"])

                response.should.have.length_of(len(test["expected_result"]))
                for real_resp, expected_resp in zip(response, test["expected_result"]):
                    real_resp.processedTime = 10.0
                    real_resp.should.equal(expected_resp)

                TestSuggestService.shutdown_server(test["test_calls"])

    @utils.ignore_warnings
    def test_clean_suggest_info_logs(self):
        """Test cleaning suggest info logs"""
        tests = [
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2_suggest",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    }, ],
                "rq":             launch_objects.CleanIndex(ids=[1], project=2),
                "expected_count": 0
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1_suggest/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_suggest_info_ids_query),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_suggest_info_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.delete_suggest_logs_rq),
                                    "rs":             utils.get_fixture(self.delete_logs_rs),
                                    }],
                "rq":             launch_objects.CleanIndex(ids=[1], project=1),
                "expected_count": 1
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_1_suggest",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_1_suggest/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_suggest_info_ids_query),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_suggest_info_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.delete_suggest_logs_rq_with_prefix),
                                    "rs":             utils.get_fixture(self.delete_logs_rs),
                                    }],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts":     False,
                    "esUseSsl":          False,
                    "esSslShowWarn":     False,
                    "turnOffSslVerification": True,
                    "esCAcert":          "",
                    "esClientCert":      "",
                    "esClientKey":       "",
                    "appVersion":        "",
                    "minioRegion":       "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber":     1000,
                    "binaryStoreType":   "minio",
                    "minioHost":         "",
                    "minioAccessKey":    "",
                    "minioSecretKey":    "",
                    "esProjectIndexPrefix": "rp_"
                },
                "rq":             launch_objects.CleanIndex(ids=[1], project=1),
                "expected_count": 1
            }
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_service = SuggestService(app_config=app_config,
                                                 search_cfg=self.get_default_search_config())

                suggest_service.es_client.es_client.scroll = MagicMock(
                    return_value=json.loads(utils.get_fixture(self.no_hits_search_rs)))
                response = suggest_service.clean_suggest_info_logs(test["rq"])
                test["expected_count"].should.equal(response)
                TestSuggestService.shutdown_server(test["test_calls"])

    @utils.ignore_warnings
    def test_delete_suggest_info_index(self):
        """Test deleting an index"""
        tests = [
            {
                "test_calls": [{"method":         httpretty.DELETE,
                                "uri":            "/1_suggest",
                                "status":         HTTPStatus.OK,
                                "content_type":   "application/json",
                                "rs":             utils.get_fixture(self.index_deleted_rs),
                                }, ],
                "index":      1,
                "result":     True,
            },
            {
                "test_calls": [{"method":         httpretty.DELETE,
                                "uri":            "/2_suggest",
                                "status":         HTTPStatus.NOT_FOUND,
                                "content_type":   "application/json",
                                "rs":             utils.get_fixture(self.index_not_found_rs),
                                }, ],
                "index":      2,
                "result":     False,
            },
            {
                "test_calls": [{"method":         httpretty.DELETE,
                                "uri":            "/rp_2_suggest",
                                "status":         HTTPStatus.NOT_FOUND,
                                "content_type":   "application/json",
                                "rs":             utils.get_fixture(self.index_not_found_rs),
                                }, ],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts":     False,
                    "esUseSsl":          False,
                    "esSslShowWarn":     False,
                    "turnOffSslVerification": True,
                    "esCAcert":          "",
                    "esClientCert":      "",
                    "esClientKey":       "",
                    "appVersion":        "",
                    "minioRegion":       "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber":     1000,
                    "binaryStoreType":   "minio",
                    "minioHost":         "",
                    "minioAccessKey":    "",
                    "minioSecretKey":    "",
                    "esProjectIndexPrefix": "rp_"
                },
                "index":      2,
                "result":     False,
            }
        ]
        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_service = SuggestService(app_config=app_config,
                                                 search_cfg=self.get_default_search_config())

                response = suggest_service.remove_suggest_info(test["index"])

                test["result"].should.equal(response)

                TestSuggestService.shutdown_server(test["test_calls"])

    @utils.ignore_warnings
    def test_index_suggest_info_logs(self):
        """Test indexing suggest info"""
        tests = [
            {
                "test_calls":     [],
                "index_rq":       "[]",
                "has_errors":     False,
                "expected_count": 0
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_suggestions_info_metrics",
                                    "status":         HTTPStatus.NOT_FOUND
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/rp_suggestions_info_metrics",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(self.index_created_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.NOT_FOUND
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
                                    }],
                "index_rq":       utils.get_fixture(self.suggest_info_list),
                "has_errors":     False,
                "expected_count": 2
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_suggestions_info_metrics",
                                    "status":         HTTPStatus.OK
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/rp_suggestions_info_metrics/_mapping",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(self.index_created_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.OK
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/1_suggest/_mapping",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
                                    }],
                "index_rq":       utils.get_fixture(self.suggest_info_list),
                "has_errors":     False,
                "expected_count": 2
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_suggestions_info_metrics",
                                    "status":         HTTPStatus.OK
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/rp_suggestions_info_metrics/_mapping",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(self.index_created_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_1_suggest",
                                    "status":         HTTPStatus.OK
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/rp_1_suggest/_mapping",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
                                    }],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts":     False,
                    "esUseSsl":          False,
                    "esSslShowWarn":     False,
                    "turnOffSslVerification": True,
                    "esCAcert":          "",
                    "esClientCert":      "",
                    "esClientKey":       "",
                    "appVersion":        "",
                    "minioRegion":       "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber":     1000,
                    "binaryStoreType":   "minio",
                    "minioHost":         "",
                    "minioAccessKey":    "",
                    "minioSecretKey":    "",
                    "esProjectIndexPrefix": "rp_"
                },
                "index_rq":       utils.get_fixture(self.suggest_info_list),
                "has_errors":     False,
                "expected_count": 2
            }
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_service = SuggestService(app_config=app_config,
                                                 search_cfg=self.get_default_search_config())
                response = suggest_service.index_suggest_info(
                    [launch_objects.SuggestAnalysisResult(**res) for res in json.loads(test["index_rq"])])

                test["has_errors"].should.equal(response.errors)
                test["expected_count"].should.equal(response.took)

                TestSuggestService.shutdown_server(test["test_calls"])

    def test_remove_test_items_suggests(self):
        tests = [
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    "content_type":   "application/json",
                                    }],
                "item_remove_info": {
                    "project": 1,
                    "itemsToDelete": [1, 2]},
                "result":     0
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/1_suggest/_delete_by_query",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.delete_by_query_suggest_1),
                                    "rs":             json.dumps({"deleted": 1})}],
                "item_remove_info": {
                    "project": 1,
                    "itemsToDelete": [1, 2]},
                "result":     1
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_1_suggest",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/rp_1_suggest/_delete_by_query",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.delete_by_query_suggest_1),
                                    "rs":             json.dumps({"deleted": 3}),
                                    }],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts":     False,
                    "esUseSsl":          False,
                    "esSslShowWarn":     False,
                    "turnOffSslVerification": True,
                    "esCAcert":          "",
                    "esClientCert":      "",
                    "esClientKey":       "",
                    "appVersion":        "",
                    "minioRegion":       "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber":     1000,
                    "binaryStoreType":   "minio",
                    "minioHost":         "",
                    "minioAccessKey":    "",
                    "minioSecretKey":    "",
                    "esProjectIndexPrefix": "rp_"
                },
                "item_remove_info": {
                    "project": 1,
                    "itemsToDelete": [1, 2]},
                "result":    3
            }
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_service = SuggestService(app_config=app_config,
                                                 search_cfg=self.get_default_search_config())
                response = suggest_service.clean_suggest_info_logs_by_test_item(test["item_remove_info"])

                test["result"].should.equal(response)

                TestSuggestService.shutdown_server(test["test_calls"])

    def test_remove_launches_suggests(self):
        tests = [
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    "content_type":   "application/json",
                                    }],
                "launch_remove_info": {
                    "project": 1,
                    "launch_ids": [1, 2]},
                "result": 0
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/1_suggest/_delete_by_query",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.delete_by_query_suggest_2),
                                    "rs":             json.dumps({"deleted": 1})}],
                "launch_remove_info": {
                    "project": 1,
                    "launch_ids": [1, 2]},
                "result": 1
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_1_suggest",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/rp_1_suggest/_delete_by_query",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.delete_by_query_suggest_2),
                                    "rs":             json.dumps({"deleted": 3}),
                                    }],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts":     False,
                    "esUseSsl":          False,
                    "esSslShowWarn":     False,
                    "turnOffSslVerification": True,
                    "esCAcert":          "",
                    "esClientCert":      "",
                    "esClientKey":       "",
                    "appVersion":        "",
                    "minioRegion":       "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber":     1000,
                    "binaryStoreType":   "minio",
                    "minioHost":         "",
                    "minioAccessKey":    "",
                    "minioSecretKey":    "",
                    "esProjectIndexPrefix": "rp_"
                },
                "launch_remove_info": {
                    "project": 1,
                    "launch_ids": [1, 2]},
                "result": 3
            }
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_service = SuggestService(app_config=app_config,
                                                 search_cfg=self.get_default_search_config())
                response = suggest_service.clean_suggest_info_logs_by_launch_id(test["launch_remove_info"])

                test["result"].should.equal(response)

                TestSuggestService.shutdown_server(test["test_calls"])

    def test_suggest_info_update(self):
        tests = [
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    "content_type":   "application/json",
                                    }],
                "defect_update_info": {
                    "project": 1,
                    "itemsToUpdate": {1: "pb001", 2: "ab001"}},
                "result":     0
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1_suggest/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.get_suggest_info_by_test_item_ids_query),
                                    "rs":             utils.get_fixture(
                                        self.suggest_info_test_items_by_id_1),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.suggest_index_test_item_update),
                                    "rs":             utils.get_fixture(
                                        self.index_logs_rs),
                                    }],
                "defect_update_info": {
                    "project": 1,
                    "itemsToUpdate": {1: "pb001", 2: "ab001"}},
                "result":     1
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1_suggest",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1_suggest/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.get_suggest_info_by_test_item_ids_query),
                                    "rs":             utils.get_fixture(
                                        self.suggest_info_test_items_by_id_2),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.suggest_index_test_item_update_2),
                                    "rs":             utils.get_fixture(
                                        self.index_logs_rs),
                                    }],
                "defect_update_info": {
                    "project": 1,
                    "itemsToUpdate": {1: "pb001", 2: "ab001"}},
                "result":     2
            },
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_service = SuggestService(app_config=app_config,
                                                 search_cfg=self.get_default_search_config())
                suggest_service.es_client.es_client.scroll = MagicMock(return_value=json.loads(
                    utils.get_fixture(self.no_hits_search_rs)))
                response = suggest_service.update_suggest_info(test["defect_update_info"])

                test["result"].should.equal(response)

                TestSuggestService.shutdown_server(test["test_calls"])


if __name__ == '__main__':
    unittest.main()
