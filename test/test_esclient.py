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
import os
from http import HTTPStatus
import logging
import sure # noqa
import httpretty

import commons.launch_objects as launch_objects
import commons.esclient as esclient
from boosting_decision_making.boosting_decision_maker import BoostingDecisionMaker
from utils import utils


class TestEsClient(unittest.TestCase):
    """Tests elasticsearch client functionality"""

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
        self.suggest_test_item_info_w_merged_logs = "suggest_test_item_info_w_merged_logs.json"
        self.one_hit_search_rs_merged_wrong = "one_hit_search_rs_merged_wrong.json"
        self.es_host = "http://localhost:9200"
        self.model_settings = utils.read_json_file("", "model_settings.json", to_json=True)
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @staticmethod
    @utils.ignore_warnings
    def get_fixture(fixture_name, to_json=False):
        """Read fixture from file"""
        with open(os.path.join("fixtures", fixture_name), "r") as file:
            return file.read() if not to_json else json.loads(file.read())

    @utils.ignore_warnings
    def get_default_search_config(self):
        """Get default search config"""
        return {
            "MinShouldMatch": "80%",
            "MinTermFreq":    1,
            "MinDocFreq":     1,
            "BoostAA":        2,
            "BoostLaunch":    2,
            "BoostUniqueID":  2,
            "MaxQueryTerms":  50,
            "SearchLogsMinShouldMatch": "98%",
            "SearchLogsMinSimilarity": 0.9,
            "MinWordLength":  0,
            "BoostModelFolderAllLines":
                self.model_settings["BOOST_MODEL_FOLDER_ALL_LINES"],
            "BoostModelFolderNotAllLines":
                self.model_settings["BOOST_MODEL_FOLDER_NOT_ALL_LINES"],
            "SimilarityWeightsFolder":
                self.model_settings["SIMILARITY_WEIGHTS_FOLDER"],
            "SuggestBoostModelFolderAllLines":
                self.model_settings["SUGGEST_BOOST_MODEL_FOLDER_ALL_LINES"],
            "SuggestBoostModelFolderNotAllLines":
                self.model_settings["SUGGEST_BOOST_MODEL_FOLDER_NOT_ALL_LINES"]
        }

    @utils.ignore_warnings
    def _start_server(self, test_calls):
        httpretty.reset()
        httpretty.enable(allow_net_connect=False)
        for test_info in test_calls:
            if "content_type" in test_info:
                httpretty.register_uri(
                    test_info["method"],
                    self.es_host + test_info["uri"],
                    body=test_info["rs"] if "rs" in test_info else "",
                    status=test_info["status"],
                    content_type=test_info["content_type"],
                )
            else:
                httpretty.register_uri(
                    test_info["method"],
                    self.es_host + test_info["uri"],
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
    def test_list_indices(self):
        """Test checking getting indices from elasticsearch"""
        tests = [
            {
                "test_calls": [{"method":         httpretty.GET,
                                "uri":            "/_cat/indices?format=json",
                                "status":         HTTPStatus.OK,
                                "rs":             "[]",
                                }, ],
                "expected_count": 0,
            },
            {
                "test_calls": [{"method":         httpretty.GET,
                                "uri":            "/_cat/indices?format=json",
                                "status":         HTTPStatus.OK,
                                "rs":             TestEsClient.get_fixture(self.two_indices_rs),
                                }, ],
                "expected_count": 2,
            },
            {
                "test_calls": [{"method":         httpretty.GET,
                                "uri":            "/_cat/indices?format=json",
                                "status":         HTTPStatus.INTERNAL_SERVER_ERROR,
                                }, ],
                "expected_count": 0,
            },
        ]
        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])

                es_client = esclient.EsClient(host=self.es_host,
                                              search_cfg=self.get_default_search_config())

                response = es_client.list_indices()
                response.should.have.length_of(test["expected_count"])

                TestEsClient.shutdown_server(test["test_calls"])

    @utils.ignore_warnings
    def test_create_index(self):
        """Test creating index"""
        tests = [
            {
                "test_calls": [{"method":         httpretty.PUT,
                                "uri":            "/idx0",
                                "status":         HTTPStatus.OK,
                                "content_type":   "application/json",
                                "rs":             TestEsClient.get_fixture(self.index_created_rs),
                                }, ],
                "index":        "idx0",
                "acknowledged": True,
            },
            {
                "test_calls": [{"method":         httpretty.PUT,
                                "uri":            "/idx1",
                                "status":         HTTPStatus.BAD_REQUEST,
                                "content_type":   "application/json",
                                "rs":             TestEsClient.get_fixture(
                                    self.index_already_exists_rs),
                                }, ],
                "index":        "idx1",
                "acknowledged": False,
            },
        ]
        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])

                es_client = esclient.EsClient(host=self.es_host,
                                              search_cfg=self.get_default_search_config())

                response = es_client.create_index(test["index"])
                response.acknowledged.should.equal(test["acknowledged"])

                TestEsClient.shutdown_server(test["test_calls"])

    @utils.ignore_warnings
    def test_exists_index(self):
        """Test existance of a index"""
        tests = [
            {
                "test_calls": [{"method":         httpretty.GET,
                                "uri":            "/idx0",
                                "status":         HTTPStatus.OK,
                                }, ],
                "exists":     True,
                "index":      "idx0",
            },
            {
                "test_calls": [{"method":         httpretty.GET,
                                "uri":            "/idx1",
                                "status":         HTTPStatus.NOT_FOUND,
                                }, ],
                "exists":       False,
                "index":        "idx1",
            },
        ]
        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])

                es_client = esclient.EsClient(host=self.es_host,
                                              search_cfg=self.get_default_search_config())

                response = es_client.index_exists(test["index"])
                response.should.equal(test["exists"])

                TestEsClient.shutdown_server(test["test_calls"])

    @utils.ignore_warnings
    def test_delete_index(self):
        """Test deleting an index"""
        tests = [
            {
                "test_calls": [{"method":         httpretty.DELETE,
                                "uri":            "/1",
                                "status":         HTTPStatus.OK,
                                "content_type":   "application/json",
                                "rs":             TestEsClient.get_fixture(self.index_deleted_rs),
                                }, ],
                "index":      1,
                "result":     1,
            },
            {
                "test_calls": [{"method":         httpretty.DELETE,
                                "uri":            "/2",
                                "status":         HTTPStatus.NOT_FOUND,
                                "content_type":   "application/json",
                                "rs":             TestEsClient.get_fixture(self.index_not_found_rs),
                                }, ],
                "index":      2,
                "result":     0,
            },
        ]
        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])

                es_client = esclient.EsClient(host=self.es_host,
                                              search_cfg=self.get_default_search_config())

                response = es_client.delete_index(test["index"])

                test["result"].should.equal(response)

                TestEsClient.shutdown_server(test["test_calls"])

    @utils.ignore_warnings
    def test_clean_index(self):
        """Test cleaning index logs"""
        tests = [
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(
                                        self.search_not_merged_logs_for_delete),
                                    "rs":             TestEsClient.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             TestEsClient.get_fixture(
                                        self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(self.search_merged_logs),
                                    "rs":             TestEsClient.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             TestEsClient.get_fixture(self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(self.search_not_merged_logs),
                                    "rs":             TestEsClient.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(self.index_logs_rq),
                                    "rs":             TestEsClient.get_fixture(self.index_logs_rs),
                                    }, ],
                "rq":             launch_objects.CleanIndex(ids=[1], project=1),
                "expected_count": 1
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    }, ],
                "rq":             launch_objects.CleanIndex(ids=[1], project=2),
                "expected_count": 0
            },
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])

                es_client = esclient.EsClient(host=self.es_host,
                                              search_cfg=self.get_default_search_config())
                es_client.es_client.scroll = MagicMock(return_value=json.loads(
                    TestEsClient.get_fixture(self.no_hits_search_rs)))

                response = es_client.delete_logs(test["rq"])

                test["expected_count"].should.equal(response)

                TestEsClient.shutdown_server(test["test_calls"])

    @utils.ignore_warnings
    def test_search_logs(self):
        """Test search logs"""
        tests = [
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(self.search_logs_rq),
                                    "rs":             TestEsClient.get_fixture(
                                        self.no_hits_search_rs),
                                    }, ],
                "rq":             launch_objects.SearchLogs(launchId=1,
                                                            launchName="Launch 1",
                                                            itemId=3,
                                                            projectId=1,
                                                            filteredLaunchIds=[1],
                                                            logMessages=["error"],
                                                            logLines=-1),
                "expected_count": 0
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    }, ],
                "rq":             launch_objects.SearchLogs(launchId=1,
                                                            launchName="Launch 1",
                                                            itemId=3,
                                                            projectId=1,
                                                            filteredLaunchIds=[1],
                                                            logMessages=[""],
                                                            logLines=-1),
                "expected_count": 0
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(self.search_logs_rq),
                                    "rs":             TestEsClient.get_fixture(
                                        self.one_hit_search_rs_search_logs),
                                    }, ],
                "rq":             launch_objects.SearchLogs(launchId=1,
                                                            launchName="Launch 1",
                                                            itemId=3,
                                                            projectId=1,
                                                            filteredLaunchIds=[1],
                                                            logMessages=["error"],
                                                            logLines=-1),
                "expected_count": 0
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(
                                        self.search_logs_rq_not_found),
                                    "rs":             TestEsClient.get_fixture(
                                        self.two_hits_search_rs_search_logs),
                                    }, ],
                "rq":             launch_objects.SearchLogs(launchId=1,
                                                            launchName="Launch 1",
                                                            itemId=3,
                                                            projectId=1,
                                                            filteredLaunchIds=[1],
                                                            logMessages=["error occured once"],
                                                            logLines=-1),
                "expected_count": 1
            },
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])

                es_client = esclient.EsClient(host=self.es_host,
                                              search_cfg=self.get_default_search_config())

                response = es_client.search_logs(test["rq"])
                response.should.have.length_of(test["expected_count"])

                TestEsClient.shutdown_server(test["test_calls"])

    @utils.ignore_warnings
    def test_index_logs(self):
        """Test indexing logs from launches"""
        tests = [
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    }, ],
                "index_rq":       TestEsClient.get_fixture(self.launch_wo_test_items),
                "has_errors":     False,
                "expected_count": 0,
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    }, ],
                "index_rq":       TestEsClient.get_fixture(self.launch_w_test_items_wo_logs),
                "has_errors":     False,
                "expected_count": 0,
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    }, ],
                "index_rq":       TestEsClient.get_fixture(self.launch_w_test_items_w_empty_logs),
                "has_errors":     False,
                "expected_count": 0,
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    "rs":             TestEsClient.get_fixture(
                                        self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(
                                        self.index_logs_rq_big_messages),
                                    "rs":             TestEsClient.get_fixture(
                                        self.index_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/2/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(
                                        self.search_merged_logs),
                                    "rs":             TestEsClient.get_fixture(
                                        self.two_hits_search_with_big_messages_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             TestEsClient.get_fixture(
                                        self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/2/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(
                                        self.search_not_merged_logs),
                                    "rs":             TestEsClient.get_fixture(
                                        self.two_hits_search_with_big_messages_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(
                                        self.index_logs_rq_merged_logs),
                                    "rs":             TestEsClient.get_fixture(
                                        self.index_logs_rs),
                                    }, ],
                "index_rq":       TestEsClient.get_fixture(self.launch_w_test_items_w_logs),
                "has_errors":     False,
                "expected_count": 2,
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    "rs":             TestEsClient.get_fixture(
                                        self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(
                                        self.index_logs_rq_different_log_level),
                                    "rs":             TestEsClient.get_fixture(
                                        self.index_logs_rs_different_log_level),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/2/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(self.search_merged_logs),
                                    "rs":             TestEsClient.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             TestEsClient.get_fixture(self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/2/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(self.search_not_merged_logs),
                                    "rs":             TestEsClient.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             TestEsClient.get_fixture(
                                        self.index_logs_rq_different_log_level_merged),
                                    "rs":             TestEsClient.get_fixture(
                                        self.index_logs_rs_different_log_level),
                                    }, ],
                "index_rq":       TestEsClient.get_fixture(
                    self.launch_w_test_items_w_logs_different_log_level),
                "has_errors":     False,
                "expected_count": 1,
            },
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])

                es_client = esclient.EsClient(host=self.es_host,
                                              search_cfg=self.get_default_search_config())
                es_client.es_client.scroll = MagicMock(return_value=json.loads(
                    TestEsClient.get_fixture(self.no_hits_search_rs)))
                launches = [launch_objects.Launch(**launch)
                            for launch in json.loads(test["index_rq"])]
                response = es_client.index_logs(launches)

                test["has_errors"].should.equal(response.errors)
                test["expected_count"].should.equal(response.took)

                TestEsClient.shutdown_server(test["test_calls"])

    @utils.ignore_warnings
    def test_analyze_logs(self):
        """Test analyzing logs"""
        tests = [
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.OK,
                                         }, ],
                "index_rq":            TestEsClient.get_fixture(self.launch_wo_test_items),
                "expected_count":      0,
                "expected_issue_type": "",
                "boost_predict":       ([], [])
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.OK,
                                         }, ],
                "index_rq":            TestEsClient.get_fixture(
                    self.launch_w_test_items_wo_logs),
                "expected_count":      0,
                "expected_issue_type": "",
                "boost_predict":       ([], [])
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/2",
                                         "status":         HTTPStatus.OK,
                                         }, ],
                "index_rq":            TestEsClient.get_fixture(
                    self.launch_w_test_items_w_empty_logs),
                "expected_count":      0,
                "expected_issue_type": "",
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }, ],
                "msearch_results": [TestEsClient.get_fixture(self.no_hits_search_rs, to_json=True),
                                    TestEsClient.get_fixture(self.no_hits_search_rs, to_json=True)],
                "index_rq":       TestEsClient.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count":      0,
                "expected_issue_type": "",
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    }, ],
                "index_rq":       TestEsClient.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count":      0,
                "expected_issue_type": "",
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [TestEsClient.get_fixture(self.no_hits_search_rs, to_json=True),
                                    TestEsClient.get_fixture(self.one_hit_search_rs, to_json=True)],
                "index_rq":       TestEsClient.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict":       ([1], [[0.2, 0.8]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [TestEsClient.get_fixture(self.one_hit_search_rs, to_json=True),
                                    TestEsClient.get_fixture(self.two_hits_search_rs, to_json=True)],
                "index_rq":       TestEsClient.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict":       ([1, 0], [[0.2, 0.8], [0.7, 0.3]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [TestEsClient.get_fixture(self.two_hits_search_rs, to_json=True),
                                    TestEsClient.get_fixture(self.three_hits_search_rs, to_json=True)],
                "index_rq":       TestEsClient.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict":       ([1, 1], [[0.2, 0.8], [0.3, 0.7]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [TestEsClient.get_fixture(self.no_hits_search_rs, to_json=True),
                                    TestEsClient.get_fixture(self.three_hits_search_rs, to_json=True)],
                "index_rq":       TestEsClient.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "PB001",
                "boost_predict":       ([0, 1], [[0.8, 0.2], [0.3, 0.7]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [TestEsClient.get_fixture(self.no_hits_search_rs, to_json=True),
                                    TestEsClient.get_fixture(self.three_hits_search_rs, to_json=True)],
                "index_rq":       TestEsClient.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict":       ([1, 0], [[0.2, 0.8], [0.7, 0.3]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [TestEsClient.get_fixture(self.two_hits_search_rs, to_json=True)],
                "index_rq":       TestEsClient.get_fixture(
                    self.launch_w_test_items_w_logs_to_be_merged),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict":       ([], [])
            }
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                config = self.get_default_search_config()
                es_client = esclient.EsClient(host=self.es_host,
                                              search_cfg=config)
                _boosting_decision_maker = BoostingDecisionMaker()
                _boosting_decision_maker.get_feature_ids = MagicMock(return_value=[0])
                _boosting_decision_maker.predict = MagicMock(return_value=test["boost_predict"])
                if "msearch_results" in test:
                    es_client.es_client.msearch = MagicMock(
                        return_value={"responses": test["msearch_results"]})
                es_client.boosting_decision_maker_all_lines = _boosting_decision_maker
                launches = [launch_objects.Launch(**launch)
                            for launch in json.loads(test["index_rq"])]
                response = es_client.analyze_logs(launches)

                response.should.have.length_of(test["expected_count"])

                if test["expected_issue_type"] != "":
                    test["expected_issue_type"].should.equal(response[0].issueType)

                if "expected_id" in test:
                    test["expected_id"].should.equal(response[0].relevantItem)

                TestEsClient.shutdown_server(test["test_calls"])

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
                                    "rq":           TestEsClient.get_fixture(self.search_rq_first),
                                    "rs":           TestEsClient.get_fixture(
                                        self.no_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           TestEsClient.get_fixture(self.search_rq_second),
                                    "rs":           TestEsClient.get_fixture(
                                        self.no_hits_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **TestEsClient.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
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
                                    "rq":           TestEsClient.get_fixture(self.search_rq_first),
                                    "rs":           TestEsClient.get_fixture(
                                        self.no_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           TestEsClient.get_fixture(self.search_rq_second),
                                    "rs":           TestEsClient.get_fixture(
                                        self.no_hits_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **TestEsClient.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
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
                                    "rq":           TestEsClient.get_fixture(self.search_rq_first),
                                    "rs":           TestEsClient.get_fixture(
                                        self.no_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           TestEsClient.get_fixture(self.search_rq_second),
                                    "rs":           TestEsClient.get_fixture(
                                        self.one_hit_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **TestEsClient.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=80.0)],
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
                                    "rq":           TestEsClient.get_fixture(self.search_rq_first),
                                    "rs":           TestEsClient.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           TestEsClient.get_fixture(self.search_rq_second),
                                    "rs":           TestEsClient.get_fixture(
                                        self.one_hit_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **TestEsClient.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=70.0)],
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
                                    "rq":           TestEsClient.get_fixture(self.search_rq_first),
                                    "rs":           TestEsClient.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           TestEsClient.get_fixture(self.search_rq_second),
                                    "rs":           TestEsClient.get_fixture(
                                        self.two_hits_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **TestEsClient.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=70.0)],
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
                                    "rq":           TestEsClient.get_fixture(self.search_rq_first),
                                    "rs":           TestEsClient.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           TestEsClient.get_fixture(self.search_rq_second),
                                    "rs":           TestEsClient.get_fixture(
                                        self.two_hits_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **TestEsClient.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=70.0),
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='PB001',
                                                         relevantItem=2,
                                                         relevantLogId=2,
                                                         matchScore=45.0)],
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
                                    "rq":           TestEsClient.get_fixture(self.search_rq_first),
                                    "rs":           TestEsClient.get_fixture(
                                        self.two_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           TestEsClient.get_fixture(self.search_rq_second),
                                    "rs":           TestEsClient.get_fixture(
                                        self.three_hits_search_rs),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **TestEsClient.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='PB001',
                                                         relevantItem=3,
                                                         relevantLogId=3,
                                                         matchScore=80.0),
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=70.0),
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='PB001',
                                                         relevantItem=2,
                                                         relevantLogId=2,
                                                         matchScore=45.0)],
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
                                    "rq":           TestEsClient.get_fixture(self.search_rq_first),
                                    "rs":           TestEsClient.get_fixture(
                                        self.two_hits_search_rs),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           TestEsClient.get_fixture(self.search_rq_second),
                                    "rs":           TestEsClient.get_fixture(
                                        self.three_hits_search_rs_with_duplicate),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **TestEsClient.get_fixture(self.suggest_test_item_info_w_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='PB001',
                                                         relevantItem=3,
                                                         relevantLogId=3,
                                                         matchScore=70.0),
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=70.0)],
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
                                    "rq":           TestEsClient.get_fixture(self.search_rq_merged_first),
                                    "rs":           TestEsClient.get_fixture(
                                        self.one_hit_search_rs_merged),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           TestEsClient.get_fixture(self.search_rq_merged_second),
                                    "rs":           TestEsClient.get_fixture(
                                        self.one_hit_search_rs_merged),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **TestEsClient.get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)),
                "expected_result":     [
                    launch_objects.SuggestAnalysisResult(testItem=123,
                                                         issueType='AB001',
                                                         relevantItem=1,
                                                         relevantLogId=1,
                                                         matchScore=90.0)],
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
                                    "rq":           TestEsClient.get_fixture(self.search_rq_merged_first),
                                    "rs":           TestEsClient.get_fixture(
                                        self.one_hit_search_rs_merged_wrong),
                                    },
                                   {"method":       httpretty.GET,
                                    "uri":          "/1/_search",
                                    "status":       HTTPStatus.OK,
                                    "content_type": "application/json",
                                    "rq":           TestEsClient.get_fixture(self.search_rq_merged_second),
                                    "rs":           TestEsClient.get_fixture(
                                        self.one_hit_search_rs_merged_wrong),
                                    }, ],
                "test_item_info":      launch_objects.TestItemInfo(
                    **TestEsClient.get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)),
                "expected_result":     [],
                "boost_predict":       ([], [])
            },
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                config = self.get_default_search_config()
                es_client = esclient.EsClient(host=self.es_host,
                                              search_cfg=config)
                _boosting_decision_maker = BoostingDecisionMaker()
                _boosting_decision_maker.get_feature_ids = MagicMock(return_value=[0])
                _boosting_decision_maker.predict = MagicMock(return_value=test["boost_predict"])
                es_client.suggest_decision_maker_all_lines = _boosting_decision_maker
                response = es_client.suggest_items(test["test_item_info"])

                response.should.have.length_of(len(test["expected_result"]))
                for real_resp, expected_resp in zip(response, test["expected_result"]):
                    real_resp.should.equal(expected_resp)

                TestEsClient.shutdown_server(test["test_calls"])


if __name__ == '__main__':
    unittest.main()
