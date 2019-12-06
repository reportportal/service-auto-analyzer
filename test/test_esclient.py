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
import os
from http import HTTPStatus
import logging
import warnings
import sure
import httpretty

import commons.launch_objects as launch_objects
import commons.esclient as esclient

def ignore_warnings(method):
    def _inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rv = method(*args, **kwargs)
        return rv
    return _inner 


class TestEsClient(unittest.TestCase):

    ERROR_LOGGING_LEVEL = 40000

    def setUp(self):
        self.two_indices_rs                                 = "two_indices_rs.json"
        self.index_created_rs                               = "index_created_rs.json"
        self.index_already_exists_rs                        = "index_already_exists_rs.json"
        self.index_deleted_rs                               = "index_deleted_rs.json"
        self.index_not_found_rs                             = "index_not_found_rs.json"
        self.launch_wo_test_items                           = "launch_wo_test_items.json"
        self.launch_w_test_items_wo_logs                    = "launch_w_test_items_wo_logs.json"
        self.launch_w_test_items_w_logs                     = "launch_w_test_items_w_logs.json"
        self.launch_w_test_items_w_empty_logs               = "launch_w_test_items_w_empty_logs.json"
        self.launch_w_test_items_w_logs_to_be_merged        = "launch_w_test_items_w_logs_to_be_merged.json"
        self.index_logs_rq                                  = "index_logs_rq.json"
        self.index_logs_rs                                  = "index_logs_rs.json"
        self.search_rq                                      = "search_rq.json"
        self.no_hits_search_rs                              = "no_hits_search_rs.json"
        self.one_hit_search_rs                              = "one_hit_search_rs.json"
        self.two_hits_search_rs                             = "two_hits_search_rs.json"
        self.three_hits_search_rs                           = "three_hits_search_rs.json"
        self.launch_w_test_items_w_logs_different_log_level = "launch_w_test_items_w_logs_different_log_level.json"
        self.index_logs_rq_different_log_level              = "index_logs_rq_different_log_level.json"
        self.index_logs_rs_different_log_level              = "index_logs_rs_different_log_level.json"
        self.delete_logs_rs                                 = "delete_logs_rs.json"
        self.two_hits_search_with_big_messages_rs           = "two_hits_search_with_big_messages_rs.json"
        self.es_host = "http://localhost:9200"
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.DEBUG)

    def get_fixture(self, fixture_name):
        with open(os.path.join("fixtures", fixture_name), "r") as f:
            return f.read()

    def get_default_search_config(self):
        return {
            "MinShouldMatch": "80%",
            "MinTermFreq":    1,
            "MinDocFreq":     1,
            "BoostAA":        2,
            "BoostLaunch":    2,
            "BoostUniqueID":  2,
            "MaxQueryTerms":  50,
        }

    def start_server(self, test_calls):
        httpretty.reset()
        httpretty.enable()
        for test_info in test_calls:
            if "content_type" in test_info:
                httpretty.register_uri(
                    test_info["method"],
                    self.es_host + test_info["uri"],
                    body = test_info["rs"] if "rs" in test_info else "",
                    status = test_info["status"],
                    content_type = test_info["content_type"],
                )
            else:
                httpretty.register_uri(
                    test_info["method"],
                    self.es_host + test_info["uri"],
                    body = test_info["rs"] if "rs" in test_info else "",
                    status = test_info["status"],
                )

    def shutdown_server(self, test_calls):
        httpretty.latest_requests().should.have.length_of(len(test_calls))
        for expected_test_call, test_call in zip(test_calls,httpretty.latest_requests()):
            expected_test_call["method"].should.equal(test_call.method)
            expected_test_call["uri"].should.equal(test_call.path)
        httpretty.disable()
        httpretty.reset()

    @ignore_warnings
    def test_list_indices(self):
        tests =[{
                    "test_calls":   [{
                                        "method":         httpretty.GET,
                                        "uri":            "/_cat/indices?format=json",
                                        "status":         HTTPStatus.OK,
                                        "rs":             "[]",
                                    }],
                    "expected_count": 0,
                },
                {
                    "test_calls":   [{
                                        "method":         httpretty.GET,
                                        "uri":            "/_cat/indices?format=json",
                                        "status":         HTTPStatus.OK,
                                        "rs":             self.get_fixture(self.two_indices_rs),
                                    }],
                    "expected_count": 2,
                },
                {
                    "test_calls":   [{
                                        "method":         httpretty.GET,
                                        "uri":            "/_cat/indices?format=json",
                                        "status":         HTTPStatus.INTERNAL_SERVER_ERROR,
                                    }],
                    "expected_count": 0,
                },
            ]
        for test in tests:
            self.start_server(test["test_calls"])

            es_client = esclient.EsClient(host = self.es_host, search_cfg = self.get_default_search_config())

            response = es_client.list_indices()
            response.should.have.length_of(test["expected_count"])

            self.shutdown_server(test["test_calls"])

    @ignore_warnings
    def test_create_index(self):
        tests =[{
                    "test_calls":   [{
                                        "method":         httpretty.PUT,
                                        "uri":            "/idx0",
                                        "status":         HTTPStatus.OK,
                                        "content_type":   "application/json",
                                        "rs":             self.get_fixture(self.index_created_rs),
                                    }],
                    "index":        "idx0",
                    "acknowledged": True,
                },
                {
                    "test_calls":   [{
                                        "method":         httpretty.PUT,
                                        "uri":            "/idx1",
                                        "status":         HTTPStatus.BAD_REQUEST,
                                        "content_type":   "application/json",
                                        "rs":             self.get_fixture(self.index_already_exists_rs),
                                    }],
                    "index":        "idx1",
                    "acknowledged": False,
                },
            ]
        for test in tests:
            self.start_server(test["test_calls"])

            es_client = esclient.EsClient(host = self.es_host, search_cfg = self.get_default_search_config())

            response = es_client.create_index(test["index"])
            response.acknowledged.should.equal(test["acknowledged"])

            self.shutdown_server(test["test_calls"])

    @ignore_warnings
    def test_exists_index(self):
        tests =[{
                    "test_calls":   [{
                                        "method":         httpretty.GET,
                                        "uri":            "/idx0",
                                        "status":         HTTPStatus.OK,
                                    }],
                    "exists":       True,
                    "index":          "idx0",
                },
                {
                    "test_calls":   [{
                                        "method":         httpretty.GET,
                                        "uri":            "/idx1",
                                        "status":         HTTPStatus.NOT_FOUND,
                                    }],
                    "exists":       False,
                    "index":        "idx1",
                },
            ]
        for test in tests:
            self.start_server(test["test_calls"])

            es_client = esclient.EsClient(host = self.es_host, search_cfg = self.get_default_search_config())

            response = es_client.index_exists(test["index"])
            response.should.equal(test["exists"])

            self.shutdown_server(test["test_calls"])

    @ignore_warnings
    def test_delete_index(self):
        tests =[{
                    "test_calls":   [{
                                        "method":         httpretty.DELETE,
                                        "uri":            "/1",
                                        "status":         HTTPStatus.OK,
                                        "content_type":   "application/json",
                                        "rs":             self.get_fixture(self.index_deleted_rs),
                                    }],
                    "index":        1,
                    "has_errors":   False,
                },
                {
                    "test_calls":   [{
                                        "method":         httpretty.DELETE,
                                        "uri":            "/2",
                                        "status":         HTTPStatus.NOT_FOUND,
                                        "content_type":   "application/json",
                                        "rs":             self.get_fixture(self.index_not_found_rs),
                                    }],
                    "index":        2,
                    "has_errors":   True,
                },
            ]
        for test in tests:
            self.start_server(test["test_calls"])

            es_client = esclient.EsClient(host = self.es_host, search_cfg = self.get_default_search_config())

            response = es_client.delete_index(test["index"])

            test["has_errors"].should.equal(len(response.error) > 0)

            self.shutdown_server(test["test_calls"])

    @ignore_warnings
    def test_clean_index(self):
        tests =[{
                    "test_calls":     [{
                                            "method":         httpretty.GET,
                                            "uri":            "/1/_search",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rq":             self.get_fixture(self.search_rq),
                                            "rs":             self.get_fixture(self.one_hit_search_rs),
                                       },
                                       {
                                            "method":         httpretty.POST,
                                            "uri":            "/_bulk?refresh=true",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rs":             self.get_fixture(self.delete_logs_rs),
                                       },
                                       {
                                            "method":         httpretty.GET,
                                            "uri":            "/1/_search",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rq":             self.get_fixture(self.search_rq),
                                            "rs":             self.get_fixture(self.one_hit_search_rs),
                                        },
                                       {
                                            "method":         httpretty.POST,
                                            "uri":            "/_bulk?refresh=true",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rs":             self.get_fixture(self.delete_logs_rs),
                                        },
                                        {
                                            "method":         httpretty.GET,
                                            "uri":            "/1/_search",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rq":             self.get_fixture(self.search_rq),
                                            "rs":             self.get_fixture(self.one_hit_search_rs),
                                        },
                                        {
                                            "method":         httpretty.POST,
                                            "uri":            "/_bulk?refresh=true",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rq":             self.get_fixture(self.index_logs_rq),
                                            "rs":             self.get_fixture(self.index_logs_rs),
                                        },
                                      ],
                    "rq":             launch_objects.CleanIndex(ids = [1], project = 1),
                    "has_errors":     False,
                    "expected_count": 1
                },
                {
                    "test_calls":     [ {
                                            "method":         httpretty.GET,
                                            "uri":            "/2/_search",
                                            "status":         HTTPStatus.NOT_FOUND,
                                            "content_type":   "application/json",
                                            "rq":             self.get_fixture(self.search_rq),
                                            "rs":             self.get_fixture(self.no_hits_search_rs),
                                        },
                                        {
                                          "method":         httpretty.POST,
                                          "uri":            "/_bulk?refresh=true",
                                          "status":         HTTPStatus.NOT_FOUND,
                                          "content_type":   "application/json",
                                          "rs":             self.get_fixture(self.delete_logs_rs),
                                      }],
                    "rq":             launch_objects.CleanIndex(ids = [1], project = 2),
                    "has_errors":     True,
                    "expected_count": 0
                },
            ]

        for test in tests:
            self.start_server(test["test_calls"])

            es_client = esclient.EsClient(host = self.es_host, search_cfg = self.get_default_search_config())

            response = es_client.delete_logs(test["rq"])

            test["has_errors"].should.equal(response.errors)
            test["expected_count"].should.equal(response.took)

            self.shutdown_server(test["test_calls"])

    @ignore_warnings
    def test_index_logs(self):
        tests =[{
                    "test_calls":     [{
                                        "method":         httpretty.GET,
                                        "uri":            "/1",
                                        "status":         HTTPStatus.OK,
                                        "content_type":   "application/json",
                                      }],
                    "index_rq":       self.get_fixture(self.launch_wo_test_items),
                    "has_errors":     False,
                    "expected_count": 0,
                },
                {
                    "test_calls":     [{
                                        "method":         httpretty.GET,
                                        "uri":            "/1",
                                        "status":         HTTPStatus.OK,
                                        "content_type":   "application/json",
                                      }],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_wo_logs),
                    "has_errors":     False,
                    "expected_count": 0, 
                },
                {
                    "test_calls":     [{
                                        "method":         httpretty.GET,
                                        "uri":            "/2",
                                        "status":         HTTPStatus.OK,
                                        "content_type":   "application/json",
                                      }],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_w_empty_logs),
                    "has_errors":     False,
                    "expected_count": 0, 
                },
                {
                    "test_calls":     [{
                                            "method":         httpretty.GET,
                                            "uri":            "/2",
                                            "status":         HTTPStatus.NOT_FOUND,
                                        },
                                        {
                                            "method":         httpretty.PUT,
                                            "uri":            "/2",
                                            "status":         HTTPStatus.OK,
                                            "rs":             self.get_fixture(self.index_created_rs),
                                        },
                                        {
                                            "method":         httpretty.POST,
                                            "uri":            "/_bulk?refresh=true",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rq":             self.get_fixture(self.index_logs_rq),
                                            "rs":             self.get_fixture(self.index_logs_rs),
                                        },
                                        {
                                          "method":         httpretty.GET,
                                          "uri":            "/2/_search",
                                          "status":         HTTPStatus.OK,
                                          "content_type":   "application/json",
                                          "rq":             self.get_fixture(self.search_rq),
                                          "rs":             self.get_fixture(self.two_hits_search_with_big_messages_rs),
                                        },
                                        {
                                            "method":         httpretty.POST,
                                            "uri":            "/_bulk?refresh=true",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rs":             self.get_fixture(self.delete_logs_rs),
                                        },
                                        {
                                          "method":         httpretty.GET,
                                          "uri":            "/2/_search",
                                          "status":         HTTPStatus.OK,
                                          "content_type":   "application/json",
                                          "rq":             self.get_fixture(self.search_rq),
                                          "rs":             self.get_fixture(self.two_hits_search_with_big_messages_rs),
                                        },
                                        {
                                            "method":         httpretty.POST,
                                            "uri":            "/_bulk?refresh=true",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rq":             self.get_fixture(self.index_logs_rq),
                                            "rs":             self.get_fixture(self.index_logs_rs),
                                        },
                                      ],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_w_logs),
                    "has_errors":     False,
                    "expected_count": 2, 
                },
                {
                    "test_calls":     [{
                                            "method":         httpretty.GET,
                                            "uri":            "/2",
                                            "status":         HTTPStatus.NOT_FOUND,
                                        },
                                        {
                                            "method":         httpretty.PUT,
                                            "uri":            "/2",
                                            "status":         HTTPStatus.OK,
                                            "rs":             self.get_fixture(self.index_created_rs),
                                        },
                                        {
                                            "method":         httpretty.POST,
                                            "uri":            "/_bulk?refresh=true",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rq":             self.get_fixture(self.index_logs_rq_different_log_level),
                                            "rs":             self.get_fixture(self.index_logs_rs_different_log_level),
                                        },
                                        {
                                          "method":         httpretty.GET,
                                          "uri":            "/2/_search",
                                          "status":         HTTPStatus.OK,
                                          "content_type":   "application/json",
                                          "rq":             self.get_fixture(self.search_rq),
                                          "rs":             self.get_fixture(self.one_hit_search_rs),
                                        },
                                        {
                                            "method":         httpretty.POST,
                                            "uri":            "/_bulk?refresh=true",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rs":             self.get_fixture(self.delete_logs_rs),
                                        },
                                        {
                                          "method":         httpretty.GET,
                                          "uri":            "/2/_search",
                                          "status":         HTTPStatus.OK,
                                          "content_type":   "application/json",
                                          "rq":             self.get_fixture(self.search_rq),
                                          "rs":             self.get_fixture(self.one_hit_search_rs),
                                        },
                                        {
                                            "method":         httpretty.POST,
                                            "uri":            "/_bulk?refresh=true",
                                            "status":         HTTPStatus.OK,
                                            "content_type":   "application/json",
                                            "rq":             self.get_fixture(self.index_logs_rq_different_log_level),
                                            "rs":             self.get_fixture(self.index_logs_rs_different_log_level),
                                        },
                                      ],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_w_logs_different_log_level),
                    "has_errors":     False,
                    "expected_count": 1, 
                },
            ]

        for test in tests:
            self.start_server(test["test_calls"])

            es_client = esclient.EsClient(host = self.es_host, search_cfg = self.get_default_search_config())
            launches = [launch_objects.Launch(**launch) for launch in json.loads(test["index_rq"])]
            
            response = es_client.index_logs(launches)

            test["has_errors"].should.equal(response.errors)
            test["expected_count"].should.equal(response.took)

            self.shutdown_server(test["test_calls"])

    @ignore_warnings
    def test_analyze_logs(self):
        tests =[{
                    "test_calls":          [],
                    "index_rq":            self.get_fixture(self.launch_wo_test_items),
                    "expected_count":      0,
                    "expected_issue_type": "", 
                },
                {
                    "test_calls":     [],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_wo_logs),
                    "expected_count": 0,
                    "expected_issue_type": "",
                },
                {
                    "test_calls":     [],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_w_empty_logs),
                    "expected_count": 0,
                    "expected_issue_type": "",
                },
                {
                    "test_calls":     [{"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(self.no_hits_search_rs),
                                       },
                                       {"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(self.no_hits_search_rs),
                                       },],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_w_logs),
                    "expected_count": 0,
                    "expected_issue_type": "",
                },
                {
                    "test_calls":     [{"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(self.no_hits_search_rs),
                                       },
                                       {"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(self.one_hit_search_rs),
                                       },],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_w_logs),
                    "expected_count": 1,
                    "expected_issue_type": "AB001",
                },
                {
                    "test_calls":     [{"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(self.one_hit_search_rs),
                                       },
                                       {"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(self.two_hits_search_rs),
                                       },],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_w_logs),
                    "expected_count": 1,
                    "expected_issue_type": "AB001",
                },
                {
                    "test_calls":     [{"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(self.two_hits_search_rs),
                                       },
                                       {"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(
                                            self.three_hits_search_rs),
                                       },],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_w_logs),
                    "expected_count": 1,
                    "expected_issue_type": "AB001",
                },
                {
                    "test_calls":     [{"method":      httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(
                                            self.no_hits_search_rs),
                                        },
                                       {"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(
                                            self.three_hits_search_rs),
                                        },],
                    "index_rq":       self.get_fixture(self.launch_w_test_items_w_logs),
                    "expected_count": 1,
                    "expected_issue_type": "PB001",
                },
                {
                    "test_calls":     [{"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(
                                            self.two_hits_search_rs),},],
                    "index_rq":       self.get_fixture(
                        self.launch_w_test_items_w_logs_different_log_level),
                    "expected_count": 1,
                    "expected_issue_type": "AB001",
                },
                {
                    "test_calls":     [{"method":       httpretty.GET,
                                        "uri":          "/2/_search",
                                        "status":       HTTPStatus.OK,
                                        "content_type": "application/json",
                                        "rq":           self.get_fixture(self.search_rq),
                                        "rs":           self.get_fixture(
                                            self.two_hits_search_rs),},],
                    "index_rq":       self.get_fixture(
                        self.launch_w_test_items_w_logs_to_be_merged),
                    "expected_count": 1,
                    "expected_issue_type": "AB001",
                },]

        for test in tests:
            self.start_server(test["test_calls"])

            es_client = esclient.EsClient(host=self.es_host,
                                          search_cfg=self.get_default_search_config())
            launches = [launch_objects.Launch(**launch) for launch in json.loads(test["index_rq"])]
            response = es_client.analyze_logs(launches)

            response.should.have.length_of(test["expected_count"])

            if test["expected_issue_type"] != "":
                test["expected_issue_type"].should.equal(response[0].issueType)

            self.shutdown_server(test["test_calls"])

if __name__ == '__main__':
    unittest.main()
