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
import sure # noqa
import httpretty

import commons.launch_objects as launch_objects
from commons import esclient
from test.test_service import TestService
from utils import utils


class TestEsClient(TestService):

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
                                "rs":             utils.get_fixture(self.two_indices_rs),
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

                es_client = esclient.EsClient(app_config=self.app_config,
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
                                "rs":             utils.get_fixture(self.index_created_rs),
                                }, ],
                "index":        "idx0",
                "acknowledged": True,
            },
            {
                "test_calls": [{"method":         httpretty.PUT,
                                "uri":            "/idx1",
                                "status":         HTTPStatus.BAD_REQUEST,
                                "content_type":   "application/json",
                                "rs":             utils.get_fixture(
                                    self.index_already_exists_rs),
                                }, ],
                "index":        "idx1",
                "acknowledged": False,
            },
        ]
        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])

                es_client = esclient.EsClient(app_config=self.app_config,
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

                es_client = esclient.EsClient(app_config=self.app_config,
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
                                "rs":             utils.get_fixture(self.index_deleted_rs),
                                }, ],
                "index":      1,
                "result":     True,
            },
            {
                "test_calls": [{"method":         httpretty.DELETE,
                                "uri":            "/2",
                                "status":         HTTPStatus.NOT_FOUND,
                                "content_type":   "application/json",
                                "rs":             utils.get_fixture(self.index_not_found_rs),
                                }, ],
                "index":      2,
                "result":     False,
            },
        ]
        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])

                es_client = esclient.EsClient(app_config=self.app_config,
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
                                    "rq":             utils.get_fixture(
                                        self.search_not_merged_logs_for_delete),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(
                                        self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_not_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.index_logs_rs),
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
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.search_not_merged_logs_for_delete),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(
                                        self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_not_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.index_logs_rq),
                                    "rs":             utils.get_fixture(self.index_logs_rs),
                                    }],
                "rq":             launch_objects.CleanIndex(ids=[1], project=1),
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
                "expected_count": 1
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_2",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    }],
                "rq":             launch_objects.CleanIndex(ids=[1], project=2),
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
                "expected_count": 0
            }
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                es_client = esclient.EsClient(app_config=app_config,
                                              search_cfg=self.get_default_search_config())
                es_client.es_client.scroll = MagicMock(return_value=json.loads(
                    utils.get_fixture(self.no_hits_search_rs)))

                response = es_client.delete_logs(test["rq"])

                test["expected_count"].should.equal(response)

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
                "index_rq":       utils.get_fixture(self.launch_wo_test_items),
                "has_errors":     False,
                "expected_count": 0,
                "expected_log_exceptions": []
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/1",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    }, ],
                "index_rq":       utils.get_fixture(self.launch_w_test_items_wo_logs),
                "has_errors":     False,
                "expected_count": 0,
                "expected_log_exceptions": []
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    }, ],
                "index_rq":       utils.get_fixture(self.launch_w_test_items_w_empty_logs),
                "has_errors":     False,
                "expected_count": 0,
                "expected_log_exceptions": []
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(
                                        self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.index_logs_rq_big_messages),
                                    "rs":             utils.get_fixture(
                                        self.index_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/2/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.search_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.two_hits_search_with_big_messages_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(
                                        self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/2/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.search_not_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.two_hits_search_with_big_messages_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.index_logs_rq_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.index_logs_rs),
                                    }, ],
                "index_rq":       utils.get_fixture(self.launch_w_test_items_w_logs),
                "has_errors":     False,
                "expected_count": 2,
                "expected_log_exceptions":  [
                    launch_objects.LogExceptionResult(
                        logId=1, foundExceptions=['java.lang.NoClassDefFoundError']),
                    launch_objects.LogExceptionResult(
                        logId=2, foundExceptions=['java.lang.NoClassDefFoundError'])]
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(
                                        self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.index_logs_rq_different_log_level),
                                    "rs":             utils.get_fixture(
                                        self.index_logs_rs_different_log_level),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/2/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/2/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_not_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.index_logs_rq_different_log_level_merged),
                                    "rs":             utils.get_fixture(
                                        self.index_logs_rs_different_log_level),
                                    }, ],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs_different_log_level),
                "has_errors":     False,
                "expected_count": 1,
                "expected_log_exceptions": [launch_objects.LogExceptionResult(logId=1, foundExceptions=[])]
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_2",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    },
                                   {"method":         httpretty.PUT,
                                    "uri":            "/rp_2",
                                    "status":         HTTPStatus.OK,
                                    "rs":             utils.get_fixture(
                                        self.index_created_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.index_logs_rq_different_log_level_with_prefix),
                                    "rs":             utils.get_fixture(
                                        self.index_logs_rs_different_log_level),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_2/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rs":             utils.get_fixture(self.delete_logs_rs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_2/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_not_merged_logs),
                                    "rs":             utils.get_fixture(
                                        self.one_hit_search_rs),
                                    },
                                   {"method":         httpretty.POST,
                                    "uri":            "/_bulk?refresh=true",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.index_logs_rq_different_log_level_merged),
                                    "rs":             utils.get_fixture(
                                        self.index_logs_rs_different_log_level),
                                    }, ],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs_different_log_level),
                "has_errors":     False,
                "expected_count": 1,
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
                "expected_log_exceptions": [launch_objects.LogExceptionResult(logId=1, foundExceptions=[])]
            }
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                es_client = esclient.EsClient(app_config=app_config,
                                              search_cfg=self.get_default_search_config())
                es_client.es_client.scroll = MagicMock(return_value=json.loads(
                    utils.get_fixture(self.no_hits_search_rs)))
                launches = [launch_objects.Launch(**launch)
                            for launch in json.loads(test["index_rq"])]
                response = es_client.index_logs(launches)

                test["has_errors"].should.equal(response.errors)
                test["expected_count"].should.equal(response.took)
                test["expected_log_exceptions"].should.equal(response.logResults)

                TestEsClient.shutdown_server(test["test_calls"])


if __name__ == '__main__':
    unittest.main()
