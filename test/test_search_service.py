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
# import sure # noqa
import httpretty

import commons.launch_objects as launch_objects
from service.search_service import SearchService
from test.test_service import TestService
from utils import utils


class TestSearchService(TestService):

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
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_logs_rq),
                                    "rs":             utils.get_fixture(
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
                                    "uri":            "/rp_1",
                                    "status":         HTTPStatus.OK,
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_logs_rq),
                                    "rs":             utils.get_fixture(
                                        self.no_hits_search_rs),
                                    }, ],
                "rq":             launch_objects.SearchLogs(launchId=1,
                                                            launchName="Launch 1",
                                                            itemId=3,
                                                            projectId=1,
                                                            filteredLaunchIds=[1],
                                                            logMessages=["error"],
                                                            logLines=-1),
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
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(self.search_logs_rq),
                                    "rs":             utils.get_fixture(
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
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.search_logs_rq_not_found),
                                    "rs":             utils.get_fixture(
                                        self.two_hits_search_rs_search_logs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.search_not_merged_logs_by_test_item),
                                    "rs":             utils.get_fixture(
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
                                        self.search_logs_rq_with_status_codes),
                                    "rs":             utils.get_fixture(
                                        self.two_hits_search_rs_search_logs_with_status_codes),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.search_not_merged_logs_by_test_item),
                                    "rs":             utils.get_fixture(
                                        self.two_hits_search_rs_search_logs_with_status_codes),
                                    }],
                "rq":             launch_objects.SearchLogs(
                    launchId=1,
                    launchName="Launch 1",
                    itemId=3,
                    projectId=1,
                    filteredLaunchIds=[1],
                    logMessages=["error occured once status code: 500 but got 200"],
                    logLines=-1),
                "expected_count": 1,
                "response": [launch_objects.SearchLogInfo(logId=2, testItemId=1, matchScore=100)]
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
                                        self.search_logs_rq_not_found),
                                    "rs":             utils.get_fixture(
                                        self.two_hits_search_rs_search_logs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/rp_1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.search_not_merged_logs_by_test_item),
                                    "rs":             utils.get_fixture(
                                        self.two_hits_search_rs_search_logs),
                                    }],
                "rq":             launch_objects.SearchLogs(launchId=1,
                                                            launchName="Launch 1",
                                                            itemId=3,
                                                            projectId=1,
                                                            filteredLaunchIds=[1],
                                                            logMessages=["error occured once"],
                                                            logLines=-1),
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
                "expected_count": 1,
                "response": [launch_objects.SearchLogInfo(logId=1, testItemId=1, matchScore=100)]
            },
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
                                        self.search_logs_rq_not_found),
                                    "rs":             utils.get_fixture(
                                        self.two_hits_search_rs_search_logs),
                                    },
                                   {"method":         httpretty.GET,
                                    "uri":            "/1/_search?scroll=5m&size=1000",
                                    "status":         HTTPStatus.OK,
                                    "content_type":   "application/json",
                                    "rq":             utils.get_fixture(
                                        self.search_not_merged_logs_by_test_item),
                                    "rs":             utils.get_fixture(
                                        self.two_hits_search_rs_search_logs),
                                    }],
                "rq":             launch_objects.SearchLogs(launchId=1,
                                                            launchName="Launch 1",
                                                            itemId=3,
                                                            projectId=1,
                                                            filteredLaunchIds=[1],
                                                            logMessages=["error occured once"],
                                                            logLines=-1,
                                                            analyzerConfig=launch_objects.AnalyzerConf(
                                                                allMessagesShouldMatch=True)),
                "expected_count": 1,
                "response": [launch_objects.SearchLogInfo(logId=1, testItemId=1, matchScore=100)]
            }
        ]

        for idx, test in enumerate(tests):
            # with sure.ensure('Error in the test case number: {0}', idx):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                search_service = SearchService(app_config=app_config,
                                               search_cfg=self.get_default_search_config())

                search_service.es_client.es_client.scroll = MagicMock(return_value=json.loads(
                    utils.get_fixture(self.no_hits_search_rs)))

                response = search_service.search_logs(test["rq"])
                # response.should.have.length_of(test["expected_count"])
                assert len(response) == test["expected_count"]
                if "response" in test:
                    # response.should.equal(test["response"])
                    assert response == test["response"]

                TestSearchService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}').with_traceback(err.__traceback__)


if __name__ == '__main__':
    unittest.main()
