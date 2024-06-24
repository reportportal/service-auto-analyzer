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

import json
import unittest
from http import HTTPStatus
from unittest.mock import MagicMock

import httpretty

from app.commons import launch_objects
from app.service import SearchService
from app.utils import utils
from test import get_fixture, APP_CONFIG
from test.mock_service import TestService


class TestSearchService(TestService):

    @utils.ignore_warnings
    def test_search_logs(self):
        """Test search logs"""
        tests = [
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq),
                                "rs": get_fixture(
                                    self.no_hits_search_rs),
                                }, ],
                "rq": launch_objects.SearchLogs(launchId=1,
                                                launchName="Launch 1",
                                                itemId=3,
                                                projectId=1,
                                                filteredLaunchIds=[1],
                                                logMessages=["error"],
                                                logLines=-1),
                "expected_count": 0
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/rp_1",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/rp_1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq),
                                "rs": get_fixture(
                                    self.no_hits_search_rs),
                                }, ],
                "rq": launch_objects.SearchLogs(launchId=1,
                                                launchName="Launch 1",
                                                itemId=3,
                                                projectId=1,
                                                filteredLaunchIds=[1],
                                                logMessages=["error"],
                                                logLines=-1),
                "app_config": APP_CONFIG,
                "expected_count": 0
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1",
                                "status": HTTPStatus.OK,
                                }, ],
                "rq": launch_objects.SearchLogs(launchId=1,
                                                launchName="Launch 1",
                                                itemId=3,
                                                projectId=1,
                                                filteredLaunchIds=[1],
                                                logMessages=[""],
                                                logLines=-1),
                "expected_count": 0
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq),
                                "rs": get_fixture(
                                    self.one_hit_search_rs_search_logs),
                                }, ],
                "rq": launch_objects.SearchLogs(launchId=1,
                                                launchName="Launch 1",
                                                itemId=3,
                                                projectId=1,
                                                filteredLaunchIds=[1],
                                                logMessages=["error"],
                                                logLines=-1),
                "expected_count": 0
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.search_logs_rq_not_found),
                                "rs": get_fixture(
                                    self.two_hits_search_rs_search_logs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.search_not_merged_logs_by_test_item),
                                "rs": get_fixture(
                                    self.two_hits_search_rs_search_logs),
                                }, ],
                "rq": launch_objects.SearchLogs(launchId=1,
                                                launchName="Launch 1",
                                                itemId=3,
                                                projectId=1,
                                                filteredLaunchIds=[1],
                                                logMessages=["error occurred once"],
                                                logLines=-1),
                "expected_count": 1
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_with_status_codes),
                                "rs": get_fixture(self.two_hits_search_rs_search_logs_with_status_codes),
                                },
                               {"method": httpretty.GET,
                                "uri": "/1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_not_merged_logs_by_test_item),
                                "rs": get_fixture(self.two_hits_search_rs_search_logs_with_status_codes),
                                }],
                "rq": launch_objects.SearchLogs(
                    launchId=1,
                    launchName="Launch 1",
                    itemId=3,
                    projectId=1,
                    filteredLaunchIds=[1],
                    logMessages=["error occurred once status code: 500 but got 200"],
                    logLines=-1),
                "expected_count": 1,
                "response": [launch_objects.SearchLogInfo(logId=2, testItemId=1, matchScore=95)]
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/rp_1",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/rp_1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_not_found),
                                "rs": get_fixture(self.two_hits_search_rs_search_logs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/rp_1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_not_merged_logs_by_test_item),
                                "rs": get_fixture(self.two_hits_search_rs_search_logs),
                                }],
                "rq": launch_objects.SearchLogs(launchId=1,
                                                launchName="Launch 1",
                                                itemId=3,
                                                projectId=1,
                                                filteredLaunchIds=[1],
                                                logMessages=["error occurred once"],
                                                logLines=-1),
                "app_config": APP_CONFIG,
                "expected_count": 1,
                "response": [launch_objects.SearchLogInfo(logId=1, testItemId=1, matchScore=95)]
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_not_found),
                                "rs": get_fixture(self.two_hits_search_rs_search_logs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/1/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_not_merged_logs_by_test_item),
                                "rs": get_fixture(self.two_hits_search_rs_search_logs),
                                }],
                "rq": launch_objects.SearchLogs(launchId=1,
                                                launchName="Launch 1",
                                                itemId=3,
                                                projectId=1,
                                                filteredLaunchIds=[1],
                                                logMessages=["error occurred once"],
                                                logLines=-1,
                                                analyzerConfig=launch_objects.AnalyzerConf(
                                                    allMessagesShouldMatch=True)),
                "expected_count": 1,
                "response": [launch_objects.SearchLogInfo(logId=1, testItemId=1, matchScore=95)]
            }
        ]

        for idx, test in enumerate(tests):
            print(f'Running test case idx: {idx}')
            self._start_server(test["test_calls"])
            app_config = self.app_config
            if "app_config" in test:
                app_config = test["app_config"]
            search_service = SearchService(app_config=app_config, search_cfg=self.get_default_search_config())

            search_service.es_client.es_client.scroll = MagicMock(return_value=json.loads(
                get_fixture(self.no_hits_search_rs)))

            response = search_service.search_logs(test["rq"])
            assert len(response) == test["expected_count"]
            if "response" in test:
                assert response == test["response"]

            TestSearchService.shutdown_server(test["test_calls"])


if __name__ == '__main__':
    unittest.main()
