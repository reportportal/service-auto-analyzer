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
from app.service import SuggestInfoService
from app.utils import utils
from test import get_fixture
from test.mock_service import TestService


class TestSuggestInfoService(TestService):

    @utils.ignore_warnings
    def test_clean_suggest_info_logs(self):
        """Test cleaning suggest info logs"""
        tests = [
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/2_suggest",
                                "status": HTTPStatus.NOT_FOUND,
                                }, ],
                "rq": launch_objects.CleanIndex(ids=[1], project=2),
                "expected_count": 0
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/1_suggest/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_suggest_info_ids_query),
                                "rs": get_fixture(
                                    self.one_hit_search_suggest_info_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=true",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.delete_suggest_logs_rq),
                                "rs": get_fixture(self.delete_logs_rs),
                                }],
                "rq": launch_objects.CleanIndex(ids=[1], project=1),
                "expected_count": 1
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/rp_1_suggest",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/rp_1_suggest/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_suggest_info_ids_query),
                                "rs": get_fixture(
                                    self.one_hit_search_suggest_info_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=true",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.delete_suggest_logs_rq_with_prefix),
                                "rs": get_fixture(self.delete_logs_rs),
                                }],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts": False,
                    "esUseSsl": False,
                    "esSslShowWarn": False,
                    "turnOffSslVerification": True,
                    "esCAcert": "",
                    "esClientCert": "",
                    "esClientKey": "",
                    "appVersion": "",
                    "minioRegion": "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber": 1000,
                    "binaryStoreType": "filesystem",
                    "minioHost": "",
                    "minioAccessKey": "",
                    "minioSecretKey": "",
                    "esProjectIndexPrefix": "rp_"
                },
                "rq": launch_objects.CleanIndex(ids=[1], project=1),
                "expected_count": 1
            }
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)

                suggest_info_service.es_client.es_client.scroll = MagicMock(
                    return_value=json.loads(get_fixture(self.no_hits_search_rs)))
                response = suggest_info_service.clean_suggest_info_logs(test["rq"])
                assert test["expected_count"] == response
                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}'). \
                    with_traceback(err.__traceback__)

    @utils.ignore_warnings
    def test_delete_suggest_info_index(self):
        """Test deleting an index"""
        tests = [
            {
                "test_calls": [{"method": httpretty.DELETE,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rs": get_fixture(self.index_deleted_rs),
                                }, ],
                "index": 1,
                "result": True,
            },
            {
                "test_calls": [{"method": httpretty.DELETE,
                                "uri": "/2_suggest",
                                "status": HTTPStatus.NOT_FOUND,
                                "content_type": "application/json",
                                "rs": get_fixture(self.index_not_found_rs),
                                }, ],
                "index": 2,
                "result": False,
            },
            {
                "test_calls": [{"method": httpretty.DELETE,
                                "uri": "/rp_2_suggest",
                                "status": HTTPStatus.NOT_FOUND,
                                "content_type": "application/json",
                                "rs": get_fixture(self.index_not_found_rs),
                                }, ],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts": False,
                    "esUseSsl": False,
                    "esSslShowWarn": False,
                    "turnOffSslVerification": True,
                    "esCAcert": "",
                    "esClientCert": "",
                    "esClientKey": "",
                    "appVersion": "",
                    "minioRegion": "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber": 1000,
                    "binaryStoreType": "filesystem",
                    "minioHost": "",
                    "minioAccessKey": "",
                    "minioSecretKey": "",
                    "esProjectIndexPrefix": "rp_"
                },
                "index": 2,
                "result": False,
            }
        ]
        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)

                response = suggest_info_service.remove_suggest_info(test["index"])

                assert test["result"] == response

                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}'). \
                    with_traceback(err.__traceback__)

    @utils.ignore_warnings
    def test_index_suggest_info_logs(self):
        """Test indexing suggest info"""
        tests = [
            {
                "test_calls": [],
                "index_rq": "[]",
                "has_errors": False,
                "expected_count": 0
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/rp_suggestions_info_metrics",
                                "status": HTTPStatus.NOT_FOUND
                                },
                               {"method": httpretty.PUT,
                                "uri": "/rp_suggestions_info_metrics",
                                "status": HTTPStatus.OK,
                                "rs": get_fixture(self.index_created_rs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.NOT_FOUND
                                },
                               {"method": httpretty.PUT,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.OK,
                                "rs": get_fixture(self.index_created_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=true",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rs": get_fixture(self.index_logs_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=true",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "index_rq": get_fixture(self.suggest_info_list),
                "has_errors": False,
                "expected_count": 2
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/rp_suggestions_info_metrics",
                                "status": HTTPStatus.OK
                                },
                               {"method": httpretty.PUT,
                                "uri": "/rp_suggestions_info_metrics/_mapping",
                                "status": HTTPStatus.OK,
                                "rs": get_fixture(self.index_created_rs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.OK
                                },
                               {"method": httpretty.PUT,
                                "uri": "/1_suggest/_mapping",
                                "status": HTTPStatus.OK,
                                "rs": get_fixture(self.index_created_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=true",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rs": get_fixture(self.index_logs_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=true",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "index_rq": get_fixture(self.suggest_info_list),
                "has_errors": False,
                "expected_count": 2
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/rp_suggestions_info_metrics",
                                "status": HTTPStatus.OK
                                },
                               {"method": httpretty.PUT,
                                "uri": "/rp_suggestions_info_metrics/_mapping",
                                "status": HTTPStatus.OK,
                                "rs": get_fixture(self.index_created_rs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/rp_1_suggest",
                                "status": HTTPStatus.OK
                                },
                               {"method": httpretty.PUT,
                                "uri": "/rp_1_suggest/_mapping",
                                "status": HTTPStatus.OK,
                                "rs": get_fixture(self.index_created_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=true",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rs": get_fixture(self.index_logs_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=true",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts": False,
                    "esUseSsl": False,
                    "esSslShowWarn": False,
                    "turnOffSslVerification": True,
                    "esCAcert": "",
                    "esClientCert": "",
                    "esClientKey": "",
                    "appVersion": "",
                    "minioRegion": "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber": 1000,
                    "binaryStoreType": "filesystem",
                    "minioHost": "",
                    "minioAccessKey": "",
                    "minioSecretKey": "",
                    "esProjectIndexPrefix": "rp_"
                },
                "index_rq": get_fixture(self.suggest_info_list),
                "has_errors": False,
                "expected_count": 2
            }
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)
                response = suggest_info_service.index_suggest_info(
                    [launch_objects.SuggestAnalysisResult(**res) for res in json.loads(test["index_rq"])])

                assert test["has_errors"] == response.errors
                assert test["expected_count"] == response.took

                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}'). \
                    with_traceback(err.__traceback__)

    def test_remove_test_items_suggests(self):
        tests = [
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.NOT_FOUND,
                                "content_type": "application/json",
                                }],
                "item_remove_info": {
                    "project": 1,
                    "itemsToDelete": [1, 2]},
                "result": 0
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                },
                               {"method": httpretty.POST,
                                "uri": "/1_suggest/_delete_by_query",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.delete_by_query_suggest_1),
                                "rs": json.dumps({"deleted": 1})}],
                "item_remove_info": {
                    "project": 1,
                    "itemsToDelete": [1, 2]},
                "result": 1
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/rp_1_suggest",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                },
                               {"method": httpretty.POST,
                                "uri": "/rp_1_suggest/_delete_by_query",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.delete_by_query_suggest_1),
                                "rs": json.dumps({"deleted": 3}),
                                }],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts": False,
                    "esUseSsl": False,
                    "esSslShowWarn": False,
                    "turnOffSslVerification": True,
                    "esCAcert": "",
                    "esClientCert": "",
                    "esClientKey": "",
                    "appVersion": "",
                    "minioRegion": "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber": 1000,
                    "binaryStoreType": "filesystem",
                    "minioHost": "",
                    "minioAccessKey": "",
                    "minioSecretKey": "",
                    "esProjectIndexPrefix": "rp_"
                },
                "item_remove_info": {
                    "project": 1,
                    "itemsToDelete": [1, 2]},
                "result": 3
            }
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)
                response = suggest_info_service.clean_suggest_info_logs_by_test_item(
                    test["item_remove_info"])

                assert test["result"] == response

                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}'). \
                    with_traceback(err.__traceback__)

    def test_remove_launches_suggests(self):
        tests = [
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.NOT_FOUND,
                                "content_type": "application/json",
                                }],
                "launch_remove_info": {
                    "project": 1,
                    "launch_ids": [1, 2]},
                "result": 0
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                },
                               {"method": httpretty.POST,
                                "uri": "/1_suggest/_delete_by_query",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.delete_by_query_suggest_2),
                                "rs": json.dumps({"deleted": 1})}],
                "launch_remove_info": {
                    "project": 1,
                    "launch_ids": [1, 2]},
                "result": 1
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/rp_1_suggest",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                },
                               {"method": httpretty.POST,
                                "uri": "/rp_1_suggest/_delete_by_query",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.delete_by_query_suggest_2),
                                "rs": json.dumps({"deleted": 3}),
                                }],
                "app_config": {
                    "esHost": "http://localhost:9200",
                    "esUser": "",
                    "esPassword": "",
                    "esVerifyCerts": False,
                    "esUseSsl": False,
                    "esSslShowWarn": False,
                    "turnOffSslVerification": True,
                    "esCAcert": "",
                    "esClientCert": "",
                    "esClientKey": "",
                    "appVersion": "",
                    "minioRegion": "",
                    "minioBucketPrefix": "",
                    "filesystemDefaultPath": "",
                    "esChunkNumber": 1000,
                    "binaryStoreType": "filesystem",
                    "minioHost": "",
                    "minioAccessKey": "",
                    "minioSecretKey": "",
                    "esProjectIndexPrefix": "rp_"
                },
                "launch_remove_info": {
                    "project": 1,
                    "launch_ids": [1, 2]},
                "result": 3
            }
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)
                response = suggest_info_service.clean_suggest_info_logs_by_launch_id(
                    test["launch_remove_info"])

                assert test["result"] == response

                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}'). \
                    with_traceback(err.__traceback__)

    def test_suggest_info_update(self):
        tests = [
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.NOT_FOUND,
                                "content_type": "application/json",
                                }],
                "defect_update_info": {
                    "project": 1,
                    "itemsToUpdate": {1: "pb001", 2: "ab001"}},
                "result": 0
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                },
                               {"method": httpretty.GET,
                                "uri": "/1_suggest/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.get_suggest_info_by_test_item_ids_query),
                                "rs": get_fixture(
                                    self.suggest_info_test_items_by_id_1),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=true",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.suggest_index_test_item_update),
                                "rs": get_fixture(
                                    self.index_logs_rs),
                                }],
                "defect_update_info": {
                    "project": 1,
                    "itemsToUpdate": {1: "pb001", 2: "ab001"}},
                "result": 1
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1_suggest",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                },
                               {"method": httpretty.GET,
                                "uri": "/1_suggest/_search?scroll=5m&size=1000",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.get_suggest_info_by_test_item_ids_query),
                                "rs": get_fixture(
                                    self.suggest_info_test_items_by_id_2),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=true",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.suggest_index_test_item_update_2),
                                "rs": get_fixture(
                                    self.index_logs_rs),
                                }],
                "defect_update_info": {
                    "project": 1,
                    "itemsToUpdate": {1: "pb001", 2: "ab001"}},
                "result": 2
            },
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)
                suggest_info_service.es_client.es_client.scroll = MagicMock(return_value=json.loads(
                    get_fixture(self.no_hits_search_rs)))
                response = suggest_info_service.update_suggest_info(test["defect_update_info"])

                assert test["result"] == response

                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}'). \
                    with_traceback(err.__traceback__)


if __name__ == '__main__':
    unittest.main()
