"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
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

from app.commons.model import launch_objects
from app.service import SuggestInfoService
from app.utils import utils
from test import APP_CONFIG, get_fixture
from test.mock_service import TestService
from test.service import (
    get_clean_index_object,
    get_defect_update_info,
    get_delete_by_query_call,
    get_delete_suggest_index_call,
    get_item_remove_info,
    get_launch_remove_info,
    get_suggest_index_creation_calls,
    get_suggest_index_found_call,
    get_suggest_index_not_found_call,
    get_suggest_info_cleanup_calls,
)


class TestSuggestInfoService(TestService):

    @utils.ignore_warnings
    def test_clean_suggest_info_logs(self):
        """Test cleaning suggest info logs"""

        # Common test data
        basic_clean_index = get_clean_index_object([1], 1)
        cleanup_calls = get_suggest_info_cleanup_calls(
            "1",
            self.search_suggest_info_ids_query,
            self.one_hit_search_suggest_info_rs,
            self.delete_suggest_logs_rq,
            self.delete_logs_rs,
        )
        cleanup_calls_with_prefix = get_suggest_info_cleanup_calls(
            "rp_1",
            self.search_suggest_info_ids_query,
            self.one_hit_search_suggest_info_rs,
            self.delete_suggest_logs_rq_with_prefix,
            self.delete_logs_rs,
        )

        tests = [
            # Test case 0: Index not found
            {
                "test_calls": [get_suggest_index_not_found_call("2")],
                "rq": get_clean_index_object([1], 2),
                "expected_count": 0,
            },
            # Test case 1: Index found with cleanup
            {
                "test_calls": cleanup_calls,
                "rq": basic_clean_index,
                "expected_count": 1,
            },
            # Test case 2: Index found with app config and cleanup
            {
                "test_calls": cleanup_calls_with_prefix,
                "app_config": APP_CONFIG,
                "rq": basic_clean_index,
                "expected_count": 1,
            },
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)

                suggest_info_service.es_client.es_client.scroll = MagicMock(
                    return_value=json.loads(get_fixture(self.no_hits_search_rs))
                )
                response = suggest_info_service.clean_suggest_info_logs(test["rq"])
                assert test["expected_count"] == response
                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f"Error in the test case number: {idx}").with_traceback(err.__traceback__)

    @utils.ignore_warnings
    def test_delete_suggest_info_index(self):
        """Test deleting an index"""

        tests = [
            # Test case 0: Successful deletion
            {
                "test_calls": [get_delete_suggest_index_call("1", HTTPStatus.OK, self.index_deleted_rs)],
                "index": 1,
                "result": True,
            },
            # Test case 1: Index not found
            {
                "test_calls": [get_delete_suggest_index_call("2", HTTPStatus.NOT_FOUND, self.index_not_found_rs)],
                "index": 2,
                "result": False,
            },
            # Test case 2: Index not found with app config
            {
                "test_calls": [get_delete_suggest_index_call("rp_2", HTTPStatus.NOT_FOUND, self.index_not_found_rs)],
                "app_config": APP_CONFIG,
                "index": 2,
                "result": False,
            },
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
                raise AssertionError(f"Error in the test case number: {idx}").with_traceback(err.__traceback__)

    @utils.ignore_warnings
    def test_index_suggest_info_logs(self):
        """Test indexing suggest info"""

        tests = [
            # Test case 0: Empty index request
            {
                "test_calls": [],
                "index_rq": "[]",
                "has_errors": False,
                "expected_count": 0,
            },
            # Test case 1: Index creation - both not found
            {
                "test_calls": get_suggest_index_creation_calls(
                    HTTPStatus.NOT_FOUND,
                    HTTPStatus.NOT_FOUND,
                    self.index_created_rs,
                    self.index_created_rs,
                    "1",
                    self.index_logs_rs,
                ),
                "index_rq": get_fixture(self.suggest_info_list),
                "has_errors": False,
                "expected_count": 2,
            },
            # Test case 2: Index creation - both found
            {
                "test_calls": get_suggest_index_creation_calls(
                    HTTPStatus.OK, HTTPStatus.OK, self.index_created_rs, self.index_created_rs, "1", self.index_logs_rs
                ),
                "index_rq": get_fixture(self.suggest_info_list),
                "has_errors": False,
                "expected_count": 2,
            },
            # Test case 3: Index creation with app config
            {
                "test_calls": get_suggest_index_creation_calls(
                    HTTPStatus.OK,
                    HTTPStatus.OK,
                    self.index_created_rs,
                    self.index_created_rs,
                    "rp_1",
                    self.index_logs_rs,
                ),
                "app_config": APP_CONFIG,
                "index_rq": get_fixture(self.suggest_info_list),
                "has_errors": False,
                "expected_count": 2,
            },
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)
                response = suggest_info_service.index_suggest_info(
                    [launch_objects.SuggestAnalysisResult(**res) for res in json.loads(test["index_rq"])]
                )

                assert test["has_errors"] == response.errors
                assert test["expected_count"] == response.took

                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f"Error in the test case number: {idx}").with_traceback(err.__traceback__)

    def test_remove_test_items_suggests(self):
        """Test removing test items from suggests"""

        # Common test data
        item_remove_info = get_item_remove_info(1, [1, 2])

        tests = [
            # Test case 0: Index not found
            {
                "test_calls": [get_suggest_index_not_found_call("1")],
                "item_remove_info": item_remove_info,
                "result": 0,
            },
            # Test case 1: Index found with deletion
            {
                "test_calls": [
                    get_suggest_index_found_call("1"),
                    get_delete_by_query_call("1", self.delete_by_query_suggest_1, {"deleted": 1}),
                ],
                "item_remove_info": item_remove_info,
                "result": 1,
            },
            # Test case 2: Index found with app config
            {
                "test_calls": [
                    get_suggest_index_found_call("rp_1"),
                    get_delete_by_query_call("rp_1", self.delete_by_query_suggest_1, {"deleted": 3}),
                ],
                "app_config": APP_CONFIG,
                "item_remove_info": item_remove_info,
                "result": 3,
            },
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)
                response = suggest_info_service.clean_suggest_info_logs_by_test_item(test["item_remove_info"])

                assert test["result"] == response

                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f"Error in the test case number: {idx}").with_traceback(err.__traceback__)

    def test_remove_launches_suggests(self):
        """Test removing launches from suggests"""

        # Common test data
        launch_remove_info = get_launch_remove_info(1, [1, 2])

        tests = [
            # Test case 0: Index not found
            {
                "test_calls": [get_suggest_index_not_found_call("1")],
                "launch_remove_info": launch_remove_info,
                "result": 0,
            },
            # Test case 1: Index found with deletion
            {
                "test_calls": [
                    get_suggest_index_found_call("1"),
                    get_delete_by_query_call("1", self.delete_by_query_suggest_2, {"deleted": 1}),
                ],
                "launch_remove_info": launch_remove_info,
                "result": 1,
            },
            # Test case 2: Index found with app config
            {
                "test_calls": [
                    get_suggest_index_found_call("rp_1"),
                    get_delete_by_query_call("rp_1", self.delete_by_query_suggest_2, {"deleted": 3}),
                ],
                "app_config": APP_CONFIG,
                "launch_remove_info": launch_remove_info,
                "result": 3,
            },
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)
                response = suggest_info_service.clean_suggest_info_logs_by_launch_id(test["launch_remove_info"])

                assert test["result"] == response

                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f"Error in the test case number: {idx}").with_traceback(err.__traceback__)

    def test_suggest_info_update(self):
        """Test updating suggest info"""

        # Common test data
        defect_update_info = get_defect_update_info(1, {1: "pb001", 2: "ab001"})

        tests = [
            # Test case 0: Index not found
            {
                "test_calls": [get_suggest_index_not_found_call("1")],
                "defect_update_info": defect_update_info,
                "result": 0,
            },
            # Test case 1: Index found with update (result 1)
            {
                "test_calls": [
                    get_suggest_index_found_call("1"),
                    {
                        "method": httpretty.GET,
                        "uri": "/1_suggest/_search?scroll=5m&size=1000",
                        "status": HTTPStatus.OK,
                        "content_type": "application/json",
                        "rq": get_fixture(self.get_suggest_info_by_test_item_ids_query),
                        "rs": get_fixture(self.suggest_info_test_items_by_id_1),
                    },
                    {
                        "method": httpretty.POST,
                        "uri": "/_bulk?refresh=true",
                        "status": HTTPStatus.OK,
                        "content_type": "application/json",
                        "rq": get_fixture(self.suggest_index_test_item_update),
                        "rs": get_fixture(self.index_logs_rs),
                    },
                ],
                "defect_update_info": defect_update_info,
                "result": 1,
            },
            # Test case 2: Index found with update (result 2)
            {
                "test_calls": [
                    get_suggest_index_found_call("1"),
                    {
                        "method": httpretty.GET,
                        "uri": "/1_suggest/_search?scroll=5m&size=1000",
                        "status": HTTPStatus.OK,
                        "content_type": "application/json",
                        "rq": get_fixture(self.get_suggest_info_by_test_item_ids_query),
                        "rs": get_fixture(self.suggest_info_test_items_by_id_2),
                    },
                    {
                        "method": httpretty.POST,
                        "uri": "/_bulk?refresh=true",
                        "status": HTTPStatus.OK,
                        "content_type": "application/json",
                        "rq": get_fixture(self.suggest_index_test_item_update_2),
                        "rs": get_fixture(self.index_logs_rs),
                    },
                ],
                "defect_update_info": defect_update_info,
                "result": 2,
            },
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                suggest_info_service = SuggestInfoService(app_config=app_config)
                suggest_info_service.es_client.es_client.scroll = MagicMock(
                    return_value=json.loads(get_fixture(self.no_hits_search_rs))
                )
                response = suggest_info_service.update_suggest_info(test["defect_update_info"])

                assert test["result"] == response

                TestSuggestInfoService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f"Error in the test case number: {idx}").with_traceback(err.__traceback__)


if __name__ == "__main__":
    unittest.main()
