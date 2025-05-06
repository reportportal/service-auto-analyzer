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
from http import HTTPStatus
from test import APP_CONFIG, get_fixture
from test.mock_service import TestService

import httpretty

from app.service import DeleteIndexService
from app.utils import utils


class TestDeleteIndexService(TestService):

    @utils.ignore_warnings
    def test_delete_index(self):
        """Test deleting index"""
        tests = [
            {
                "test_calls": [
                    {
                        "method": httpretty.DELETE,
                        "uri": "/1",
                        "status": HTTPStatus.OK,
                        "content_type": "application/json",
                        "rs": get_fixture(self.index_deleted_rs),
                    },
                ],
                "index": 1,
                "result": True,
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.DELETE,
                        "uri": "/2",
                        "status": HTTPStatus.NOT_FOUND,
                        "content_type": "application/json",
                        "rs": get_fixture(self.index_not_found_rs),
                    },
                ],
                "index": 2,
                "result": False,
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.DELETE,
                        "uri": "/rp_2",
                        "status": HTTPStatus.NOT_FOUND,
                        "content_type": "application/json",
                        "rs": get_fixture(self.index_not_found_rs),
                    },
                ],
                "index": 2,
                "app_config": APP_CONFIG,
                "result": False,
            },
        ]
        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                _delete_index_service = DeleteIndexService(
                    self.model_chooser, app_config=app_config, search_cfg=self.get_default_search_config()
                )

                response = _delete_index_service.delete_index(test["index"])

                assert test["result"] == response

                TestDeleteIndexService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f"Error in the test case number: {idx}").with_traceback(err.__traceback__)


if __name__ == "__main__":
    unittest.main()
