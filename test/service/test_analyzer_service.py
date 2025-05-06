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

import json
import unittest
from http import HTTPStatus
from test import APP_CONFIG, get_fixture
from test.mock_service import TestService
from unittest.mock import MagicMock

import httpretty

from app.commons import object_saving
from app.commons.model import launch_objects
from app.machine_learning.models.boosting_decision_maker import BoostingDecisionMaker
from app.service import AutoAnalyzerService
from app.utils import utils


class TestAutoAnalyzerService(TestService):

    @utils.ignore_warnings
    def test_analyze_logs(self):
        """Test analyzing logs"""
        tests = [
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/1",
                        "status": HTTPStatus.OK,
                    },
                ],
                "index_rq": get_fixture(self.launch_wo_test_items),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/1",
                        "status": HTTPStatus.OK,
                    },
                ],
                "index_rq": get_fixture(self.launch_w_test_items_wo_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    },
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_empty_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    },
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.NOT_FOUND,
                    },
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.one_hit_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict": ([1], [[0.2, 0.8]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict": ([1, 0], [[0.2, 0.8], [0.7, 0.3]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict": ([1, 1], [[0.2, 0.8], [0.3, 0.7]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_search_rs, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "PB001",
                "boost_predict": ([0, 1], [[0.8, 0.2], [0.3, 0.7]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict": ([1, 0], [[0.2, 0.8], [0.7, 0.3]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs_to_be_merged),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True),
                    get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict": ([1, 0], [[0.2, 0.8], [0.7, 0.3]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/rp_2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "app_config": APP_CONFIG,
                "boost_predict": ([1, 0], [[0.2, 0.8], [0.7, 0.3]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_with_no_defect, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_id": 34,
                "expected_issue_type": "ND001",
                "boost_predict": ([1], [[0.1, 0.9]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_with_no_defect, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict": ([0], [[0.9, 0.1]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_with_no_defect, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_id": 2,
                "expected_issue_type": "PB001",
                "boost_predict": ([0, 1], [[0.8, 0.2], [0.3, 0.7]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True),
                    get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "analyzer_config": launch_objects.AnalyzerConf(allMessagesShouldMatch=True),
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.two_hits_search_rs_second_message, to_json=True),
                    get_fixture(self.two_hits_search_rs_second_message, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "analyzer_config": launch_objects.AnalyzerConf(allMessagesShouldMatch=True),
                "boost_predict": ([1], [[0.3, 0.7]]),
            },
            {
                "test_calls": [
                    {
                        "method": httpretty.GET,
                        "uri": "/2",
                        "status": HTTPStatus.OK,
                    }
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_with_no_defect, to_json=True),
                ],
                "index_rq": get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "analyzer_config": launch_objects.AnalyzerConf(allMessagesShouldMatch=True),
                "boost_predict": ([], []),
            },
        ]

        for idx, test in enumerate(tests):
            print(f"Running test case idx: {idx}")
            self._start_server(test["test_calls"])
            config = self.get_default_search_config()
            app_config = self.app_config
            if "app_config" in test:
                app_config = test["app_config"]
            analyzer_service = AutoAnalyzerService(self.model_chooser, app_config=app_config, search_cfg=config)
            _boosting_decision_maker = BoostingDecisionMaker(object_saving.create_filesystem(""), "", features=[0])
            _boosting_decision_maker.predict = MagicMock(return_value=test["boost_predict"])
            if "msearch_results" in test:
                analyzer_service.es_client.es_client.msearch = MagicMock(
                    return_value={"responses": test["msearch_results"]}
                )
            analyzer_service.model_chooser.choose_model = MagicMock(return_value=_boosting_decision_maker)

            launches = [launch_objects.Launch(**launch) for launch in json.loads(test["index_rq"])]
            if "analyzer_config" in test:
                for launch in launches:
                    launch.analyzerConfig = test["analyzer_config"]
            response = analyzer_service.analyze_logs(launches)

            assert len(response) == test["expected_count"]

            if test["expected_issue_type"] != "":
                assert response[0].issueType == test["expected_issue_type"]

            if "expected_id" in test:
                assert response[0].relevantItem == test["expected_id"]

            TestAutoAnalyzerService.shutdown_server(test["test_calls"])


if __name__ == "__main__":
    unittest.main()
