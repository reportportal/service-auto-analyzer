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
import httpretty

import commons.launch_objects as launch_objects
from boosting_decision_making.boosting_decision_maker import BoostingDecisionMaker
from service.auto_analyzer_service import AutoAnalyzerService
from test.test_service import TestService
from utils import utils


class TestAutoAnalyzerService(TestService):

    @utils.ignore_warnings
    def test_analyze_logs(self):
        """Test analyzing logs"""
        tests = [
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.OK,
                                         }, ],
                "index_rq":            utils.get_fixture(self.launch_wo_test_items),
                "expected_count":      0,
                "expected_issue_type": "",
                "boost_predict":       ([], [])
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.OK,
                                         }, ],
                "index_rq":            utils.get_fixture(
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
                "index_rq":            utils.get_fixture(
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
                "msearch_results": [utils.get_fixture(self.no_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True)],
                "index_rq":       utils.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count":      0,
                "expected_issue_type": "",
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.NOT_FOUND,
                                    }, ],
                "index_rq":       utils.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count":      0,
                "expected_issue_type": "",
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [utils.get_fixture(self.no_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.one_hit_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True)],
                "index_rq":       utils.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict":       ([1], [[0.2, 0.8]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [utils.get_fixture(self.one_hit_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.two_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True)],
                "index_rq":       utils.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict":       ([1, 0], [[0.2, 0.8], [0.7, 0.3]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [utils.get_fixture(self.two_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.two_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.three_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True)],
                "index_rq":       utils.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict":       ([1, 1], [[0.2, 0.8], [0.3, 0.7]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [utils.get_fixture(self.no_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.three_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.three_hits_search_rs, to_json=True)],
                "index_rq":       utils.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "PB001",
                "boost_predict":       ([0, 1], [[0.8, 0.2], [0.3, 0.7]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [utils.get_fixture(self.no_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.three_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.no_hits_search_rs, to_json=True)],
                "index_rq":       utils.get_fixture(self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict":       ([1, 0], [[0.2, 0.8], [0.7, 0.3]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [utils.get_fixture(self.two_hits_search_rs, to_json=True),
                                    utils.get_fixture(self.two_hits_search_rs, to_json=True)],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs_to_be_merged),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True),
                    utils.get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True)],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "boost_predict":       ([1, 0], [[0.2, 0.8], [0.7, 0.3]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/rp_2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True),
                    utils.get_fixture(self.no_hits_search_rs, to_json=True)],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
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
                "boost_predict":       ([1, 0], [[0.2, 0.8], [0.7, 0.3]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.two_hits_search_rs, to_json=True),
                    utils.get_fixture(self.two_hits_with_no_defect, to_json=True)],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_id": 34,
                "expected_issue_type": "ND001",
                "boost_predict":       ([1], [[0.1, 0.9]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.two_hits_search_rs, to_json=True),
                    utils.get_fixture(self.two_hits_with_no_defect, to_json=True)],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "boost_predict":       ([0], [[0.9, 0.1]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.three_hits_search_rs, to_json=True),
                    utils.get_fixture(self.three_hits_with_no_defect, to_json=True)],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_id": 2,
                "expected_issue_type": "PB001",
                "boost_predict":       ([0, 1], [[0.8, 0.2], [0.3, 0.7]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True),
                    utils.get_fixture(self.three_hits_search_rs_with_one_unique_id, to_json=True)],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "analyzer_config":     launch_objects.AnalyzerConf(allMessagesShouldMatch=True),
                "boost_predict":       ([], [])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [
                    utils.get_fixture(self.two_hits_search_rs_second_message, to_json=True),
                    utils.get_fixture(self.two_hits_search_rs_second_message, to_json=True),
                    utils.get_fixture(self.two_hits_search_rs, to_json=True),
                    utils.get_fixture(self.two_hits_search_rs, to_json=True)],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs),
                "expected_count": 1,
                "expected_issue_type": "AB001",
                "analyzer_config":     launch_objects.AnalyzerConf(allMessagesShouldMatch=True),
                "boost_predict":       ([1], [[0.3, 0.7]])
            },
            {
                "test_calls":     [{"method":         httpretty.GET,
                                    "uri":            "/2",
                                    "status":         HTTPStatus.OK,
                                    }],
                "msearch_results": [
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.no_hits_search_rs, to_json=True),
                    utils.get_fixture(self.two_hits_search_rs, to_json=True),
                    utils.get_fixture(self.two_hits_with_no_defect, to_json=True)],
                "index_rq":       utils.get_fixture(
                    self.launch_w_test_items_w_logs),
                "expected_count": 0,
                "expected_issue_type": "",
                "analyzer_config":     launch_objects.AnalyzerConf(allMessagesShouldMatch=True),
                "boost_predict":       ([], [])
            }
        ]

        for idx, test in enumerate(tests):
            try:
            # with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                config = self.get_default_search_config()
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                analyzer_service = AutoAnalyzerService(self.model_chooser,
                                                       app_config=app_config,
                                                       search_cfg=config)
                _boosting_decision_maker = BoostingDecisionMaker()
                _boosting_decision_maker.get_feature_ids = MagicMock(return_value=[0])
                _boosting_decision_maker.get_feature_names = MagicMock(return_value=["0"])
                _boosting_decision_maker.predict = MagicMock(return_value=test["boost_predict"])
                if "msearch_results" in test:
                    analyzer_service.es_client.es_client.msearch = MagicMock(
                        return_value={"responses": test["msearch_results"]})
                analyzer_service.model_chooser.choose_model = MagicMock(
                    return_value=_boosting_decision_maker)

                launches = [launch_objects.Launch(**launch)
                            for launch in json.loads(test["index_rq"])]
                if "analyzer_config" in test:
                    for launch in launches:
                        launch.analyzerConfig = test["analyzer_config"]
                response = analyzer_service.analyze_logs(launches)

                # response.should.have.length_of(test["expected_count"])
                assert len(response) == test["expected_count"]

                if test["expected_issue_type"] != "":
                    # test["expected_issue_type"].should.equal(response[0].issueType)
                    assert response[0].issueType == test["expected_issue_type"]

                if "expected_id" in test:
                    # test["expected_id"].should.equal(response[0].relevantItem)
                    assert response[0].relevantItem == test["expected_id"]

                TestAutoAnalyzerService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}').with_traceback(err.__traceback__)

if __name__ == '__main__':
    unittest.main()
