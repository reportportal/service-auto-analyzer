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
from http import HTTPStatus
import logging
import sure # noqa
import httpretty

import commons.launch_objects as launch_objects
from utils import utils
from service.cluster_service import ClusterService


class TestClusterService(unittest.TestCase):
    """Tests cluster service functionality"""

    ERROR_LOGGING_LEVEL = 40000

    @utils.ignore_warnings
    def setUp(self):
        self.launch_wo_test_items = "launch_wo_test_items.json"
        self.launch_w_test_items_wo_logs = "launch_w_test_items_wo_logs.json"
        self.launch_w_test_items_w_empty_logs = "launch_w_test_items_w_empty_logs.json"
        self.index_logs_rs = "index_logs_rs.json"
        self.no_hits_search_rs = "no_hits_search_rs.json"
        self.launch_w_items_clustering = "launch_w_items_clustering.json"
        self.cluster_update_all_the_same = "cluster_update_all_the_same.json"
        self.search_logs_rq_first_group = "search_logs_rq_first_group.json"
        self.search_logs_rq_second_group = "search_logs_rq_second_group.json"
        self.one_hit_search_rs_clustering = "one_hit_search_rs_clustering.json"
        self.search_logs_rq_first_group_2lines = "search_logs_rq_first_group_2lines.json"
        self.cluster_update_es_update = "cluster_update_es_update.json"
        self.cluster_update_all_the_same_es_update = "cluster_update_all_the_same_es_update.json"
        self.cluster_update = "cluster_update.json"
        self.app_config = {
            "esHost": "http://localhost:9200",
            "esVerifyCerts":     False,
            "esUseSsl":          False,
            "esSslShowWarn":     False,
            "esCAcert":          "",
            "esClientCert":      "",
            "esClientKey":       "",
            "appVersion":        ""
        }
        self.model_settings = utils.read_json_file("", "model_settings.json", to_json=True)
        logging.disable(logging.CRITICAL)

    @utils.ignore_warnings
    def tearDown(self):
        logging.disable(logging.DEBUG)

    @utils.ignore_warnings
    def get_default_search_config(self):
        """Get default search config"""
        return {
            "MinShouldMatch": "80%",
            "MinTermFreq":    1,
            "MinDocFreq":     1,
            "BoostAA": -2,
            "BoostLaunch":    2,
            "BoostUniqueID":  2,
            "MaxQueryTerms":  50,
            "SearchLogsMinShouldMatch": "98%",
            "SearchLogsMinSimilarity": 0.9,
            "MinWordLength":  0,
            "BoostModelFolder":
                self.model_settings["BOOST_MODEL_FOLDER"],
            "SimilarityWeightsFolder":
                self.model_settings["SIMILARITY_WEIGHTS_FOLDER"],
            "SuggestBoostModelFolder":
                self.model_settings["SUGGEST_BOOST_MODEL_FOLDER"],
            "GlobalDefectTypeModelFolder":
                self.model_settings["GLOBAL_DEFECT_TYPE_MODEL_FOLDER"]
        }

    @utils.ignore_warnings
    def _start_server(self, test_calls):
        httpretty.reset()
        httpretty.enable(allow_net_connect=False)
        for test_info in test_calls:
            if "content_type" in test_info:
                httpretty.register_uri(
                    test_info["method"],
                    self.app_config["esHost"] + test_info["uri"],
                    body=test_info["rs"] if "rs" in test_info else "",
                    status=test_info["status"],
                    content_type=test_info["content_type"],
                )
            else:
                httpretty.register_uri(
                    test_info["method"],
                    self.app_config["esHost"] + test_info["uri"],
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
    def test_find_clusters(self):
        """Test finding clusters"""
        tests = [
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.OK,
                                         }, ],
                "launch_info":         launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_wo_test_items, to_json=True))[0]),
                    for_update=False,
                    numberOfLogLines=-1),
                "expected_result":     []
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.OK,
                                         }, ],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_w_test_items_wo_logs, to_json=True))[0]),
                    for_update=False,
                    numberOfLogLines=-1),
                "expected_result":     []
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/2",
                                         "status":         HTTPStatus.OK,
                                         }, ],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_w_test_items_w_empty_logs, to_json=True)[0])),
                    for_update=False,
                    numberOfLogLines=-1),
                "expected_result":     []
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/2",
                                         "status":         HTTPStatus.OK,
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=true",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True)),
                    for_update=False,
                    numberOfLogLines=-1),
                "expected_result":     [
                    launch_objects.ClusterResult(
                        logId=4,
                        testItemId=2,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=5,
                        testItemId=5,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=9,
                        testItemId=6,
                        project=2,
                        launchId=1,
                        clusterId="")]
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/2",
                                         "status":         HTTPStatus.OK,
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=true",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update_all_the_same),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True)),
                    for_update=False,
                    numberOfLogLines=2),
                "expected_result":     [
                    launch_objects.ClusterResult(
                        logId=4,
                        testItemId=2,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=5,
                        testItemId=5,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=9,
                        testItemId=6,
                        project=2,
                        launchId=1,
                        clusterId="1")]
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/2",
                                         "status":         HTTPStatus.OK,
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_first_group),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_second_group),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=true",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True)),
                    for_update=True,
                    numberOfLogLines=-1),
                "expected_result":     [
                    launch_objects.ClusterResult(
                        logId=4,
                        testItemId=2,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=5,
                        testItemId=5,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=9,
                        testItemId=6,
                        project=2,
                        launchId=1,
                        clusterId="")]
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/2",
                                         "status":         HTTPStatus.OK,
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_first_group),
                                         "rs":             utils.get_fixture(
                                             self.one_hit_search_rs_clustering),
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_second_group),
                                         "rs":             utils.get_fixture(
                                             self.one_hit_search_rs_clustering),
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=true",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update_es_update),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True)),
                    for_update=True,
                    numberOfLogLines=-1),
                "expected_result":     [
                    launch_objects.ClusterResult(
                        logId=4,
                        testItemId=2,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=5,
                        testItemId=5,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=111,
                        testItemId=12,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=9,
                        testItemId=6,
                        project=2,
                        launchId=1,
                        clusterId="")]
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/2",
                                         "status":         HTTPStatus.OK,
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_first_group_2lines),
                                         "rs":             utils.get_fixture(
                                             self.one_hit_search_rs_clustering),
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=true",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update_all_the_same_es_update),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True)),
                    for_update=True,
                    numberOfLogLines=2),
                "expected_result":     [
                    launch_objects.ClusterResult(
                        logId=4,
                        testItemId=2,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=5,
                        testItemId=5,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=9,
                        testItemId=6,
                        project=2,
                        launchId=1,
                        clusterId="1"),
                    launch_objects.ClusterResult(
                        logId=111,
                        testItemId=12,
                        project=2,
                        launchId=1,
                        clusterId="1")]
            },
        ]

        for idx, test in enumerate(tests):
            with sure.ensure('Error in the test case number: {0}', idx):
                self._start_server(test["test_calls"])
                config = self.get_default_search_config()
                _cluster_service = ClusterService(app_config=self.app_config,
                                                  search_cfg=config)

                response = _cluster_service.find_clusters(test["launch_info"])

                response.should.have.length_of(len(test["expected_result"]))

                cluster_ids_dict = {}
                for i in range(len(response)):
                    test["expected_result"][i].logId.should.equal(response[i].logId)
                    if test["expected_result"][i].clusterId == "":
                        test["expected_result"][i].clusterId.should.equal(response[i].clusterId)
                    elif test["expected_result"][i].clusterId not in cluster_ids_dict:
                        cluster_ids_dict[test["expected_result"][i].clusterId] = response[i].clusterId
                    elif test["expected_result"][i].clusterId in cluster_ids_dict:
                        expected_cluster_id = cluster_ids_dict[test["expected_result"][i].clusterId]
                        expected_cluster_id.should.equal(response[i].clusterId)

                for cluster_id in cluster_ids_dict:
                    test["test_calls"][-1]["rq"] = test["test_calls"][-1]["rq"].replace(
                        "\"cluster_id\":\"%s\"" % cluster_id,
                        "\"cluster_id\":\"%s\"" % cluster_ids_dict[cluster_id])

                TestClusterService.shutdown_server(test["test_calls"])


if __name__ == '__main__':
    unittest.main()
