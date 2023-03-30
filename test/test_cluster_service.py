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
from http import HTTPStatus
import httpretty
import commons.launch_objects as launch_objects
from utils import utils
from service.cluster_service import ClusterService
from test.test_service import TestService
from freezegun import freeze_time


class TestClusterService(TestService):

    @freeze_time("2021-10-18 17:00:00")
    @utils.ignore_warnings
    def test_find_clusters(self):
        """Test finding clusters"""
        tests = [
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.OK,
                                         }],
                "launch_info":         launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(launchId=1, project=1),
                    project=1,
                    forUpdate=False,
                    numberOfLogLines=-1),
                "expected_result":     launch_objects.ClusterResult(
                    project=1,
                    launchId=1,
                    clusters=[])
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/1",
                                         "status":         HTTPStatus.NOT_FOUND,
                                         }],
                "launch_info":         launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(launchId=1, project=1),
                    project=1,
                    forUpdate=False,
                    numberOfLogLines=-1),
                "expected_result":     launch_objects.ClusterResult(
                    project=1,
                    launchId=1,
                    clusters=[])
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/rp_2",
                                         "status":         HTTPStatus.OK,
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(launchId=1, project=2),
                    project=2,
                    forUpdate=False,
                    numberOfLogLines=-1),
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
                    "esProjectIndexPrefix": "rp_",
                    "esChunkNumberUpdateClusters": 500
                },
                "expected_result":     launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[])
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
                                             self.search_logs_rq_first_group_not_for_update),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_second_group_not_for_update),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=false",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=False,
                    numberOfLogLines=-1),
                "expected_result":     launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId=51305554424475301,
                            clusterMessage="error occured \r\n error found \r\n error mined",
                            logIds=[4, 5],
                            itemIds=[2, 5]),
                        launch_objects.ClusterInfo(
                            clusterId=2474938495021661,
                            clusterMessage="error occured \r\n error found \r\n assert query",
                            logIds=[9],
                            itemIds=[6])
                    ])
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
                                             self.search_logs_rq_first_group_2lines_not_for_update),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=false",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update_all_the_same),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=False,
                    numberOfLogLines=2),
                "expected_result":     launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId="53490850438321651",
                            clusterMessage="error occured \r\n error found",
                            logIds=[4, 5, 9],
                            itemIds=[2, 5, 6])
                    ])
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
                                         "uri":            "/_bulk?refresh=false",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=True,
                    numberOfLogLines=-1),
                "expected_result":     launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId="51305554424475301",
                            clusterMessage="error occured \r\n error found \r\n error mined",
                            logIds=[4, 5],
                            itemIds=[2, 5]),
                        launch_objects.ClusterInfo(
                            clusterId="2474938495021661",
                            clusterMessage="error occured \r\n error found \r\n assert query",
                            logIds=[9],
                            itemIds=[6]),
                    ])
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
                                             self.one_hit_search_rs_clustering)
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_second_group),
                                         "rs":             utils.get_fixture(
                                             self.one_hit_search_rs_clustering)
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=false",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update_es_update),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=True,
                    numberOfLogLines=-1),
                "expected_result":     launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId="123",
                            clusterMessage="error occured \n error found \n error mined",
                            logIds=[4, 5, 111],
                            itemIds=[2, 5]),
                        launch_objects.ClusterInfo(
                            clusterId="2474938495021661",
                            clusterMessage="error occured \r\n error found \r\n assert query",
                            logIds=[9],
                            itemIds=[6])
                    ])
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
                                         "uri":            "/_bulk?refresh=false",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update_all_the_same_es_update),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=True,
                    numberOfLogLines=2),
                "expected_result":     launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId="53490850438321651",
                            clusterMessage="error occured \r\n error found",
                            logIds=[4, 5, 9],
                            itemIds=[2, 5, 6])
                    ])
            },
            {
                "test_calls":          [{"method":         httpretty.GET,
                                         "uri":            "/rp_2",
                                         "status":         HTTPStatus.OK,
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/rp_2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_first_group_2lines),
                                         "rs":             utils.get_fixture(
                                             self.one_hit_search_rs_clustering),
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=false",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update_all_the_same_es_update_with_prefix),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=True,
                    numberOfLogLines=2),
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
                    "esProjectIndexPrefix": "rp_",
                    "esChunkNumberUpdateClusters": 500
                },
                "expected_result":     launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId="53490850438321651",
                            clusterMessage="error occured \r\n error found",
                            logIds=[4, 5, 9],
                            itemIds=[2, 5, 6])
                    ])
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
                                             self.search_logs_rq_first_group_assertion_error),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_first_group_assertion_error_status_code),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_first_group_no_such_element),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=false",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update_all_the_same_es_with_different_errors),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_w_items_clustering_with_different_errors, to_json=True))),
                    project=2,
                    forUpdate=False,
                    numberOfLogLines=2),
                "expected_result":     launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId="66538501077545981",
                            clusterMessage="AssertionError error occured \r\n error found",
                            logIds=[4],
                            itemIds=[2]),
                        launch_objects.ClusterInfo(
                            clusterId="30071099716448071",
                            clusterMessage="AssertionError status code: 500 error occured \r\n error found",
                            logIds=[5],
                            itemIds=[5]),
                        launch_objects.ClusterInfo(
                            clusterId="59521687023339221",
                            clusterMessage="NoSuchElementException error occured \r\n error found",
                            logIds=[9],
                            itemIds=[6]),
                    ])
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
                                             self.search_logs_rq_first_group_small_logs),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_second_group_small_logs),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.GET,
                                         "uri":            "/2/_search",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.search_logs_rq_first_group_no_such_element_all_log_lines),
                                         "rs":             utils.get_fixture(
                                             self.no_hits_search_rs),
                                         },
                                        {"method":         httpretty.POST,
                                         "uri":            "/_bulk?refresh=false",
                                         "status":         HTTPStatus.OK,
                                         "content_type":   "application/json",
                                         "rq":             utils.get_fixture(
                                             self.cluster_update_small_logs),
                                         "rs":             utils.get_fixture(
                                             self.index_logs_rs),
                                         }],
                "launch_info":            launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(utils.get_fixture(
                            self.launch_w_small_logs_for_clustering, to_json=True))),
                    project=2,
                    forUpdate=False,
                    numberOfLogLines=-1),
                "expected_result":     launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId="78342974021039661",
                            clusterMessage="error occured twice \r\nAssertionError error occured \r\n error found",  # noqa
                            logIds=[3, 4],
                            itemIds=[2]),
                        launch_objects.ClusterInfo(
                            clusterId="37054331802624341",
                            clusterMessage="AssertionError status code: 500 error occured",
                            logIds=[5],
                            itemIds=[5]),
                        launch_objects.ClusterInfo(
                            clusterId="16492834929015971",
                            clusterMessage="NoSuchElementException error occured \r\n error found \r\n assert query",  # noqa
                            logIds=[9],
                            itemIds=[6]),
                    ])
            }
        ]

        for idx, test in enumerate(tests):
            try:
                self._start_server(test["test_calls"])
                config = self.get_default_search_config()
                app_config = self.app_config
                if "app_config" in test:
                    app_config = test["app_config"]
                _cluster_service = ClusterService(app_config=app_config,
                                                  search_cfg=config)

                response = _cluster_service.find_clusters(test["launch_info"])

                assert len(response.clusters) == len(test["expected_result"].clusters)
                assert test["expected_result"] == response

                TestClusterService.shutdown_server(test["test_calls"])
            except AssertionError as err:
                raise AssertionError(f'Error in the test case number: {idx}').\
                    with_traceback(err.__traceback__)


if __name__ == '__main__':
    unittest.main()
