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

import httpretty
from freezegun import freeze_time

from app.commons.model import launch_objects
from app.service import ClusterService
from app.utils import utils
from test import get_fixture, APP_CONFIG
from test.mock_service import TestService


class TestClusterService(TestService):

    @freeze_time("2021-10-18 17:00:00")
    @utils.ignore_warnings
    def test_find_clusters(self):
        """Test finding clusters"""
        tests = [
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1",
                                "status": HTTPStatus.OK,
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(launchId=1, project=1),
                    project=1,
                    forUpdate=False,
                    numberOfLogLines=-1),
                "expected_result": launch_objects.ClusterResult(
                    project=1,
                    launchId=1,
                    clusters=[])
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/1",
                                "status": HTTPStatus.NOT_FOUND,
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(launchId=1, project=1),
                    project=1,
                    forUpdate=False,
                    numberOfLogLines=-1),
                "expected_result": launch_objects.ClusterResult(
                    project=1,
                    launchId=1,
                    clusters=[])
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/rp_2",
                                "status": HTTPStatus.OK,
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(launchId=1, project=2),
                    project=2,
                    forUpdate=False,
                    numberOfLogLines=-1),
                "app_config": APP_CONFIG,
                "expected_result": launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[])
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/2",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.search_logs_rq_first_group_not_for_update),
                                "rs": get_fixture(
                                    self.no_hits_search_rs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.search_logs_rq_second_group_not_for_update),
                                "rs": get_fixture(
                                    self.no_hits_search_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=false",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.cluster_update),
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=False,
                    numberOfLogLines=-1),
                "expected_result": launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId=21874152824769751,
                            clusterMessage="error occurred\nerror found\nerror mined",
                            logIds=[4, 5],
                            itemIds=[2, 5]),
                        launch_objects.ClusterInfo(
                            clusterId=44972330576749361,
                            clusterMessage="error occurred\nerror found\nassert query",
                            logIds=[9],
                            itemIds=[6])
                    ])
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/2",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_first_group_2lines_not_for_update),
                                "rs": get_fixture(self.no_hits_search_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=false",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.cluster_update_all_the_same),
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=False,
                    numberOfLogLines=2),
                "expected_result": launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId=48859729558090231,
                            clusterMessage="error occurred\nerror found",
                            logIds=[4, 5, 9],
                            itemIds=[2, 5, 6])
                    ])
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/2",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_first_group),
                                "rs": get_fixture(self.no_hits_search_rs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_second_group),
                                "rs": get_fixture(self.no_hits_search_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=false",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.cluster_update),
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(get_fixture(self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=True,
                    numberOfLogLines=-1),
                "expected_result": launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId=21874152824769751,
                            clusterMessage="error occurred\nerror found\nerror mined",
                            logIds=[4, 5],
                            itemIds=[2, 5]),
                        launch_objects.ClusterInfo(
                            clusterId=44972330576749361,
                            clusterMessage="error occurred\nerror found\nassert query",
                            logIds=[9],
                            itemIds=[6]),
                    ])
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/2",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_first_group),
                                "rs": get_fixture(self.one_hit_search_rs_clustering)
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_second_group),
                                "rs": get_fixture(self.one_hit_search_rs_clustering)
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=false",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.cluster_update_es_update),
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=True,
                    numberOfLogLines=-1),
                "expected_result": launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId=123,
                            clusterMessage="error occurred \n error found \n error mined",
                            logIds=[4, 5, 111],
                            itemIds=[2, 5]),
                        launch_objects.ClusterInfo(
                            clusterId=44972330576749361,
                            clusterMessage="error occurred\nerror found\nassert query",
                            logIds=[9],
                            itemIds=[6])
                    ])
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/2",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_first_group_2lines),
                                "rs": get_fixture(self.one_hit_search_rs_clustering),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=false",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.cluster_update_all_the_same_es_update),
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=True,
                    numberOfLogLines=2),
                "expected_result": launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId=48859729558090231,
                            clusterMessage="error occurred\nerror found",
                            logIds=[4, 5, 9],
                            itemIds=[2, 5, 6])
                    ])
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/rp_2",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/rp_2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_first_group_2lines),
                                "rs": get_fixture(self.one_hit_search_rs_clustering),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=false",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.cluster_update_all_the_same_es_update_with_prefix),
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(get_fixture(
                            self.launch_w_items_clustering, to_json=True))),
                    project=2,
                    forUpdate=True,
                    numberOfLogLines=2),
                "app_config": APP_CONFIG,
                "expected_result": launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId=48859729558090231,
                            clusterMessage="error occurred\nerror found",
                            logIds=[4, 5, 9],
                            itemIds=[2, 5, 6])
                    ])
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/2",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(
                                    self.search_logs_rq_first_group_assertion_error),
                                "rs": get_fixture(
                                    self.no_hits_search_rs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_first_group_assertion_error_status_code),
                                "rs": get_fixture(self.no_hits_search_rs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_first_group_no_such_element),
                                "rs": get_fixture(self.no_hits_search_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=false",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.cluster_update_all_the_same_es_with_different_errors),
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(get_fixture(
                            self.launch_w_items_clustering_with_different_errors, to_json=True))),
                    project=2,
                    forUpdate=False,
                    numberOfLogLines=2),
                "expected_result": launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId=37711525315085941,
                            clusterMessage="AssertionError error occurred\nerror found",
                            logIds=[4],
                            itemIds=[2]),
                        launch_objects.ClusterInfo(
                            clusterId=83179189436345941,
                            clusterMessage="AssertionError status code 500 error occurred\nerror found",
                            logIds=[5],
                            itemIds=[5]),
                        launch_objects.ClusterInfo(
                            clusterId=90988898127574211,
                            clusterMessage="NoSuchElementException error occurred\nerror found",
                            logIds=[9],
                            itemIds=[6]),
                    ])
            },
            {
                "test_calls": [{"method": httpretty.GET,
                                "uri": "/2",
                                "status": HTTPStatus.OK,
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_first_group_small_logs),
                                "rs": get_fixture(self.no_hits_search_rs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_second_group_small_logs),
                                "rs": get_fixture(self.no_hits_search_rs),
                                },
                               {"method": httpretty.GET,
                                "uri": "/2/_search",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.search_logs_rq_first_group_no_such_element_all_log_lines),
                                "rs": get_fixture(self.no_hits_search_rs),
                                },
                               {"method": httpretty.POST,
                                "uri": "/_bulk?refresh=false",
                                "status": HTTPStatus.OK,
                                "content_type": "application/json",
                                "rq": get_fixture(self.cluster_update_small_logs),
                                "rs": get_fixture(self.index_logs_rs),
                                }],
                "launch_info": launch_objects.LaunchInfoForClustering(
                    launch=launch_objects.Launch(
                        **(get_fixture(
                            self.launch_w_small_logs_for_clustering, to_json=True))),
                    project=2,
                    forUpdate=False,
                    numberOfLogLines=-1),
                "expected_result": launch_objects.ClusterResult(
                    project=2,
                    launchId=1,
                    clusters=[
                        launch_objects.ClusterInfo(
                            clusterId=60604459849884091,
                            clusterMessage="error occurred twice\nAssertionError error occurred\nerror found",
                            # noqa
                            logIds=[3, 4],
                            itemIds=[2]),
                        launch_objects.ClusterInfo(
                            clusterId=58202398056526781,
                            clusterMessage="AssertionError status code 500 error occurred",
                            logIds=[5],
                            itemIds=[5]),
                        launch_objects.ClusterInfo(
                            clusterId=86465058569810291,
                            clusterMessage="NoSuchElementException error occurred\nerror found\nassert query",
                            # noqa
                            logIds=[9],
                            itemIds=[6]),
                    ])
            }
        ]

        for idx, test in enumerate(tests):
            print(f'Test case number: {idx}')
            self._start_server(test["test_calls"])
            config = self.get_default_search_config()
            app_config = self.app_config
            if "app_config" in test:
                app_config = test["app_config"]
            _cluster_service = ClusterService(app_config=app_config, search_cfg=config)

            response = _cluster_service.find_clusters(test["launch_info"])

            assert len(response.clusters) == len(test["expected_result"].clusters)
            assert response == test["expected_result"]

            TestClusterService.shutdown_server(test["test_calls"])


if __name__ == '__main__':
    unittest.main()
