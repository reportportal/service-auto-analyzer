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

from app.commons import object_saving
from app.commons.model import launch_objects
from app.machine_learning.models.boosting_decision_maker import BoostingDecisionMaker
from app.service import SuggestService
from app.utils import utils
from test import APP_CONFIG, get_fixture
from test.mock_service import TestService
from test.service import (
    get_bulk_call,
    get_index_found_call,
    get_index_not_found_call,
    get_search_for_logs_call_no_parameters,
    get_search_for_logs_call_with_parameters,
)

GET_METRICS_INFO_CALL = {"method": httpretty.GET, "uri": "/rp_suggestions_info_metrics", "status": HTTPStatus.OK}
CREATE_METRICS_INFO_MAPPING_CALL = {
    "method": httpretty.PUT,
    "uri": "/rp_suggestions_info_metrics/_mapping",
    "status": HTTPStatus.OK,
    "rs": get_fixture("index_created_rs.json"),
}


class TestSuggestService(TestService):

    @utils.ignore_warnings
    def test_suggest_items(self):
        """Test suggesting test items"""
        tests = [
            {
                "test_calls": [
                    get_index_found_call("1"),
                    GET_METRICS_INFO_CALL,
                    CREATE_METRICS_INFO_MAPPING_CALL,
                    get_bulk_call(None, get_fixture(self.index_logs_rs)),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    testItemId=1, uniqueId="341", testCaseHash=123, launchId=1, launchName="Launch", project=1, logs=[]
                ),
                "expected_result": [],
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    get_index_not_found_call("2"),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    testItemId=1,
                    uniqueId="341",
                    testCaseHash=123,
                    launchId=1,
                    launchName="Launch",
                    project=2,
                    logs=[launch_objects.Log(logId=1, message="error found", logLevel=40000)],
                ),
                "expected_result": [],
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                    GET_METRICS_INFO_CALL,
                    CREATE_METRICS_INFO_MAPPING_CALL,
                    get_bulk_call(None, get_fixture(self.index_logs_rs)),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    testItemId=1,
                    uniqueId="341",
                    testCaseHash=123,
                    launchId=1,
                    launchName="Launch",
                    project=1,
                    logs=[launch_objects.Log(logId=1, message=" ", logLevel=40000)],
                ),
                "expected_result": [],
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                    GET_METRICS_INFO_CALL,
                    CREATE_METRICS_INFO_MAPPING_CALL,
                    get_bulk_call(None, get_fixture(self.index_logs_rs)),
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_logs, to_json=True)
                ),
                "expected_result": [],
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                    GET_METRICS_INFO_CALL,
                    CREATE_METRICS_INFO_MAPPING_CALL,
                    get_bulk_call(None, get_fixture(self.index_logs_rs)),
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_logs, to_json=True)
                ),
                "expected_result": [],
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                ],
                "msearch_results": [
                    get_fixture(self.no_hits_search_rs, to_json=True),
                    get_fixture(self.one_hit_search_rs, to_json=True),
                    get_fixture(self.one_hit_search_rs, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_logs, to_json=True)
                ),
                "expected_result": [
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        relevantItem=1,
                        relevantLogId=1,
                        matchScore=80.0,
                        esScore=10.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=0,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    )
                ],
                "boost_predict": ([1], [[0.2, 0.8]]),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                ],
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs, to_json=True),
                    get_fixture(self.one_hit_search_rs, to_json=True),
                    get_fixture(self.one_hit_search_rs, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_logs, to_json=True)
                ),
                "expected_result": [
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        relevantItem=1,
                        relevantLogId=1,
                        matchScore=70.0,
                        esScore=10.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=0,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    )
                ],
                "boost_predict": ([1], [[0.3, 0.7]]),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                ],
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_logs, to_json=True)
                ),
                "expected_result": [
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        relevantItem=1,
                        relevantLogId=1,
                        matchScore=70.0,
                        esScore=15.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=0,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    )
                ],
                "boost_predict": ([1, 0], [[0.3, 0.7], [0.9, 0.1]]),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                ],
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_logs, to_json=True)
                ),
                "expected_result": [
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        relevantItem=1,
                        relevantLogId=1,
                        matchScore=70.0,
                        esScore=15.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=0,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    ),
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="PB001",
                        relevantItem=2,
                        relevantLogId=2,
                        matchScore=45.0,
                        esScore=10.0,
                        esPosition=1,
                        modelFeatureNames="0",
                        modelFeatureValues="0.67",
                        modelInfo="",
                        resultPosition=1,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    ),
                ],
                "boost_predict": ([1, 0], [[0.3, 0.7], [0.55, 0.45]]),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                ],
                "msearch_results": [
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_logs, to_json=True)
                ),
                "expected_result": [
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="PB001",
                        relevantItem=3,
                        relevantLogId=3,
                        matchScore=80.0,
                        esScore=10.0,
                        esPosition=2,
                        modelFeatureNames="0",
                        modelFeatureValues="0.67",
                        modelInfo="",
                        resultPosition=0,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    ),
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        relevantItem=1,
                        relevantLogId=1,
                        matchScore=70.0,
                        esScore=15.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=1,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    ),
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="PB001",
                        relevantItem=2,
                        relevantLogId=2,
                        matchScore=45.0,
                        esScore=10.0,
                        esPosition=1,
                        modelFeatureNames="0",
                        modelFeatureValues="0.67",
                        modelInfo="",
                        resultPosition=2,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    ),
                ],
                "boost_predict": ([1, 0, 1], [[0.3, 0.7], [0.55, 0.45], [0.2, 0.8]]),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                ],
                "msearch_results": [
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.three_hits_search_rs_with_duplicate, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_logs, to_json=True)
                ),
                "expected_result": [
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        relevantItem=3,
                        relevantLogId=3,
                        matchScore=70.0,
                        esScore=15.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=0,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    ),
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        relevantItem=1,
                        relevantLogId=1,
                        matchScore=70.0,
                        esScore=15.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=1,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    ),
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="PB001",
                        relevantItem=2,
                        relevantLogId=2,
                        matchScore=70.0,
                        esScore=10.0,
                        esPosition=1,
                        modelFeatureNames="0",
                        modelFeatureValues="0.67",
                        modelInfo="",
                        resultPosition=2,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    ),
                ],
                "boost_predict": ([1, 1, 1], [[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                ],
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs_merged, to_json=True),
                    get_fixture(self.one_hit_search_rs_merged, to_json=True),
                    get_fixture(self.one_hit_search_rs_merged, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)
                ),
                "expected_result": [
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        relevantItem=1,
                        relevantLogId=1,
                        isMergedLog=True,
                        matchScore=90.0,
                        esScore=10.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=0,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    )
                ],
                "boost_predict": ([1], [[0.1, 0.9]]),
            },
            {
                "test_calls": [
                    get_index_found_call("rp_1"),
                ],
                "app_config": APP_CONFIG,
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs_merged, to_json=True),
                    get_fixture(self.one_hit_search_rs_merged, to_json=True),
                    get_fixture(self.one_hit_search_rs_merged, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)
                ),
                "expected_result": [
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=123,
                        testItemLogId=178,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        relevantItem=1,
                        relevantLogId=1,
                        isMergedLog=True,
                        matchScore=90.0,
                        esScore=10.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=0,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                    )
                ],
                "boost_predict": ([1], [[0.1, 0.9]]),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                    GET_METRICS_INFO_CALL,
                    CREATE_METRICS_INFO_MAPPING_CALL,
                    get_bulk_call(None, get_fixture(self.index_logs_rs)),
                ],
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs_merged_wrong, to_json=True),
                    get_fixture(self.one_hit_search_rs_merged_wrong, to_json=True),
                    get_fixture(self.one_hit_search_rs_merged_wrong, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)
                ),
                "expected_result": [],
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    get_index_found_call("rp_1"),
                    GET_METRICS_INFO_CALL,
                    CREATE_METRICS_INFO_MAPPING_CALL,
                    get_bulk_call(None, get_fixture(self.index_logs_rs)),
                ],
                "app_config": APP_CONFIG,
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs_merged_wrong, to_json=True),
                    get_fixture(self.one_hit_search_rs_merged_wrong, to_json=True),
                    get_fixture(self.one_hit_search_rs_merged_wrong, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_w_merged_logs, to_json=True)
                ),
                "expected_result": [],
                "boost_predict": ([], []),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                    get_search_for_logs_call_no_parameters(
                        "1", get_fixture(self.search_test_item_cluster), get_fixture(self.three_hits_search_rs)
                    ),
                    get_search_for_logs_call_with_parameters(
                        "1",
                        get_fixture(self.search_logs_by_test_item),
                        get_fixture(self.three_hits_search_rs),
                    ),
                ],
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs, to_json=True),
                    get_fixture(self.two_hits_search_rs, to_json=True),
                    get_fixture(self.no_hits_search_rs, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_cluster, to_json=True)
                ),
                "expected_result": [
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=1,
                        testItemLogId=1,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        isMergedLog=False,
                        relevantItem=1,
                        relevantLogId=1,
                        matchScore=70.0,
                        esScore=15.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=0,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                        clusterId=5349085043832165,
                    ),
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=1,
                        testItemLogId=1,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="PB001",
                        isMergedLog=False,
                        relevantItem=2,
                        relevantLogId=2,
                        matchScore=45.0,
                        esScore=10.0,
                        esPosition=1,
                        modelFeatureNames="0",
                        modelFeatureValues="0.67",
                        modelInfo="",
                        resultPosition=1,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                        clusterId=5349085043832165,
                    ),
                ],
                "boost_predict": ([1, 0], [[0.3, 0.7], [0.55, 0.45]]),
            },
            {
                "test_calls": [
                    get_index_found_call("1"),
                    get_search_for_logs_call_no_parameters(
                        "1", get_fixture(self.search_test_item_cluster), get_fixture(self.one_hit_search_rs_small_logs)
                    ),
                    get_search_for_logs_call_with_parameters(
                        "1",
                        get_fixture(self.search_logs_by_test_item),
                        get_fixture(self.one_hit_search_rs_small_logs),
                    ),
                ],
                "msearch_results": [
                    get_fixture(self.one_hit_search_rs_merged, to_json=True),
                    get_fixture(self.one_hit_search_rs_merged, to_json=True),
                    get_fixture(self.one_hit_search_rs_merged, to_json=True),
                ],
                "test_item_info": launch_objects.TestItemInfo(
                    **get_fixture(self.suggest_test_item_info_cluster, to_json=True)
                ),
                "expected_result": [
                    launch_objects.SuggestAnalysisResult(
                        project=1,
                        testItem=1,
                        testItemLogId=1,
                        launchId=145,
                        launchName="Launch with test items with logs",
                        launchNumber=145,
                        issueType="AB001",
                        isMergedLog=True,
                        relevantItem=1,
                        relevantLogId=1,
                        matchScore=75.0,
                        esScore=10.0,
                        esPosition=0,
                        modelFeatureNames="0",
                        modelFeatureValues="1.0",
                        modelInfo="",
                        resultPosition=0,
                        usedLogLines=-1,
                        minShouldMatch=80,
                        processedTime=10.0,
                        methodName="suggestion",
                        clusterId=5349085043832165,
                    )
                ],
                "boost_predict": ([1], [[0.25, 0.75]]),
            },
        ]

        for idx, test in enumerate(tests):
            print(f"Running test case idx: {idx}")
            self._start_server(test["test_calls"])
            config = self.get_default_search_config()
            app_config = self.app_config
            if "app_config" in test:
                app_config = test["app_config"]
            suggest_service = SuggestService(self.model_chooser, app_config=app_config, search_cfg=config)
            suggest_service.es_client.es_client.scroll = MagicMock(
                return_value=json.loads(get_fixture(self.no_hits_search_rs))
            )
            if "msearch_results" in test:
                suggest_service.es_client.es_client.msearch = MagicMock(
                    return_value={"responses": test["msearch_results"]}
                )
            _boosting_decision_maker = BoostingDecisionMaker(object_saving.create_filesystem(""), "", features=[0])
            _boosting_decision_maker.predict = MagicMock(return_value=test["boost_predict"])
            suggest_service.model_chooser.choose_model = MagicMock(return_value=_boosting_decision_maker)
            response = suggest_service.suggest_items(test["test_item_info"])

            assert len(response) == len(test["expected_result"])
            for real_resp, expected_resp in zip(response, test["expected_result"]):
                real_resp.processedTime = 10.0
                assert real_resp == expected_resp

            TestSuggestService.shutdown_server(test["test_calls"])


if __name__ == "__main__":
    unittest.main()
