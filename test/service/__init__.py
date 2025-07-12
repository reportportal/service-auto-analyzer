#   Copyright 2023 EPAM Systems
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from http import HTTPStatus
from typing import Any

import httpretty

from app.commons.model import launch_objects
from test import get_fixture


def get_index_call(index_name: str, status: HTTPStatus) -> dict:
    """
    Returns a dictionary representing an HTTP call that simulates getting index.
    """
    return {
        "method": httpretty.GET,
        "uri": f"/{index_name}",
        "status": status,
    }


def get_index_found_call(index_name: str) -> dict:
    """
    Returns a dictionary representing an HTTP call that simulates a successful index retrieval.
    """
    # Mute invalid Sonar's "Change this argument; Function "get_index_call" expects a different type"
    return get_index_call(index_name, HTTPStatus.OK)  # NOSONAR


def get_index_not_found_call(index_name: str) -> dict:
    """
    Returns a dictionary representing an HTTP call that simulates an index not found error.
    """
    # Mute invalid Sonar's "Change this argument; Function "get_index_call" expects a different type"
    return get_index_call(index_name, HTTPStatus.NOT_FOUND)  # NOSONAR


def get_search_for_logs_call(index_name: str, query_parameters: str, rq: Any, rs: Any) -> dict:
    uri = f"/{index_name}/_search"
    if query_parameters:
        uri += f"?{query_parameters}"
    return {
        "method": httpretty.GET,
        "uri": uri,
        "status": HTTPStatus.OK,
        "content_type": "application/json",
        "rq": rq,
        "rs": rs,
    }


def get_search_for_logs_call_no_parameters(index_name: str, rq: Any, rs: Any) -> dict:
    return get_search_for_logs_call(index_name, "", rq, rs)


def get_search_for_logs_call_with_parameters(index_name: str, rq: Any, rs: Any) -> dict:
    return get_search_for_logs_call(index_name, "scroll=5m&size=1000", rq, rs)


def get_bulk_call(rq: Any, rs: Any) -> dict:
    call = {
        "method": httpretty.POST,
        "uri": "/_bulk?refresh=true",
        "status": HTTPStatus.OK,
        "content_type": "application/json",
        "rs": rs,
    }
    if rq is not None:
        call["rq"] = rq
    return call


# Helper functions for SuggestService tests
def get_metrics_info_call() -> dict:
    """Returns a dictionary representing the GET metrics info call."""
    return {
        "method": httpretty.GET,
        "uri": "/rp_suggestions_info_metrics",
        "status": HTTPStatus.OK,
    }


def get_create_metrics_info_mapping_call() -> dict:
    """Returns a dictionary representing the create metrics info mapping call."""
    return {
        "method": httpretty.PUT,
        "uri": "/rp_suggestions_info_metrics/_mapping",
        "status": HTTPStatus.OK,
        "rs": get_fixture("index_created_rs.json"),
    }


def get_basic_suggest_test_calls(index_name: str, index_logs_rs_fixture: str) -> list[dict]:
    """Returns basic test calls for suggest service tests."""
    return [
        get_index_found_call(index_name),
        get_metrics_info_call(),
        get_create_metrics_info_mapping_call(),
        get_bulk_call(None, get_fixture(index_logs_rs_fixture)),
    ]


def get_basic_test_item_info() -> launch_objects.TestItemInfo:
    """Returns basic test item info for suggest service tests."""
    return launch_objects.TestItemInfo(
        testItemId=1, uniqueId="341", testCaseHash=123, launchId=1, launchName="Launch", project=1, logs=[]
    )


def get_test_item_info_with_logs() -> launch_objects.TestItemInfo:
    """Returns test item info with logs for suggest service tests."""
    return launch_objects.TestItemInfo(
        testItemId=1,
        uniqueId="341",
        testCaseHash=123,
        launchId=1,
        launchName="Launch",
        project=2,
        logs=[launch_objects.Log(logId=1, message="error found", logLevel=40000)],
    )


def get_test_item_info_with_empty_logs() -> launch_objects.TestItemInfo:
    """Returns test item info with empty logs for suggest service tests."""
    return launch_objects.TestItemInfo(
        testItemId=1,
        uniqueId="341",
        testCaseHash=123,
        launchId=1,
        launchName="Launch",
        project=1,
        logs=[launch_objects.Log(logId=1, message=" ", logLevel=40000)],
    )


def get_suggest_analysis_result(
    project: int = 1,
    test_item: int = 123,
    test_item_log_id: int = 178,
    launch_id: int = 145,
    launch_name: str = "Launch with test items with logs",
    launch_number: int = 145,
    issue_type: str = "AB001",
    relevant_item: int = 1,
    relevant_log_id: int = 1,
    match_score: float = 70.0,
    es_score: float = 10.0,
    es_position: int = 0,
    model_feature_names: str = "0",
    model_feature_values: str = "1.0",
    model_info: str = "",
    result_position: int = 0,
    used_log_lines: int = -1,
    min_should_match: int = 80,
    processed_time: float = 10.0,
    method_name: str = "suggestion",
    is_merged_log: bool = False,
    cluster_id: int | None = None,
) -> launch_objects.SuggestAnalysisResult:
    """Returns a SuggestAnalysisResult with default values that can be overridden."""
    result = launch_objects.SuggestAnalysisResult(
        project=project,
        testItem=test_item,
        testItemLogId=test_item_log_id,
        launchId=launch_id,
        launchName=launch_name,
        launchNumber=launch_number,
        issueType=issue_type,
        relevantItem=relevant_item,
        relevantLogId=relevant_log_id,
        matchScore=match_score,
        esScore=es_score,
        esPosition=es_position,
        modelFeatureNames=model_feature_names,
        modelFeatureValues=model_feature_values,
        modelInfo=model_info,
        resultPosition=result_position,
        usedLogLines=used_log_lines,
        minShouldMatch=min_should_match,
        processedTime=processed_time,
        methodName=method_name,
        isMergedLog=is_merged_log,
    )
    if cluster_id is not None:
        result.clusterId = cluster_id
    return result
