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

import json
from http import HTTPStatus
from typing import Any

import httpretty

from app.commons.model import launch_objects
from test import get_fixture

SUGGESTIONS_INFO_METRICS = "/rp_suggestions_info_metrics"
APPLICATION_JSON = "application/json"
ERROR_COMMON_ERROR = "assertionError commonError"
HELLO_WORLD = "hello world"
HELLO_WORLD_SDF = f"{HELLO_WORLD} 'sdf'"


def get_index_call(index_name: str, status: int) -> dict:
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
    # Mute invalid Sonar's 'Change this argument; Function "get_index_call" expects a different type'
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
        "method": httpretty.POST,
        "uri": uri,
        "status": HTTPStatus.OK,
        "content_type": APPLICATION_JSON,
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
        "content_type": APPLICATION_JSON,
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
        "uri": SUGGESTIONS_INFO_METRICS,
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
    test_item: int = 123,
    test_item_log_id: int = 178,
    issue_type: str = "AB001",
    relevant_item: int = 1,
    relevant_log_id: int = 1,
    match_score: float = 70.0,
    es_score: float = 10.0,
    es_position: int = 0,
    model_feature_values: str = "1.0",
    result_position: int = 0,
    is_merged_log: bool = False,
    cluster_id: int | None = None,
) -> launch_objects.SuggestAnalysisResult:
    """Returns a SuggestAnalysisResult with default values that can be overridden."""
    result = launch_objects.SuggestAnalysisResult(
        project=1,
        testItem=test_item,
        testItemLogId=test_item_log_id,
        launchId=145,
        launchName="Launch with test items with logs",
        launchNumber=145,
        issueType=issue_type,
        relevantItem=relevant_item,
        relevantLogId=relevant_log_id,
        matchScore=match_score,
        esScore=es_score,
        esPosition=es_position,
        modelFeatureNames="0",
        modelFeatureValues=model_feature_values,
        modelInfo="",
        resultPosition=result_position,
        usedLogLines=-1,
        minShouldMatch=80,
        processedTime=10.0,
        methodName="suggestion",
        isMergedLog=is_merged_log,
    )
    if cluster_id is not None:
        result.clusterId = cluster_id
    return result


# Helper functions for SuggestPatternsService tests
def get_suggest_pattern_label(
    pattern: str,
    total_count: int,
    percent_test_items_with_label: float,
    label: str,
) -> launch_objects.SuggestPatternLabel:
    """Returns a SuggestPatternLabel with the given parameters."""
    return launch_objects.SuggestPatternLabel(
        pattern=pattern,
        totalCount=total_count,
        percentTestItemsWithLabel=percent_test_items_with_label,
        label=label,
    )


def get_common_query_data() -> list[tuple[str, str]]:
    """Returns common query data for suggest patterns tests."""
    return [
        ("assertionError notFoundError", "ab001"),
        ("assertionError ifElseError", "pb001"),
        (ERROR_COMMON_ERROR, "ab001"),
        (ERROR_COMMON_ERROR, "ab001"),
        ("assertionError", "ab001"),
        (ERROR_COMMON_ERROR, "ab001"),
        (ERROR_COMMON_ERROR, "ti001"),
    ]


def get_common_expected_patterns_with_labels() -> list[launch_objects.SuggestPatternLabel]:
    """Returns common expected patterns with labels for suggest patterns tests."""
    return [
        get_suggest_pattern_label("assertionError", 24, 0.83, "ab001"),
        get_suggest_pattern_label("commonError", 12, 1.0, "ab001"),
    ]


def get_common_expected_patterns_without_labels() -> list[launch_objects.SuggestPatternLabel]:
    """Returns common expected patterns without labels for suggest patterns tests."""
    return [
        get_suggest_pattern_label("assertionError", 28, 0.0, ""),
        get_suggest_pattern_label("commonError", 16, 0.0, ""),
    ]


# Helper functions for EsQuery tests
def get_launch_object(
    analyzer_mode: str = "ALL",
    number_of_log_lines: int = -1,
    launch_id: int = 12,
    launch_name: str = "Launch name",
    project: int = 1,
) -> launch_objects.Launch:
    """Returns a Launch object with customizable parameters."""
    return launch_objects.Launch(
        analyzerConfig=launch_objects.AnalyzerConf(
            analyzerMode=analyzer_mode,
            numberOfLogLines=number_of_log_lines,
        ),
        launchId=launch_id,
        launchName=launch_name,
        project=project,
    )


def get_test_item_info_for_suggest(
    analyzer_mode: str = "ALL",
    number_of_log_lines: int = -1,
    launch_id: int = 12,
    launch_name: str = "Launch name",
    project: int = 1,
    test_case_hash: int = 1,
    unique_id: str = "unique",
    test_item_id: int = 2,
) -> launch_objects.TestItemInfo:
    """Returns a TestItemInfo object for suggest queries with customizable parameters."""
    return launch_objects.TestItemInfo(
        analyzerConfig=launch_objects.AnalyzerConf(
            analyzerMode=analyzer_mode,
            numberOfLogLines=number_of_log_lines,
        ),
        launchId=launch_id,
        launchName=launch_name,
        project=project,
        testCaseHash=test_case_hash,
        uniqueId=unique_id,
        testItemId=test_item_id,
    )


def get_search_logs_object(
    launch_id: int = 1,
    launch_name: str = "launch 1",
    item_id: int = 2,
    project_id: int = 3,
    filtered_launch_ids: list[int] | None = None,
    log_messages: list[str] | None = None,
    log_lines: int = -1,
    analyzer_config: launch_objects.AnalyzerConf | None = None,
) -> launch_objects.SearchLogs:
    """Returns a SearchLogs object with customizable parameters."""
    if filtered_launch_ids is None:
        filtered_launch_ids = [1, 2, 3]
    if log_messages is None:
        log_messages = ["log message 1"]

    search_logs = launch_objects.SearchLogs(
        launchId=launch_id,
        launchName=launch_name,
        itemId=item_id,
        projectId=project_id,
        filteredLaunchIds=filtered_launch_ids,
        logMessages=log_messages,
        logLines=log_lines,
    )

    if analyzer_config is not None:
        search_logs.analyzerConfig = analyzer_config

    return search_logs


def get_base_log_dict(
    message: str = HELLO_WORLD,
    merged_small_logs: str = "",
    detected_message: str = HELLO_WORLD,
    detected_message_with_numbers: str = f"{HELLO_WORLD} 1",
    stacktrace: str = "",
    only_numbers: str = "1",
    found_exceptions: str = "AssertionError",
    potential_status_codes: str = "",
    found_tests_and_methods: str = "",
) -> dict:
    """Returns a base log dictionary with customizable parameters."""
    return {
        "_id": 1,
        "_index": 1,
        "_source": {
            "start_time": "2021-08-30 08:11:23",
            "unique_id": "unique",
            "test_case_hash": 1,
            "test_item": "123",
            "test_item_name": "test item Common Query",
            "message": message,
            "merged_small_logs": merged_small_logs,
            "detected_message": detected_message,
            "detected_message_with_numbers": detected_message_with_numbers,
            "detected_message_without_params_extended": HELLO_WORLD,
            "stacktrace": stacktrace,
            "only_numbers": only_numbers,
            "found_exceptions": found_exceptions,
            "potential_status_codes": potential_status_codes,
            "found_tests_and_methods": found_tests_and_methods,
        },
    }


def get_extended_log_dict(
    message: str = HELLO_WORLD_SDF,
    merged_small_logs: str = "",
    detected_message: str = HELLO_WORLD_SDF,
    detected_message_with_numbers: str = f"{HELLO_WORLD} 1 'sdf'",
    stacktrace: str = "",
    found_exceptions: str = "AssertionError",
    message_params: str = "sdf",
    message_without_params_extended: str = HELLO_WORLD,
    stacktrace_extended: str = "",
    message_extended: str = HELLO_WORLD_SDF,
    detected_message_extended: str = HELLO_WORLD_SDF,
    potential_status_codes: str = "",
    found_tests_and_methods: str = "",
) -> dict:
    """Returns an extended log dictionary for search/suggest queries with customizable parameters."""
    base_log = get_base_log_dict(
        message=message,
        merged_small_logs=merged_small_logs,
        detected_message=detected_message,
        detected_message_with_numbers=detected_message_with_numbers,
        stacktrace=stacktrace,
        found_exceptions=found_exceptions,
        potential_status_codes=potential_status_codes,
        found_tests_and_methods=found_tests_and_methods,
    )

    # Add extended fields
    base_log["_source"].update(
        {
            "found_exceptions_extended": "AssertionError",
            "message_params": message_params,
            "urls": "",
            "paths": "",
            "message_without_params_extended": message_without_params_extended,
            "stacktrace_extended": stacktrace_extended,
            "message_extended": message_extended,
            "detected_message_extended": detected_message_extended,
        }
    )

    return base_log


# Helper functions for SuggestInfoService tests
def get_suggest_index_call(index_name: str, status: int) -> dict:
    """Returns a dictionary representing an HTTP call to a suggest index."""
    return {
        "method": httpretty.GET,
        "uri": f"/{index_name}_suggest",
        "status": status,
        "content_type": APPLICATION_JSON,
    }


def get_suggest_index_not_found_call(index_name: str) -> dict:
    """Returns a dictionary representing a suggest index not found call."""
    return get_suggest_index_call(index_name, HTTPStatus.NOT_FOUND)


def get_suggest_index_found_call(index_name: str) -> dict:
    """Returns a dictionary representing a suggest index found call."""
    return get_suggest_index_call(index_name, HTTPStatus.OK)


def get_delete_suggest_index_call(index_name: str, status: int, fixture_rs: str) -> dict:
    """Returns a dictionary representing a DELETE call to a suggest index."""
    return {
        "method": httpretty.DELETE,
        "uri": f"/{index_name}_suggest",
        "status": status,
        "content_type": APPLICATION_JSON,
        "rs": get_fixture(fixture_rs),
    }


def get_delete_by_query_call(index_name: str, fixture_rq: str, response_data: dict) -> dict:
    """Returns a dictionary representing a DELETE by query call."""
    return {
        "method": httpretty.POST,
        "uri": f"/{index_name}_suggest/_delete_by_query",
        "status": HTTPStatus.OK,
        "content_type": APPLICATION_JSON,
        "rq": get_fixture(fixture_rq),
        "rs": json.dumps(response_data),
    }


def get_put_index_call(index_name: str, fixture_rs: str) -> dict:
    """Returns a dictionary representing a PUT call to create an index."""
    return {
        "method": httpretty.PUT,
        "uri": f"/{index_name}",
        "status": HTTPStatus.OK,
        "rs": get_fixture(fixture_rs),
    }


def get_put_mapping_call(index_name: str, fixture_rs: str) -> dict:
    """Returns a dictionary representing a PUT call to create a mapping."""
    return {
        "method": httpretty.PUT,
        "uri": f"/{index_name}/_mapping",
        "status": HTTPStatus.OK,
        "rs": get_fixture(fixture_rs),
    }


def get_clean_index_object(ids: list[int], project: int) -> launch_objects.CleanIndex:
    """Returns a CleanIndex object with the given parameters."""
    return launch_objects.CleanIndex(ids=ids, project=project)


def get_item_remove_info(project: int, items_to_delete: list[int]) -> dict:
    """Returns an item removal info dictionary."""
    return {"project": project, "itemsToDelete": items_to_delete}


def get_launch_remove_info(project: int, launch_ids: list[int]) -> dict:
    """Returns a launch removal info dictionary."""
    return {"project": project, "launch_ids": launch_ids}


def get_defect_update_info(project: int, items_to_update: dict) -> dict:
    """Returns a defect update info dictionary."""
    return {"project": project, "itemsToUpdate": items_to_update}


def get_suggest_info_cleanup_calls(
    index_name: str, search_fixture: str, search_rs_fixture: str, delete_fixture: str, delete_rs_fixture: str
) -> list[dict]:
    """Returns test calls for suggest info cleanup operations."""
    return [
        get_suggest_index_found_call(index_name),
        get_search_for_logs_call_with_parameters(
            f"{index_name}_suggest",
            get_fixture(search_fixture),
            get_fixture(search_rs_fixture),
        ),
        get_bulk_call(get_fixture(delete_fixture), get_fixture(delete_rs_fixture)),
    ]


def get_suggest_index_creation_calls(
    metrics_status: int,
    suggest_index_status: int,
    metrics_fixture: str,
    suggest_fixture: str,
    index_name: str,
    bulk_fixture: str,
    bulk_count: int = 2,
) -> list[dict]:
    """Returns test calls for suggest index creation operations."""
    calls = []

    # Metrics index handling
    if metrics_status == HTTPStatus.NOT_FOUND:
        calls.extend(
            [
                {"method": httpretty.GET, "uri": SUGGESTIONS_INFO_METRICS, "status": HTTPStatus.NOT_FOUND},
                get_put_index_call("rp_suggestions_info_metrics", metrics_fixture),
            ]
        )
    else:
        calls.extend(
            [
                {"method": httpretty.GET, "uri": SUGGESTIONS_INFO_METRICS, "status": HTTPStatus.OK},
                get_put_mapping_call("rp_suggestions_info_metrics", metrics_fixture),
            ]
        )

    # Suggest index handling
    if suggest_index_status == HTTPStatus.NOT_FOUND:
        calls.extend(
            [
                {"method": httpretty.GET, "uri": f"/{index_name}_suggest", "status": HTTPStatus.NOT_FOUND},
                get_put_index_call(f"{index_name}_suggest", suggest_fixture),
            ]
        )
    else:
        calls.extend(
            [
                {"method": httpretty.GET, "uri": f"/{index_name}_suggest", "status": HTTPStatus.OK},
                get_put_mapping_call(f"{index_name}_suggest", suggest_fixture),
            ]
        )

    # Bulk operations
    for _ in range(bulk_count):
        calls.append(get_bulk_call(None, get_fixture(bulk_fixture)))

    return calls


# Helper functions for ClusterService tests
def get_cluster_search_call(index_name: str, fixture_rq: str, fixture_rs: str) -> dict:
    """Returns a dictionary representing a GET search call for clustering."""
    return {
        "method": httpretty.POST,
        "uri": f"/{index_name}/_search",
        "status": HTTPStatus.OK,
        "content_type": APPLICATION_JSON,
        "rq": get_fixture(fixture_rq),
        "rs": get_fixture(fixture_rs),
    }


def get_cluster_bulk_call(fixture_rq: str, fixture_rs: str) -> dict:
    """Returns a dictionary representing a POST bulk call for clustering."""
    return {
        "method": httpretty.POST,
        "uri": "/_bulk?refresh=false",
        "status": HTTPStatus.OK,
        "content_type": APPLICATION_JSON,
        "rq": get_fixture(fixture_rq),
        "rs": get_fixture(fixture_rs),
    }


def get_launch_from_fixture(fixture_name: str) -> launch_objects.Launch:
    """Returns a Launch object created from a fixture."""
    return launch_objects.Launch(**(get_fixture(fixture_name, to_json=True)))


def get_launch_info_for_clustering(
    launch: launch_objects.Launch,
    project: int,
    for_update: bool = False,
    number_of_log_lines: int = -1,
    clean_numbers: bool = False,
) -> launch_objects.LaunchInfoForClustering:
    """Returns a LaunchInfoForClustering object with customizable parameters."""
    launch_info = launch_objects.LaunchInfoForClustering(
        launch=launch,
        project=project,
        forUpdate=for_update,
        numberOfLogLines=number_of_log_lines,
    )
    if clean_numbers:
        launch_info.cleanNumbers = clean_numbers
    return launch_info


def get_simple_launch_info_for_clustering(
    launch_id: int, project: int, for_update: bool = False, number_of_log_lines: int = -1
) -> launch_objects.LaunchInfoForClustering:
    """Returns a simple LaunchInfoForClustering object with basic Launch."""
    launch = launch_objects.Launch(launchId=launch_id, project=project)
    return get_launch_info_for_clustering(launch, project, for_update, number_of_log_lines)


def get_cluster_info(
    cluster_id: int, cluster_message: str, log_ids: list[int], item_ids: list[int]
) -> launch_objects.ClusterInfo:
    """Returns a ClusterInfo object with the given parameters."""
    return launch_objects.ClusterInfo(
        clusterId=cluster_id,
        clusterMessage=cluster_message,
        logIds=log_ids,
        itemIds=item_ids,
    )


def get_cluster_result(
    project: int, launch_id: int, clusters: list[launch_objects.ClusterInfo] | None = None
) -> launch_objects.ClusterResult:
    """Returns a ClusterResult object with the given parameters."""
    if clusters is None:
        clusters = []
    return launch_objects.ClusterResult(project=project, launchId=launch_id, clusters=clusters)


def get_basic_cluster_test_calls(index_name: str) -> list[dict]:
    """Returns basic cluster test calls with just index check."""
    return [get_index_found_call(index_name)]


def get_two_search_cluster_calls(
    index_name: str,
    first_search_rq: str,
    first_search_rs: str,
    second_search_rq: str,
    second_search_rs: str,
    bulk_rq: str,
    bulk_rs: str,
) -> list[dict]:
    """Returns cluster test calls with two search operations and a bulk call."""
    return [
        get_index_found_call(index_name),
        get_cluster_search_call(index_name, first_search_rq, first_search_rs),
        get_cluster_search_call(index_name, second_search_rq, second_search_rs),
        get_cluster_bulk_call(bulk_rq, bulk_rs),
    ]


def get_one_search_cluster_calls(
    index_name: str, search_rq: str, search_rs: str, bulk_rq: str, bulk_rs: str
) -> list[dict]:
    """Returns cluster test calls with one search operation and a bulk call."""
    return [
        get_index_found_call(index_name),
        get_cluster_search_call(index_name, search_rq, search_rs),
        get_cluster_bulk_call(bulk_rq, bulk_rs),
    ]


def get_three_search_cluster_calls(
    index_name: str,
    first_search_rq: str,
    first_search_rs: str,
    second_search_rq: str,
    second_search_rs: str,
    third_search_rq: str,
    third_search_rs: str,
    bulk_rq: str,
    bulk_rs: str,
) -> list[dict]:
    """Returns cluster test calls with three search operations and a bulk call."""
    return [
        get_index_found_call(index_name),
        get_cluster_search_call(index_name, first_search_rq, first_search_rs),
        get_cluster_search_call(index_name, second_search_rq, second_search_rs),
        get_cluster_search_call(index_name, third_search_rq, third_search_rs),
        get_cluster_bulk_call(bulk_rq, bulk_rs),
    ]


def get_four_search_cluster_calls(
    index_name: str,
    first_search_rq: str,
    first_search_rs: str,
    second_search_rq: str,
    second_search_rs: str,
    third_search_rq: str,
    third_search_rs: str,
    fourth_search_rq: str,
    fourth_search_rs: str,
    bulk_rq: str,
    bulk_rs: str,
) -> list[dict]:
    """Returns cluster test calls with four search operations and a bulk call."""
    return [
        get_index_found_call(index_name),
        get_cluster_search_call(index_name, first_search_rq, first_search_rs),
        get_cluster_search_call(index_name, second_search_rq, second_search_rs),
        get_cluster_search_call(index_name, third_search_rq, third_search_rs),
        get_cluster_search_call(index_name, fourth_search_rq, fourth_search_rs),
        get_cluster_bulk_call(bulk_rq, bulk_rs),
    ]


# Helper functions for SearchService tests
def get_search_logs_for_search_service(
    log_messages: list[str] | None = None,
    analyzer_config: launch_objects.AnalyzerConf | None = None,
) -> launch_objects.SearchLogs:
    """Returns a SearchLogs object with defaults specific to SearchService tests."""
    if log_messages is None:
        log_messages = ["error"]

    return get_search_logs_object(
        launch_name="Launch 1",
        item_id=3,
        project_id=1,
        filtered_launch_ids=[1],
        log_messages=log_messages,
        analyzer_config=analyzer_config,
    )


def get_search_log_info(
    log_id: int,
    test_item_id: int,
    match_score: int,
) -> launch_objects.SearchLogInfo:
    """Returns a SearchLogInfo object with the given parameters."""
    return launch_objects.SearchLogInfo(
        logId=log_id,
        testItemId=test_item_id,
        matchScore=match_score,
    )


def get_basic_search_test_calls(
    index_name: str,
    fixture_rq: str,
    fixture_rs: str,
) -> list[dict]:
    """Returns basic test calls for search service tests."""
    return [
        get_index_found_call(index_name),
        get_search_for_logs_call_with_parameters(
            index_name,
            get_fixture(fixture_rq),
            get_fixture(fixture_rs),
        ),
    ]


def get_extended_search_test_calls(
    index_name: str,
    fixture_rq1: str,
    fixture_rs1: str,
    fixture_rq2: str,
    fixture_rs2: str,
) -> list[dict]:
    """Returns extended test calls for search service tests with two search calls."""
    return [
        get_index_found_call(index_name),
        get_search_for_logs_call_with_parameters(
            index_name,
            get_fixture(fixture_rq1),
            get_fixture(fixture_rs1),
        ),
        get_search_for_logs_call_with_parameters(
            index_name,
            get_fixture(fixture_rq2),
            get_fixture(fixture_rs2),
        ),
    ]


# Helper functions for AutoAnalyzerService tests
def get_analyzer_index_call(index_name: str) -> dict:
    """Returns a dictionary representing an HTTP call for analyzer index check."""
    return {
        "method": httpretty.GET,
        "uri": f"/{index_name}",
        "status": HTTPStatus.OK,
    }


def get_analyzer_index_not_found_call(index_name: str) -> dict:
    """Returns a dictionary representing an HTTP call for analyzer index not found."""
    return {
        "method": httpretty.GET,
        "uri": f"/{index_name}",
        "status": HTTPStatus.NOT_FOUND,
    }


def get_no_hits_msearch_results(fixture_name: str) -> list[dict]:
    """Returns msearch results with no hits for all queries."""
    return [
        get_fixture(fixture_name, to_json=True),
        get_fixture(fixture_name, to_json=True),
        get_fixture(fixture_name, to_json=True),
        get_fixture(fixture_name, to_json=True),
    ]


def get_mixed_hits_msearch_results(
    no_hits_fixture: str,
    one_hit_fixture: str,
) -> list[dict]:
    """Returns msearch results with mixed hits - one hit in third position."""
    return [
        get_fixture(no_hits_fixture, to_json=True),
        get_fixture(no_hits_fixture, to_json=True),
        get_fixture(one_hit_fixture, to_json=True),
        get_fixture(no_hits_fixture, to_json=True),
    ]


def get_two_hits_msearch_results(
    one_hit_fixture: str,
    two_hits_fixture: str,
    no_hits_fixture: str,
) -> list[dict]:
    """Returns msearch results with two hits pattern."""
    return [
        get_fixture(one_hit_fixture, to_json=True),
        get_fixture(no_hits_fixture, to_json=True),
        get_fixture(two_hits_fixture, to_json=True),
        get_fixture(no_hits_fixture, to_json=True),
    ]


def get_three_hits_msearch_results(
    two_hits_fixture: str,
    three_hits_fixture: str,
    no_hits_fixture: str,
) -> list[dict]:
    """Returns msearch results with three hits pattern."""
    return [
        get_fixture(two_hits_fixture, to_json=True),
        get_fixture(two_hits_fixture, to_json=True),
        get_fixture(three_hits_fixture, to_json=True),
        get_fixture(no_hits_fixture, to_json=True),
    ]


def get_pb_issue_msearch_results(
    no_hits_fixture: str,
    three_hits_fixture: str,
) -> list[dict]:
    """Returns msearch results pattern that typically leads to PB001 issue type."""
    return [
        get_fixture(no_hits_fixture, to_json=True),
        get_fixture(no_hits_fixture, to_json=True),
        get_fixture(three_hits_fixture, to_json=True),
        get_fixture(three_hits_fixture, to_json=True),
    ]


def get_nd_issue_msearch_results(
    no_hits_fixture: str,
    two_hits_fixture: str,
    no_defect_fixture: str,
) -> list[dict]:
    """Returns msearch results pattern that typically leads to ND001 issue type."""
    return [
        get_fixture(no_hits_fixture, to_json=True),
        get_fixture(no_hits_fixture, to_json=True),
        get_fixture(two_hits_fixture, to_json=True),
        get_fixture(no_defect_fixture, to_json=True),
    ]


def get_launch_objects_from_fixture(fixture_content: str) -> list[launch_objects.Launch]:
    """Returns a list of Launch objects created from fixture content."""
    return [launch_objects.Launch(**launch) for launch in json.loads(fixture_content)]


def get_launch_objects_with_analyzer_config(
    fixture_content: str,
    analyzer_config: launch_objects.AnalyzerConf,
) -> list[launch_objects.Launch]:
    """Returns a list of Launch objects from fixture with analyzer config applied."""
    launches = get_launch_objects_from_fixture(fixture_content)
    for launch in launches:
        launch.analyzerConfig = analyzer_config
    return launches


def get_analyzer_config_all_messages_match() -> launch_objects.AnalyzerConf:
    """Returns an AnalyzerConf with allMessagesShouldMatch set to True."""
    return launch_objects.AnalyzerConf(allMessagesShouldMatch=True)


def get_basic_boost_predict_empty() -> tuple[list, list]:
    """Returns empty boost prediction for tests with no expected results."""
    return [], []


def get_boost_predict_single_ab001() -> tuple[list, list]:
    """Returns boost prediction pattern for single AB001 result."""
    return [1], [[0.2, 0.8]]


def get_boost_predict_single_pb001() -> tuple[list, list]:
    """Returns boost prediction pattern for single PB001 result."""
    return [0, 1], [[0.8, 0.2], [0.3, 0.7]]


def get_boost_predict_double_ab001() -> tuple[list, list]:
    """Returns boost prediction pattern for double AB001 results."""
    return [1, 1], [[0.2, 0.8], [0.3, 0.7]]


def get_boost_predict_mixed_ab001() -> tuple[list, list]:
    """Returns boost prediction pattern for mixed AB001 results."""
    return [1, 0], [[0.2, 0.8], [0.7, 0.3]]


def get_boost_predict_nd001() -> tuple[list, list]:
    """Returns boost prediction pattern for ND001 results."""
    return [1], [[0.1, 0.9]]


def get_boost_predict_no_nd001() -> tuple[list, list]:
    """Returns boost prediction pattern for rejected ND001 results."""
    return [0], [[0.9, 0.1]]


def get_boost_predict_pb001_complex() -> tuple[list, list]:
    """Returns boost prediction pattern for complex PB001 results."""
    return [0, 1], [[0.8, 0.2], [0.3, 0.7]]


def get_boost_predict_all_match() -> tuple[list, list]:
    """Returns boost prediction pattern for all messages match scenarios."""
    return [1], [[0.3, 0.7]]
