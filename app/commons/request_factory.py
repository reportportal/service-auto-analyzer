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

"""This module contains functions for preparing requests to Vector DB such as Elasticsearch and OpenSearch."""

from datetime import datetime
from typing import Any

from app.commons.model.launch_objects import Launch, TestItem, Log, TestItemInfo
from app.commons.prepared_log import PreparedLogMessage, PreparedLogMessageClustering
from app.utils import utils, text_processing
from app.utils.log_preparation import clean_message
from app.utils.utils import compute_if_absent


def create_log_template() -> dict:
    return {
        "_id": "",
        "_index": "",
        "_source": {
            "launch_id": "",
            "launch_name": "",
            "launch_number": 0,
            "launch_start_time": "",
            "test_item": "",
            "test_item_name": "",
            "unique_id": "",
            "cluster_id": "",
            "cluster_message": "",
            "test_case_hash": 0,
            "is_auto_analyzed": False,
            "issue_type": "",
            "log_time": "",
            "log_level": 0,
            'original_message': '',
            'message_lines': 0,
            'message_words_number': 0,
            "message": "",
            "is_merged": False,
            "start_time": "",
            "merged_small_logs": "",
            "detected_message": "",
            "detected_message_with_numbers": "",
            "stacktrace": "",
            "only_numbers": "",
            "found_exceptions": "",
            "whole_message": "",
            "potential_status_codes": "",
            "found_tests_and_methods": "",
            "cluster_with_numbers": False
        }
    }


def transform_issue_type_into_lowercase(issue_type):
    return issue_type[:2].lower() + issue_type[2:]


def _fill_launch_test_item_fields(log_template: dict, launch: Launch, test_item: TestItem, project: str):
    log_template['_index'] = project
    source = compute_if_absent(log_template, '_source', {})
    source["launch_id"] = launch.launchId
    source["launch_name"] = launch.launchName
    source["launch_number"] = getattr(launch, 'launchNumber', 0)
    source["launch_start_time"] = datetime(*launch.launchStartTime[:6]).strftime("%Y-%m-%d %H:%M:%S")
    source["test_item"] = test_item.testItemId
    source["unique_id"] = test_item.uniqueId
    source["test_case_hash"] = test_item.testCaseHash
    source["is_auto_analyzed"] = test_item.isAutoAnalyzed
    source["test_item_name"] = text_processing.preprocess_test_item_name(test_item.testItemName)
    source["issue_type"] = transform_issue_type_into_lowercase(test_item.issueType)
    source["start_time"] = datetime(*test_item.startTime[:6]).strftime("%Y-%m-%d %H:%M:%S")
    return log_template


def _fill_common_fields(log_template: dict, log: Log, prepared_log: PreparedLogMessage) -> None:
    log_template['_id'] = log.logId
    source = compute_if_absent(log_template, '_source', {})
    source['cluster_id'] = str(log.clusterId)
    source['cluster_message'] = log.clusterMessage
    source['log_level'] = log.logLevel
    source['original_message'] = log.message
    source['message'] = prepared_log.message
    source['message_lines'] = text_processing.calculate_line_number(prepared_log.clean_message)
    source['message_words_number'] = len(
        text_processing.split_words(prepared_log.clean_message, split_urls=False))
    source['stacktrace'] = prepared_log.stacktrace
    source['potential_status_codes'] = prepared_log.exception_message_potential_status_codes
    source['found_exceptions'] = prepared_log.exception_found
    source['whole_message'] = '\n'.join([prepared_log.exception_message_no_params, prepared_log.stacktrace])


def _fill_log_fields(log_template: dict, log: Log, number_of_lines: int) -> dict[str, Any]:
    prepared_log = PreparedLogMessage(log.message, number_of_lines)
    _fill_common_fields(log_template, log, prepared_log)
    source = compute_if_absent(log_template, '_source', {})
    source["detected_message"] = prepared_log.exception_message_no_numbers
    source["detected_message_with_numbers"] = prepared_log.exception_message
    source["log_time"] = datetime(*log.logTime[:6]).strftime("%Y-%m-%d %H:%M:%S")
    source["cluster_with_numbers"] = utils.extract_clustering_setting(log.clusterId)
    source["only_numbers"] = prepared_log.exception_message_numbers
    source["urls"] = prepared_log.exception_message_urls
    source["paths"] = prepared_log.exception_message_paths
    source["message_params"] = prepared_log.exception_message_params
    source["found_exceptions_extended"] = prepared_log.exception_found_extended
    source["detected_message_extended"] = \
        text_processing.enrich_text_with_method_and_classes(prepared_log.exception_message)
    source["detected_message_without_params_extended"] = \
        text_processing.enrich_text_with_method_and_classes(prepared_log.exception_message_no_params)
    source["stacktrace_extended"] = \
        text_processing.enrich_text_with_method_and_classes(prepared_log.stacktrace)
    source["message_extended"] = \
        text_processing.enrich_text_with_method_and_classes(prepared_log.message)
    source["message_without_params_extended"] = \
        text_processing.enrich_text_with_method_and_classes(prepared_log.message_no_params)
    source["detected_message_without_params_and_brackets"] = \
        prepared_log.exception_message_no_params
    source["message_without_params_and_brackets"] = prepared_log.message_no_params
    source["found_tests_and_methods"] = prepared_log.test_and_methods_extended

    for field in ["message", "detected_message", "detected_message_with_numbers",
                  "stacktrace", "only_numbers", "found_exceptions", "found_exceptions_extended",
                  "detected_message_extended", "detected_message_without_params_extended",
                  "stacktrace_extended", "message_extended", "message_without_params_extended",
                  "detected_message_without_params_and_brackets", "message_without_params_and_brackets"]:
        source[field] = text_processing.leave_only_unique_lines(source[field])
        source[field] = text_processing.clean_colon_stacking(source[field])
    return log_template


def prepare_log(launch: Launch, test_item: TestItem, log: Log, project: str) -> dict:
    log_template = create_log_template()
    log_template = _fill_launch_test_item_fields(log_template, launch, test_item, project)
    log_template = _fill_log_fields(log_template, log, launch.analyzerConfig.numberOfLogLines)
    return log_template


def _fill_test_item_info_fields(log_template: dict, test_item_info: TestItemInfo, project: str) -> dict[str, Any]:
    log_template["_index"] = project
    source = compute_if_absent(log_template, '_source', {})
    source["launch_id"] = test_item_info.launchId
    source["launch_name"] = test_item_info.launchName
    source["launch_number"] = getattr(test_item_info, 'launchNumber', 0)
    source["test_item"] = test_item_info.testItemId
    source["unique_id"] = test_item_info.uniqueId
    source["test_case_hash"] = test_item_info.testCaseHash
    source["test_item_name"] = text_processing.preprocess_test_item_name(
        test_item_info.testItemName)
    source["is_auto_analyzed"] = False
    source["issue_type"] = ""
    source["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return log_template


def prepare_log_for_suggests(test_item_info: TestItemInfo, log: Log, project: str) -> dict:
    log_template = create_log_template()
    log_template = _fill_test_item_info_fields(log_template, test_item_info, project)
    log_template = _fill_log_fields(
        log_template, log, test_item_info.analyzerConfig.numberOfLogLines)
    return log_template


def prepare_log_words(launches: list[Launch]) -> tuple[dict[str, int], int]:
    log_words = {}
    project = None
    for launch in launches:
        project = launch.project
        for test_item in launch.testItems:
            for log in test_item.logs:
                if log.logLevel < utils.ERROR_LOGGING_LEVEL or not log.message.strip():
                    continue
                cleaned_message = clean_message(log.message)
                det_message, stacktrace = text_processing.detect_log_description_and_stacktrace(cleaned_message)
                for word in text_processing.split_words(stacktrace):
                    if '.' in word and len(word.split('.')) > 2:
                        log_words[word] = 1
    return log_words, project


def prepare_log_clustering_light(launch: Launch, test_item: TestItem, log: Log, project: str) -> dict[str, Any]:
    log_template = create_log_template()
    log_template = _fill_launch_test_item_fields(log_template, launch, test_item, project)
    prepared_log = PreparedLogMessageClustering(log.message, -1)
    _fill_common_fields(log_template, log, prepared_log)
    return log_template


def prepare_logs_for_clustering(launch: Launch, project: str) -> list[list[dict[str, Any]]]:
    log_messages = []
    for test_item in launch.testItems:
        prepared_logs = []
        for log in test_item.logs:
            if log.logLevel < utils.ERROR_LOGGING_LEVEL:
                continue
            prepared_logs.append(prepare_log_clustering_light(launch, test_item, log, project))
        log_messages.append(prepared_logs)
    return log_messages
