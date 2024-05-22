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

from datetime import datetime

from app.commons.launch_objects import Launch, TestItem, Log, TestItemInfo
from app.commons.log_merger import LogMerger
from app.commons.prepared_log import PreparedLogMessage
from app.utils import utils, text_processing
from app.utils.log_preparation import basic_prepare


class LogRequests:

    def __init__(self):
        self.log_merger = LogMerger()

    @staticmethod
    def _create_log_template() -> dict:
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
                "original_message_lines": 0,
                "original_message_words_number": 0,
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
                "cluster_with_numbers": False}}

    @staticmethod
    def transform_issue_type_into_lowercase(issue_type):
        return issue_type[:2].lower() + issue_type[2:]

    @staticmethod
    def _fill_launch_test_item_fields(log_template: dict, launch: Launch, test_item: TestItem, project: str):
        log_template["_index"] = project
        log_template["_source"]["launch_id"] = launch.launchId
        log_template["_source"]["launch_name"] = launch.launchName
        log_template["_source"]["launch_number"] = getattr(launch, 'launchNumber', 0)
        log_template["_source"]["launch_start_time"] = datetime(
            *launch.launchStartTime[:6]).strftime("%Y-%m-%d %H:%M:%S")
        log_template["_source"]["test_item"] = test_item.testItemId
        log_template["_source"]["unique_id"] = test_item.uniqueId
        log_template["_source"]["test_case_hash"] = test_item.testCaseHash
        log_template["_source"]["is_auto_analyzed"] = test_item.isAutoAnalyzed
        log_template["_source"]["test_item_name"] = text_processing.preprocess_test_item_name(test_item.testItemName)
        log_template["_source"]["issue_type"] = LogRequests.transform_issue_type_into_lowercase(
            test_item.issueType)
        log_template["_source"]["start_time"] = datetime(
            *test_item.startTime[:6]).strftime("%Y-%m-%d %H:%M:%S")
        return log_template

    @staticmethod
    def _fill_log_fields(log_template: dict, log: Log, number_of_lines: int):
        prepared_log = PreparedLogMessage(log.message, number_of_lines)
        log_template["_id"] = log.logId
        log_template["_source"]["log_time"] = datetime(*log.logTime[:6]).strftime("%Y-%m-%d %H:%M:%S")
        log_template["_source"]["cluster_id"] = str(log.clusterId)
        log_template["_source"]["cluster_message"] = log.clusterMessage
        log_template["_source"]["cluster_with_numbers"] = utils.extract_clustering_setting(log.clusterId)
        log_template["_source"]["log_level"] = log.logLevel
        log_template["_source"]["original_message_lines"] = text_processing.calculate_line_number(
            prepared_log.clean_message)
        log_template["_source"]["original_message_words_number"] = len(
            text_processing.split_words(prepared_log.clean_message, split_urls=False))
        log_template["_source"]["message"] = prepared_log.message
        log_template["_source"]["detected_message"] = prepared_log.exception_message_no_numbers
        log_template["_source"]["detected_message_with_numbers"] = prepared_log.exception_message
        log_template["_source"]["stacktrace"] = prepared_log.stacktrace
        log_template["_source"]["only_numbers"] = prepared_log.exception_message_numbers
        log_template["_source"]["urls"] = prepared_log.exception_message_urls
        log_template["_source"]["paths"] = prepared_log.exception_message_paths
        log_template["_source"]["message_params"] = prepared_log.exception_message_params
        log_template["_source"]["found_exceptions"] = prepared_log.exception_found
        log_template["_source"]["found_exceptions_extended"] = prepared_log.exception_found_extended
        log_template["_source"]["detected_message_extended"] = \
            text_processing.enrich_text_with_method_and_classes(prepared_log.exception_message_no_numbers)
        log_template["_source"]["detected_message_without_params_extended"] = \
            text_processing.enrich_text_with_method_and_classes(prepared_log.exception_message_no_params)
        log_template["_source"]["stacktrace_extended"] = \
            text_processing.enrich_text_with_method_and_classes(prepared_log.stacktrace)
        log_template["_source"]["message_extended"] = \
            text_processing.enrich_text_with_method_and_classes(prepared_log.message)
        log_template["_source"]["message_without_params_extended"] = \
            text_processing.enrich_text_with_method_and_classes(prepared_log.message_no_params)
        log_template["_source"]["whole_message"] = (prepared_log.exception_message_no_urls_paths + "\n"
                                                    + prepared_log.stacktrace)
        log_template["_source"]["detected_message_without_params_and_brackets"] = \
            prepared_log.exception_message_no_params_and_brackets
        log_template["_source"]["message_without_params_and_brackets"] = prepared_log.message_no_params_and_brackets
        log_template["_source"]["potential_status_codes"] = prepared_log.exception_message_potential_status_codes
        log_template["_source"]["found_tests_and_methods"] = prepared_log.test_and_methods_extended

        for field in ["message", "detected_message", "detected_message_with_numbers",
                      "stacktrace", "only_numbers", "found_exceptions", "found_exceptions_extended",
                      "detected_message_extended", "detected_message_without_params_extended",
                      "stacktrace_extended", "message_extended", "message_without_params_extended",
                      "detected_message_without_params_and_brackets",
                      "message_without_params_and_brackets"]:
            log_template["_source"][field] = text_processing.leave_only_unique_lines(log_template["_source"][field])
            log_template["_source"][field] = text_processing.clean_colon_stacking(log_template["_source"][field])
        return log_template

    @staticmethod
    def _prepare_log(launch: Launch, test_item: TestItem, log: Log, project: str) -> dict:
        log_template = LogRequests._create_log_template()
        log_template = LogRequests._fill_launch_test_item_fields(log_template, launch, test_item, project)
        log_template = LogRequests._fill_log_fields(log_template, log, launch.analyzerConfig.numberOfLogLines)
        return log_template

    @staticmethod
    def _fill_test_item_info_fields(log_template: dict, test_item_info: TestItemInfo, project: str) -> dict:
        log_template["_index"] = project
        log_template["_source"]["launch_id"] = test_item_info.launchId
        log_template["_source"]["launch_name"] = test_item_info.launchName
        log_template["_source"]["launch_number"] = getattr(test_item_info, 'launchNumber', 0)
        log_template["_source"]["test_item"] = test_item_info.testItemId
        log_template["_source"]["unique_id"] = test_item_info.uniqueId
        log_template["_source"]["test_case_hash"] = test_item_info.testCaseHash
        log_template["_source"]["test_item_name"] = text_processing.preprocess_test_item_name(
            test_item_info.testItemName)
        log_template["_source"]["is_auto_analyzed"] = False
        log_template["_source"]["issue_type"] = ""
        log_template["_source"]["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return log_template

    @staticmethod
    def _prepare_log_for_suggests(test_item_info: TestItemInfo, log: Log, project: str) -> dict:
        log_template = LogRequests._create_log_template()
        log_template = LogRequests._fill_test_item_info_fields(log_template, test_item_info, project)
        log_template = LogRequests._fill_log_fields(
            log_template, log, test_item_info.analyzerConfig.numberOfLogLines)
        return log_template

    @staticmethod
    def prepare_log_words(launches):
        log_words = {}
        project = None
        for launch in launches:
            project = str(launch.project)
            for test_item in launch.testItems:
                for log in test_item.logs:

                    if log.logLevel < utils.ERROR_LOGGING_LEVEL or not log.message.strip():
                        continue
                    cleaned_message = basic_prepare(log.message)
                    det_message, stacktrace = text_processing.detect_log_description_and_stacktrace(cleaned_message)
                    for word in text_processing.split_words(stacktrace):
                        if "." in word and len(word.split(".")) > 2:
                            log_words[word] = 1
        return log_words, project

    @staticmethod
    def prepare_log_clustering_light(launch: Launch, test_item: TestItem, log: Log, project: str):
        log_template = LogRequests._create_log_template()
        log_template = LogRequests._fill_launch_test_item_fields(log_template, launch, test_item, project)
        prepared_log = PreparedLogMessage(log.message, -1)
        log_template["_id"] = log.logId
        log_template["_source"]["cluster_id"] = str(log.clusterId)
        log_template["_source"]["cluster_message"] = log.clusterMessage
        log_template["_source"]["log_level"] = log.logLevel
        log_template["_source"]["original_message_lines"] = text_processing.calculate_line_number(
            prepared_log.clean_message)
        log_template["_source"]["original_message_words_number"] = len(
            text_processing.split_words(prepared_log.clean_message, split_urls=False))
        log_template["_source"]["message"] = text_processing.remove_numbers(prepared_log.message)
        log_template["_source"]["detected_message"] = prepared_log.exception_message_no_numbers
        log_template["_source"]["detected_message_with_numbers"] = prepared_log.exception_message
        log_template["_source"]["stacktrace"] = prepared_log.stacktrace
        log_template["_source"]["potential_status_codes"] = prepared_log.exception_message_potential_status_codes
        log_template["_source"]["found_exceptions"] = prepared_log.exception_found
        log_template["_source"]["whole_message"] = (prepared_log.exception_message_no_urls_paths + "\n"
                                                    + prepared_log.stacktrace)
        return log_template

    def prepare_logs_for_clustering(self, launch: Launch, number_of_lines: int, clean_numbers: bool, project: str):
        log_messages = []
        log_dict = {}
        ind = 0
        full_log_ids_for_merged_logs = {}
        for test_item in launch.testItems:
            prepared_logs = []
            for log in test_item.logs:
                if log.logLevel < utils.ERROR_LOGGING_LEVEL:
                    continue
                prepared_logs.append(LogRequests.prepare_log_clustering_light(launch, test_item, log, project))
            merged_logs, log_ids_for_merged_logs = self.log_merger.decompose_logs_merged_and_without_duplicates(
                prepared_logs)
            for _id in log_ids_for_merged_logs:
                full_log_ids_for_merged_logs[_id] = log_ids_for_merged_logs[_id]
            for log in merged_logs:
                number_of_log_lines = number_of_lines
                if log["_source"]["is_merged"]:
                    number_of_log_lines = -1
                log_message = text_processing.prepare_message_for_clustering(
                    log["_source"]["whole_message"], number_of_log_lines, clean_numbers)
                if not log_message.strip():
                    continue
                log_messages.append(log_message)
                log_dict[ind] = log
                ind += 1
        return log_messages, log_dict, full_log_ids_for_merged_logs
