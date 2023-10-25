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

from app.utils import utils, text_processing
from datetime import datetime
from app.commons.log_merger import LogMerger


class LogPreparation:

    def __init__(self):
        self.log_merger = LogMerger()

    def clean_message(self, message):
        # FIXME: unify line endings to '\n' to apply optimizations
        # cleaned_message = text_processing.unify_line_endings(message)
        cleaned_message = text_processing.replace_tabs_for_newlines(message)
        cleaned_message = text_processing.fix_big_encoded_urls(cleaned_message)
        cleaned_message = text_processing.remove_generated_parts(cleaned_message)
        cleaned_message = text_processing.remove_guid_uids_from_text(cleaned_message)
        cleaned_message = text_processing.clean_html(cleaned_message)
        cleaned_message = text_processing.delete_empty_lines(cleaned_message)
        cleaned_message = text_processing.leave_only_unique_lines(cleaned_message)
        return cleaned_message

    def _create_log_template(self) -> dict:
        return {
            "_id":    "",
            "_index": "",
            "_source": {
                "launch_id":        "",
                "launch_name":      "",
                "launch_number":    0,
                "launch_start_time": "",
                "test_item":        "",
                "test_item_name":   "",
                "unique_id":        "",
                "cluster_id":       "",
                "cluster_message":  "",
                "test_case_hash":   0,
                "is_auto_analyzed": False,
                "issue_type":       "",
                "log_time":         "",
                "log_level":        0,
                "original_message_lines": 0,
                "original_message_words_number": 0,
                "message":          "",
                "is_merged":        False,
                "start_time":       "",
                "merged_small_logs":  "",
                "detected_message": "",
                "detected_message_with_numbers": "",
                "stacktrace":                    "",
                "only_numbers":                  "",
                "found_exceptions":              "",
                "whole_message":                 "",
                "potential_status_codes":        "",
                "found_tests_and_methods":       "",
                "cluster_with_numbers":          False}}

    def transform_issue_type_into_lowercase(self, issue_type):
        return issue_type[:2].lower() + issue_type[2:]

    def _fill_launch_test_item_fields(self, log_template, launch, test_item, project):
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
        log_template["_source"]["issue_type"] = self.transform_issue_type_into_lowercase(
            test_item.issueType)
        log_template["_source"]["start_time"] = datetime(
            *test_item.startTime[:6]).strftime("%Y-%m-%d %H:%M:%S")
        return log_template

    def _fill_log_fields(self, log_template, log, number_of_lines):
        cleaned_message = self.clean_message(log.message)

        test_and_methods = text_processing.find_test_methods_in_text(cleaned_message)
        message = text_processing.first_lines(cleaned_message, number_of_lines)
        message = text_processing.replace_text_pieces(message, test_and_methods)
        message_without_params = message
        message = text_processing.delete_empty_lines(text_processing.sanitize_text(message))

        message_without_params = text_processing.clean_from_urls(message_without_params)
        message_without_params = text_processing.clean_from_paths(message_without_params)
        message_without_params = text_processing.clean_from_params(message_without_params)
        message_without_params = text_processing.remove_starting_datetime(message_without_params)
        message_without_params = text_processing.sanitize_text(message_without_params)
        message_without_params_and_brackets = text_processing.clean_from_brackets(message_without_params)

        detected_message, stacktrace = text_processing.detect_log_description_and_stacktrace(cleaned_message)

        detected_message_without_params = detected_message
        urls = " ".join(text_processing.extract_urls(detected_message_without_params))
        detected_message_without_params = text_processing.clean_from_urls(detected_message_without_params)
        paths = " ".join(text_processing.extract_paths(detected_message_without_params))
        detected_message_without_params = text_processing.clean_from_paths(detected_message_without_params)
        potential_status_codes = " ".join(
            text_processing.get_potential_status_codes(detected_message_without_params))
        detected_message_without_params = text_processing.replace_text_pieces(
            detected_message_without_params, test_and_methods)
        detected_message = text_processing.replace_text_pieces(detected_message, test_and_methods)
        detected_message_without_params = text_processing.remove_starting_datetime(detected_message_without_params)
        detected_message_wo_urls_and_paths = detected_message_without_params

        message_params = " ".join(text_processing.extract_message_params(detected_message_without_params))
        detected_message_without_params = text_processing.clean_from_params(detected_message_without_params)
        detected_message_without_params = text_processing.sanitize_text(detected_message_without_params)
        detected_message_without_params_and_brackets = text_processing.clean_from_brackets(
            detected_message_without_params)

        detected_message_with_numbers = text_processing.remove_starting_datetime(detected_message)
        detected_message_only_numbers = text_processing.find_only_numbers(detected_message_with_numbers)
        detected_message = text_processing.sanitize_text(detected_message)
        stacktrace = text_processing.sanitize_text(stacktrace)
        found_exceptions = text_processing.get_found_exceptions(detected_message)
        found_exceptions_extended = text_processing.enrich_found_exceptions(found_exceptions)
        found_test_methods = text_processing.enrich_text_with_method_and_classes(" ".join(test_and_methods))

        log_template["_id"] = log.logId
        log_template["_source"]["log_time"] = datetime(*log.logTime[:6]).strftime("%Y-%m-%d %H:%M:%S")
        log_template["_source"]["cluster_id"] = str(log.clusterId)
        log_template["_source"]["cluster_message"] = log.clusterMessage
        log_template["_source"]["cluster_with_numbers"] = utils.extract_clustering_setting(log.clusterId)
        log_template["_source"]["log_level"] = log.logLevel
        log_template["_source"]["original_message_lines"] = text_processing.calculate_line_number(cleaned_message)
        log_template["_source"]["original_message_words_number"] = len(
            text_processing.split_words(cleaned_message, split_urls=False))
        log_template["_source"]["message"] = message
        log_template["_source"]["detected_message"] = detected_message
        log_template["_source"]["detected_message_with_numbers"] = detected_message_with_numbers
        log_template["_source"]["stacktrace"] = stacktrace
        log_template["_source"]["only_numbers"] = detected_message_only_numbers
        log_template["_source"]["urls"] = urls
        log_template["_source"]["paths"] = paths
        log_template["_source"]["message_params"] = message_params
        log_template["_source"]["found_exceptions"] = found_exceptions
        log_template["_source"]["found_exceptions_extended"] = found_exceptions_extended
        log_template["_source"]["detected_message_extended"] =\
            text_processing.enrich_text_with_method_and_classes(detected_message)
        log_template["_source"]["detected_message_without_params_extended"] =\
            text_processing.enrich_text_with_method_and_classes(detected_message_without_params)
        log_template["_source"]["stacktrace_extended"] =\
            text_processing.enrich_text_with_method_and_classes(stacktrace)
        log_template["_source"]["message_extended"] =\
            text_processing.enrich_text_with_method_and_classes(message)
        log_template["_source"]["message_without_params_extended"] =\
            text_processing.enrich_text_with_method_and_classes(message_without_params)
        log_template["_source"]["whole_message"] = detected_message_wo_urls_and_paths + " \n " + stacktrace
        log_template["_source"]["detected_message_without_params_and_brackets"] =\
            detected_message_without_params_and_brackets
        log_template["_source"]["message_without_params_and_brackets"] =\
            message_without_params_and_brackets
        log_template["_source"]["potential_status_codes"] =\
            potential_status_codes
        log_template["_source"]["found_tests_and_methods"] = found_test_methods

        for field in ["message", "detected_message", "detected_message_with_numbers",
                      "stacktrace", "only_numbers", "found_exceptions", "found_exceptions_extended",
                      "detected_message_extended", "detected_message_without_params_extended",
                      "stacktrace_extended", "message_extended", "message_without_params_extended",
                      "detected_message_without_params_and_brackets",
                      "message_without_params_and_brackets"]:
            log_template["_source"][field] = text_processing.leave_only_unique_lines(log_template["_source"][field])
            log_template["_source"][field] = text_processing.clean_colon_stacking(log_template["_source"][field])
        return log_template

    def _prepare_log(self, launch, test_item, log, project) -> dict:
        log_template = self._create_log_template()
        log_template = self._fill_launch_test_item_fields(log_template, launch, test_item, project)
        log_template = self._fill_log_fields(log_template, log, launch.analyzerConfig.numberOfLogLines)
        return log_template

    def _fill_test_item_info_fields(self, log_template, test_item_info, project):
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

    def _prepare_log_for_suggests(self, test_item_info, log, project):
        log_template = self._create_log_template()
        log_template = self._fill_test_item_info_fields(log_template, test_item_info, project)
        log_template = self._fill_log_fields(
            log_template, log, test_item_info.analyzerConfig.numberOfLogLines)
        return log_template

    def prepare_log_words(self, launches):
        log_words = {}
        project = None
        for launch in launches:
            project = str(launch.project)
            for test_item in launch.testItems:
                for log in test_item.logs:

                    if log.logLevel < utils.ERROR_LOGGING_LEVEL or not log.message.strip():
                        continue
                    clean_message = self.clean_message(log.message)
                    det_message, stacktrace = text_processing.detect_log_description_and_stacktrace(
                        clean_message)
                    for word in text_processing.split_words(stacktrace):
                        if "." in word and len(word.split(".")) > 2:
                            log_words[word] = 1
        return log_words, project

    def prepare_log_clustering_light(self, launch, test_item, log, project):
        log_template = self._create_log_template()
        log_template = self._fill_launch_test_item_fields(log_template, launch, test_item, project)
        cleaned_message = self.clean_message(log.message)
        detected_message, stacktrace = text_processing.detect_log_description_and_stacktrace(
            cleaned_message)
        test_and_methods = text_processing.find_test_methods_in_text(cleaned_message)
        detected_message = text_processing.replace_text_pieces(detected_message, test_and_methods)
        stacktrace = text_processing.sanitize_text(stacktrace)
        message = text_processing.first_lines(cleaned_message, -1)
        message = text_processing.sanitize_text(message)
        log_template["_id"] = log.logId
        log_template["_source"]["cluster_id"] = str(log.clusterId)
        log_template["_source"]["cluster_message"] = log.clusterMessage
        log_template["_source"]["log_level"] = log.logLevel
        log_template["_source"]["original_message_lines"] = text_processing.calculate_line_number(
            cleaned_message)
        log_template["_source"]["original_message_words_number"] = len(
            text_processing.split_words(cleaned_message, split_urls=False))
        detected_message_wo_urls_and_paths = text_processing.clean_from_urls(detected_message)
        detected_message_wo_urls_and_paths = text_processing.clean_from_paths(detected_message_wo_urls_and_paths)
        detected_message_wo_urls_and_paths = text_processing.remove_starting_datetime(
            detected_message_wo_urls_and_paths)
        log_template["_source"]["message"] = message
        log_template["_source"]["detected_message"] = detected_message_wo_urls_and_paths
        log_template["_source"]["detected_message_with_numbers"] = detected_message_wo_urls_and_paths
        log_template["_source"]["stacktrace"] = stacktrace
        potential_status_codes = " ".join(
            text_processing.get_potential_status_codes(detected_message_wo_urls_and_paths))
        log_template["_source"]["potential_status_codes"] = potential_status_codes
        log_template["_source"]["found_exceptions"] = text_processing.get_found_exceptions(detected_message)
        log_template["_source"]["whole_message"] = text_processing.delete_empty_lines(
            detected_message_wo_urls_and_paths + " \n " + stacktrace)
        return log_template

    def prepare_logs_for_clustering(self, launch, number_of_lines, clean_numbers, project):
        log_messages = []
        log_dict = {}
        ind = 0
        full_log_ids_for_merged_logs = {}
        for test_item in launch.testItems:
            prepared_logs = []
            for log in test_item.logs:
                if log.logLevel < utils.ERROR_LOGGING_LEVEL:
                    continue
                prepared_logs.append(
                    self.prepare_log_clustering_light(launch, test_item, log, project))
            merged_logs, log_ids_for_merged_logs = self.log_merger.decompose_logs_merged_and_without_duplicates( # noqa
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
