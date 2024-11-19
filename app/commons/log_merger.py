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

import copy
from typing import Any, Optional

from app.utils import text_processing

FIELDS_TO_CLEAN = ["message", "detected_message", "detected_message_with_numbers", "detected_message_extended",
                   "message_extended", "message_without_params_extended", "message_without_params_and_brackets",
                   "detected_message_without_params_and_brackets"]
FIELDS_TO_MERGE = ["message", "found_exceptions", "potential_status_codes", "found_tests_and_methods", "only_numbers",
                   "urls", "paths", "message_params", "detected_message_without_params_extended", "whole_message"]


def _prepare_new_log(old_log: dict[str, Any], new_id, is_merged: bool, merged_small_logs: str,
                     fields_to_clean: Optional[list[str]] = None) -> dict[str, Any]:
    """Prepare updated log"""
    merged_log = copy.deepcopy(old_log)
    merged_log["_source"]["is_merged"] = is_merged
    merged_log["_id"] = new_id
    merged_log["_source"]["merged_small_logs"] = merged_small_logs
    if fields_to_clean:
        for field in fields_to_clean:
            merged_log["_source"][field] = ""
    return merged_log


def merge_big_and_small_logs(
        logs: list[dict[str, Any]], log_level_ids_to_add: dict[int, list[int]],
        log_level_messages: dict[str, dict[int, str]], log_level_ids_merged: dict[int, dict[str, Any]],
        logs_ids_in_merged_logs: dict[int, list[int]]) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Merge big message logs with small ones."""
    new_logs = []
    for log in logs:
        if not log["_source"]["message"].strip():
            continue
        log_level = log["_source"]["log_level"]

        if log["_id"] in log_level_ids_to_add[log_level]:
            merged_small_logs = text_processing.compress(log_level_messages["message"][log_level])
            new_logs.append(_prepare_new_log(log, log["_id"], False, merged_small_logs))

    log_ids_for_merged_logs = {}
    for log_level in log_level_messages["message"]:
        if not log_level_ids_to_add[log_level] and log_level_messages["message"][log_level].strip():
            log = log_level_ids_merged[log_level]
            merged_logs_id = str(log["_id"]) + "_m"
            new_log = _prepare_new_log(
                log, merged_logs_id, True, text_processing.compress(log_level_messages["message"][log_level]),
                fields_to_clean=FIELDS_TO_CLEAN)
            log_ids_for_merged_logs[merged_logs_id] = logs_ids_in_merged_logs[log_level]
            for field in log_level_messages:
                if field == "message":
                    continue
                if field == "whole_message":
                    new_log["_source"][field] = log_level_messages[field][log_level]
                else:
                    new_log["_source"][field] = text_processing.compress(
                        log_level_messages[field][log_level])
            new_log["_source"]["found_exceptions_extended"] = text_processing.compress(
                text_processing.enrich_found_exceptions(log_level_messages["found_exceptions"][log_level]))

            new_logs.append(new_log)
    return new_logs, log_ids_for_merged_logs


def decompose_logs_merged_and_without_duplicates(
        logs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Merge big logs with small ones without duplicates."""
    log_level_messages = {}
    for field in FIELDS_TO_MERGE:
        log_level_messages[field] = {}
    log_level_ids_to_add = {}
    log_level_ids_merged = {}
    logs_unique_log_level = {}
    logs_ids_in_merged_logs = {}

    for log in logs:
        if not log["_source"]["message"].strip():
            continue

        log_level = log["_source"]["log_level"]

        for field in log_level_messages:
            if log_level not in log_level_messages[field]:
                log_level_messages[field][log_level] = ""
        if log_level not in log_level_ids_to_add:
            log_level_ids_to_add[log_level] = []
        if log_level not in logs_unique_log_level:
            logs_unique_log_level[log_level] = set()

        if log["_source"]["original_message_lines"] <= 2 and log["_source"]["original_message_words_number"] <= 100:
            if log_level not in log_level_ids_merged:
                log_level_ids_merged[log_level] = log
            if log_level not in logs_ids_in_merged_logs:
                logs_ids_in_merged_logs[log_level] = []
            logs_ids_in_merged_logs[log_level].append(log["_id"])

            log_level_representative = log_level_ids_merged[log_level]
            current_log_word_num = log["_source"]["original_message_words_number"]
            main_log_word_num = log_level_representative["_source"]["original_message_words_number"]
            if current_log_word_num > main_log_word_num:
                log_level_ids_merged[log_level] = log

            normalized_msg = " ".join(log["_source"]["message"].strip().lower().split())
            if normalized_msg not in logs_unique_log_level[log_level]:
                logs_unique_log_level[log_level].add(normalized_msg)

                for field in log_level_messages:
                    if field in log["_source"]:
                        splitter = "\n" if field in {"message", "whole_message"} else " "
                        log_level_messages[field][log_level] = \
                            log_level_messages[field][log_level] + log["_source"][field] + splitter

        else:
            log_level_ids_to_add[log_level].append(log["_id"])

    return merge_big_and_small_logs(
        logs, log_level_ids_to_add, log_level_messages, log_level_ids_merged, logs_ids_in_merged_logs)


def merge_logs(
        logs: list[list[dict[str, Any]]], number_of_lines: int,
        clean_numbers: bool) -> tuple[list[str], dict[int, dict[str, Any]], dict[str, list[int]]]:
    full_log_ids_for_merged_logs = {}
    log_messages = []
    log_dict = {}
    ind = 0
    for prepared_log in logs:
        merged_logs, log_ids_for_merged_logs = decompose_logs_merged_and_without_duplicates(prepared_log)
        for _id, merged_list in log_ids_for_merged_logs.items():
            full_log_ids_for_merged_logs[_id] = merged_list
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
