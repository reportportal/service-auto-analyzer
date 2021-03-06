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

import utils.utils as utils
import copy


class LogMerger:

    @staticmethod
    def merge_big_and_small_logs(logs, log_level_ids_to_add,
                                 log_level_messages, log_level_ids_merged):
        """Merge big message logs with small ones"""
        new_logs = []
        for log in logs:
            if not log["_source"]["message"].strip():
                continue
            log_level = log["_source"]["log_level"]

            if log["_id"] in log_level_ids_to_add[log_level]:
                merged_small_logs = utils.compress(log_level_messages["message"][log_level])
                new_logs.append(LogMerger.prepare_new_log(
                    log, log["_id"], False, merged_small_logs))

        for log_level in log_level_messages["message"]:

            if not log_level_ids_to_add[log_level] and\
               log_level_messages["message"][log_level].strip():
                log = log_level_ids_merged[log_level]
                new_log = LogMerger.prepare_new_log(
                    log, str(log["_id"]) + "_m", True,
                    utils.compress(log_level_messages["message"][log_level]),
                    fields_to_clean=["message", "detected_message", "only_numbers",
                                     "detected_message_with_numbers", "stacktrace",
                                     "found_exceptions_extended", "detected_message_extended",
                                     "detected_message_without_params_extended",
                                     "stacktrace_extended", "message_extended",
                                     "message_without_params_extended",
                                     "urls", "paths", "message_params",
                                     "message_without_params_and_brackets",
                                     "detected_message_without_params_and_brackets"])
                for field in log_level_messages:
                    if field in ["message"]:
                        continue
                    new_log["_source"][field] = utils.compress(
                        log_level_messages[field][log_level])
                new_log["_source"]["found_exceptions_extended"] = utils.compress(
                    utils.enrich_found_exceptions(log_level_messages["found_exceptions"][log_level]))

                new_logs.append(new_log)
        return new_logs

    @staticmethod
    def decompose_logs_merged_and_without_duplicates(logs):
        """Merge big logs with small ones without duplcates"""
        log_level_messages = {"message": {}, "found_exceptions": {}, "potential_status_codes": {}}
        log_level_ids_to_add = {}
        log_level_ids_merged = {}
        logs_unique_log_level = {}

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

            if log["_source"]["original_message_lines"] <= 2 and\
                    log["_source"]["original_message_words_number"] <= 100:
                if log_level not in log_level_ids_merged:
                    log_level_ids_merged[log_level] = log

                log_level_representative = log_level_ids_merged[log_level]
                current_log_word_num = log["_source"]["original_message_words_number"]
                main_log_word_num = log_level_representative["_source"]["original_message_words_number"]
                if current_log_word_num > main_log_word_num:
                    log_level_ids_merged[log_level] = log

                normalized_msg = " ".join(log["_source"]["message"].strip().lower().split())
                if normalized_msg not in logs_unique_log_level[log_level]:
                    logs_unique_log_level[log_level].add(normalized_msg)

                    for field in log_level_messages:
                        splitter = "\r\n" if field == "message" else " "
                        log_level_messages[field][log_level] =\
                            log_level_messages[field][log_level] + log["_source"][field] + splitter

            else:
                log_level_ids_to_add[log_level].append(log["_id"])

        return LogMerger.merge_big_and_small_logs(logs, log_level_ids_to_add,
                                                  log_level_messages, log_level_ids_merged)

    @staticmethod
    def prepare_new_log(old_log, new_id, is_merged, merged_small_logs, fields_to_clean=[]):
        """Prepare updated log"""
        merged_log = copy.deepcopy(old_log)
        merged_log["_source"]["is_merged"] = is_merged
        merged_log["_id"] = new_id
        merged_log["_source"]["merged_small_logs"] = merged_small_logs
        for field in fields_to_clean:
            merged_log["_source"][field] = ""
        return merged_log
