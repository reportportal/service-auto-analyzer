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
                                     "detected_message_with_numbers", "stacktrace"])
                new_log["_source"]["found_exceptions"] = utils.compress(
                    log_level_messages["found_exceptions"][log_level])

                new_logs.append(new_log)
        return new_logs

    @staticmethod
    def decompose_logs_merged_and_without_duplicates(logs):
        """Merge big logs with small ones without duplcates"""
        log_level_messages = {"message": {}, "found_exceptions": {}}
        log_level_ids_to_add = {}
        log_level_ids_merged = {}
        logs_unique_log_level = {}

        for log in logs:
            if not log["_source"]["message"].strip():
                continue

            log_level = log["_source"]["log_level"]

            for field in ["message", "found_exceptions"]:
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
                message = log["_source"]["message"]
                normalized_msg = " ".join(message.strip().lower().split())
                if normalized_msg not in logs_unique_log_level[log_level]:
                    logs_unique_log_level[log_level].add(normalized_msg)
                    for field in ["message", "found_exceptions"]:
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
