#  Copyright 2026 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""This module splits log entries of a Test Item into big ones and groups of small ones to merge for clustering."""

from app.commons.model.test_item_index import LogData

SMALL_LOG_MAX_LINES = 2


def is_small_log(log: LogData) -> bool:
    """Check whether the log entry is small, that is it has no more than `SMALL_LOG_MAX_LINES` lines.

    :param log: Log entry to check
    :return: True if the log entry is small
    """
    return log.message_lines is not None and log.message_lines <= SMALL_LOG_MAX_LINES


def merge_small_logs(logs: list[LogData]) -> tuple[list[LogData], list[list[LogData]]]:
    """Split log entries of a Test Item into big ones and groups of small ones to merge.

    Small log entries are grouped by log level. If there is at least one big log entry of the same log level, small
    log entries of this level are dropped, since the big one is in priority.

    :param logs: Log entries of a Test Item, in the order of logs
    :return: Big log entries, in the order of logs, and groups of small log entries per log level, in the order of
             first appearance of the log level
    """
    big_logs: list[LogData] = []
    big_log_levels: set[int] = set()
    small_logs_by_level: dict[int, list[LogData]] = {}
    for log in logs:
        if is_small_log(log):
            small_logs_by_level.setdefault(log.log_level, []).append(log)
        else:
            big_logs.append(log)
            big_log_levels.add(log.log_level)
    small_log_groups = [group for level, group in small_logs_by_level.items() if level not in big_log_levels]
    return big_logs, small_log_groups
