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

"""Stateless error identity of a Test Item, used to find earlier runs which failed in exactly the same way."""

import hashlib
import json
from typing import Optional

from app.commons.model.test_item_index import LogData

# Bump on any change of the hash input, so that hashes computed by different versions never match
ERROR_HASH_VERSION = "v1"
ERROR_HASH_MAX_LINES = 40
ERROR_HASH_DIGEST_SIZE = 8


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _message_lines(log: LogData) -> list[str]:
    """Take the lines of a log's exception message, numbers are already removed there by log preparation.

    :param log: Prepared log entry
    :return: Non-empty, whitespace-normalized lines, at most `ERROR_HASH_MAX_LINES`
    """
    text = log.detected_message or log.message or ""
    lines = [_normalize(line) for line in text.split("\n")[:ERROR_HASH_MAX_LINES]]
    return [line for line in lines if line]


def compute_error_hash(logs: list[LogData]) -> Optional[str]:
    """Compute the error hash of a Test Item from its prepared logs.

    The hash covers, in the order of logs: the exception message lines (de-duplicated across logs), the exceptions
    found in each log and the exact status code sequence of each log. Status codes are compared the same way as by
    the status code filter of the search: order is kept and a log without codes differs from a log with codes.

    :param logs: Prepared log entries of the Test Item, in the order of logs
    :return: Hex digest, or None if the logs have neither message lines nor exceptions to identify the error by
    """
    lines: list[str] = []
    for log in logs:
        for line in _message_lines(log):
            if line not in lines:
                lines.append(line)
    exceptions = [_normalize(log.found_exceptions or "") for log in logs]
    if not lines and not any(exceptions):
        return None
    status_codes = [_normalize(log.potential_status_codes or "") for log in logs]
    payload = json.dumps([ERROR_HASH_VERSION, lines, exceptions, status_codes])
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=ERROR_HASH_DIGEST_SIZE).hexdigest()
