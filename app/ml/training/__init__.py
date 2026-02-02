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

"""Common package for ML Model training code."""
from typing import Any

from app.commons.model.test_item_index import TestItemHistoryData

MAX_HISTORY_NEGATIVES = 2


def normalize_issue_type(issue_type: Any) -> str:
    """
    Normalize issue type to lowercase string.

    :param issue_type: Raw issue type
    :return: Normalized issue type
    """
    if issue_type is None:
        return ""
    return str(issue_type).strip().lower()


def select_history_negative_types(
    issue_history: list[TestItemHistoryData],
    positive_issue_type: str,
) -> list[str]:
    """
    Pick up to MAX_HISTORY_NEGATIVES negative issue types from history.

    :param issue_history: Test item issue history entries
    :param positive_issue_type: Current issue type (positive class)
    :return: List of selected negative issue types
    """
    negatives = []
    unique_negatives = set()
    for entry in reversed(issue_history[:-1]):
        entry_type = normalize_issue_type(entry.issue_type)
        if not entry_type or entry_type == positive_issue_type or entry_type in unique_negatives:
            continue
        negatives.append(entry_type)
        unique_negatives.add(entry_type)
        if len(negatives) >= MAX_HISTORY_NEGATIVES:
            break
    return negatives
