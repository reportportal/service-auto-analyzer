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
import logging
from typing import Any

from utils import utils

from app.commons.model.test_item_index import TestItemHistoryData

LOGGER = logging.getLogger("analyzerApp.training")

MAX_HISTORY_NEGATIVES = 2
DUE_PROPORTION = 0.2


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


def validate_proportions(labels: list[int]) -> tuple[bool, float]:
    positives_count = sum(1 for label in labels if label == 1)
    negatives_count = sum(1 for label in labels if label == 0)
    bad_data = False
    if positives_count < 2 or negatives_count < 2:
        LOGGER.debug("Train data has too few samples: positives=%d, negatives=%d", positives_count, negatives_count)
        bad_data = True
    data_proportion = utils.calculate_proportions_for_labels(labels)
    if data_proportion < DUE_PROPORTION:
        LOGGER.debug("Train data has a bad proportion: %.3f", data_proportion)
        bad_data = True
    return bad_data, data_proportion
