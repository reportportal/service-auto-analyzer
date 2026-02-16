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
import dataclasses
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from app.commons.model.test_item_index import TestItemHistoryData
from app.utils import utils
from app.utils.utils import normalize_issue_type

LOGGER = logging.getLogger("analyzerApp.training")

DEFAULT_RANDOM_SEED = 1257
TRAIN_DATA_RANDOM_STATES = [DEFAULT_RANDOM_SEED, 1873, 1917, 2477, 3449, 353, 4561, 5417, 6427, 2029, 2137]

NEGATIVE_RATIO_MAX = 4
MAX_HISTORY_NEGATIVES = 2
DUE_PROPORTION = 0.2

T = TypeVar("T")


@dataclass(frozen=True)
class TrainingEntry(Generic[T]):
    data: T
    issue_type: str
    is_positive: bool


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


def build_issue_history_query(chunk_number: int, fields_to_retrieve: list[str]) -> dict[str, Any]:
    return {
        "_source": fields_to_retrieve,
        "size": chunk_number,
        "query": {
            "nested": {
                "path": "issue_history",
                "query": {"exists": {"field": "issue_history.issue_type"}},
            }
        },
    }


def balance_data(
    train_data: list[TrainingEntry[T]],
) -> list[TrainingEntry[T]]:
    """Make existing train data balanced for the given label.

    This function shorten the amount of negative cases if there are to many of them and extend them out of existing
    data if there are too few of them.

    :param train_data: Existing train data based on item history.
    """
    cases: dict[str, list[TrainingEntry[T]]] = defaultdict(list)
    for entry in train_data:
        cases[entry.issue_type].append(entry)

    cases_num: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    negative_cases: dict[str, list[int]] = defaultdict(list)
    for i, entry in enumerate(train_data):
        current = cases_num[entry.issue_type]
        if entry.is_positive:
            cases_num[entry.issue_type] = current[0] + 1, current[1]
        else:
            negative_cases[entry.issue_type].append(i)
            cases_num[entry.issue_type] = current[0], current[1] + 1

    rnd = random.Random(DEFAULT_RANDOM_SEED)
    results: list[TrainingEntry[T]] = train_data.copy()
    for issue_type, (positive, negative) in cases_num.items():
        if positive <= 0:
            continue
        max_negative_cases = positive * NEGATIVE_RATIO_MAX
        additional_negative_cases: list[TrainingEntry[T]] = []
        for issue, case_list in cases.items():
            if issue_type == issue:
                continue
            for case in case_list:
                if not case.is_positive:
                    continue  # ignore uncertainty
                additional_negative_cases.append(dataclasses.replace(case, issue_type=issue_type, is_positive=False))
        if not additional_negative_cases:
            continue
        rnd.shuffle(additional_negative_cases)
        if negative + len(additional_negative_cases) <= max_negative_cases:
            # We can append all additional negative cases
            results.extend(additional_negative_cases)
        else:
            # Need to balance negative cases
            negative_indexes = negative_cases[issue_type]
            rnd.shuffle(negative_indexes)
            case_num_to_remove = negative + len(additional_negative_cases) - max_negative_cases
            if negative <= positive:
                remove_history_cases_num = 0
            else:
                possible_to_remove = negative - positive
                remove_history_cases_num = (
                    possible_to_remove if possible_to_remove < case_num_to_remove else case_num_to_remove
                )
            if remove_history_cases_num < case_num_to_remove:
                remove_additional_cases_num = case_num_to_remove - remove_history_cases_num
            else:
                remove_additional_cases_num = 0
            additional_negative_cases = additional_negative_cases[
                : len(additional_negative_cases) - remove_additional_cases_num
            ]
            cases_to_remove = negative_indexes[:remove_history_cases_num]
            i = 0
            for i, case_idx in enumerate(cases_to_remove):
                results[case_idx] = additional_negative_cases[i]
            if i + 1 < len(additional_negative_cases):
                results.extend(additional_negative_cases[i + 1 :])
    return results


def _get_base_issue_type(issue_type: str) -> str:
    return issue_type[:2] if issue_type else ""


def is_supported_issue_type(issue_type: str) -> bool:
    return _get_base_issue_type(issue_type) != "ti"


def get_issue_type(issue_type: str) -> str:
    if not issue_type:
        return issue_type
    return _get_base_issue_type(issue_type) if re.match(r"^[a-z]{2}\d{,3}$", issue_type) else issue_type
