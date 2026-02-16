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

from collections import Counter

import pytest

from app.ml.training import TrainingEntry, balance_data


def _entry(message: str, issue_type: str, is_positive: bool) -> TrainingEntry:
    return TrainingEntry(data=message, project_id=None, issue_type=issue_type, is_positive=is_positive)


def _entry_counter(entries: list[TrainingEntry[str]]) -> Counter[tuple[str, str, bool]]:
    return Counter((entry.data, entry.issue_type, entry.is_positive) for entry in entries)


FIRST_POSITIVE_ENTRY = _entry("msg-a", "pb001", True)
FIRST_CROSS_NEGATIVE_ENTRY = _entry("msg-a", "ab001", False)
SECOND_POSITIVE_ENTRY = _entry("msg-b", "ab001", True)
SECOND_CROSS_NEGATIVE_ENTRY = _entry("msg-b", "pb001", False)
FIRST_NEGATIVE_ENTRY = _entry("msg-c", "pb001", False)
SECOND_NEGATIVE_ENTRY = _entry("msg-c", "ab001", False)
FORTH_NEGATIVE_ENTRY = _entry("msg-d", "pb001", False)
FIFTH_NEGATIVE_ENTRY = _entry("msg-d", "ab001", False)
OTHER_NEGATIVE_CASES = [
    _entry("msg-e", "pb001", False),
    _entry("msg-f", "pb001", False),
    _entry("msg-e", "ab001", False),
    _entry("msg-f", "ab001", False),
]


@pytest.mark.parametrize(
    (
        "train_data",
        "expected_entries",
    ),
    [
        pytest.param(
            [],
            [],
            id="empty-data",
        ),
        pytest.param(
            [FIRST_POSITIVE_ENTRY],
            [FIRST_POSITIVE_ENTRY],
            id="single-positive",
        ),
        pytest.param(
            [FIRST_POSITIVE_ENTRY, FIRST_CROSS_NEGATIVE_ENTRY],
            [FIRST_POSITIVE_ENTRY, FIRST_CROSS_NEGATIVE_ENTRY],
            id="same-message-opposite-labels",
        ),
        pytest.param(
            [FIRST_POSITIVE_ENTRY, SECOND_POSITIVE_ENTRY],
            [
                FIRST_POSITIVE_ENTRY,
                SECOND_POSITIVE_ENTRY,
                SECOND_CROSS_NEGATIVE_ENTRY,
                FIRST_CROSS_NEGATIVE_ENTRY,
            ],
            id="two-positives-create-cross-negatives",
        ),
        pytest.param(
            [
                FIRST_POSITIVE_ENTRY,
                SECOND_POSITIVE_ENTRY,
                FIRST_NEGATIVE_ENTRY,
                SECOND_NEGATIVE_ENTRY,
            ],
            [
                FIRST_POSITIVE_ENTRY,
                SECOND_POSITIVE_ENTRY,
                FIRST_NEGATIVE_ENTRY,
                SECOND_NEGATIVE_ENTRY,
                SECOND_CROSS_NEGATIVE_ENTRY,
                FIRST_CROSS_NEGATIVE_ENTRY,
            ],
            id="two-positives-keep-negatives-add-cross",
        ),
        pytest.param(
            [
                FIRST_POSITIVE_ENTRY,
                SECOND_POSITIVE_ENTRY,
                FIRST_NEGATIVE_ENTRY,
                SECOND_NEGATIVE_ENTRY,
                FORTH_NEGATIVE_ENTRY,
                FIFTH_NEGATIVE_ENTRY,
            ]
            + OTHER_NEGATIVE_CASES,
            [
                FIRST_POSITIVE_ENTRY,
                SECOND_POSITIVE_ENTRY,
                FIRST_NEGATIVE_ENTRY,
                FIRST_CROSS_NEGATIVE_ENTRY,
                FORTH_NEGATIVE_ENTRY,
                FIFTH_NEGATIVE_ENTRY,
                SECOND_CROSS_NEGATIVE_ENTRY,
                OTHER_NEGATIVE_CASES[1],
                OTHER_NEGATIVE_CASES[2],
                OTHER_NEGATIVE_CASES[3],
            ],
            id="two-positives-prune-negatives-add-cross",
        ),
    ],
)
def test_balance_data(
    train_data: list[TrainingEntry],
    expected_entries: list[TrainingEntry],
) -> None:
    result = balance_data(train_data)
    assert result == expected_entries
