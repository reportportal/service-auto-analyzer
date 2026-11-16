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
from typing import Optional

import pytest

from app.ml.training import TrainingEntry, balance_data
from app.ml.training.train_defect_type_model import build_one_vs_rest_data


def _entry(message: str, issue_type: str, is_positive: bool) -> TrainingEntry[str]:
    return TrainingEntry(data=message, project_id=None, issue_type=issue_type, is_positive=is_positive)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param([], [], id="empty-data"),
        pytest.param(
            [
                _entry("ab-1", "ab", True),
                _entry("ab-neg-1", "ab", False),
                _entry("ab-neg-2", "ab", False),
                _entry("pb-1", "pb", True),
                _entry("pb-neg-1", "pb", False),
            ],
            [("ab-1", 1), ("ab-neg-1", 0), ("ab-neg-2", 0)],
            id="ratio-reached-no-additional-negatives",
        ),
        pytest.param(
            [
                _entry("ab-1", "ab", True),
                _entry("ab-2", "ab", True),
                _entry("ab-neg-1", "ab", False),
                _entry("pb-1", "pb", True),
                _entry("si-1", "si", True),
                _entry("si-2", "si", True),
                _entry("si-3", "si", True),
                _entry("si-4", "si", True),
            ],
            None,
            id="additional-negatives-up-to-ratio",
        ),
        pytest.param(
            [
                _entry("ab-1", "ab", True),
                _entry("ab-1", "pb", True),
                _entry("pb-neg-1", "pb", False),
                _entry("si-1", "si", True),
                _entry("si-1", "si", True),
            ],
            [("ab-1", 1), ("si-1", 0)],
            id="skip-known-and-duplicate-messages",
        ),
    ],
)
def test_build_one_vs_rest_data(data: list[TrainingEntry[str]], expected: Optional[list[tuple[str, int]]]) -> None:
    messages, labels = build_one_vs_rest_data("ab", data)
    if expected is not None:
        assert Counter(zip(messages, labels)) == Counter(expected)
        return
    assert Counter(labels) == {1: 2, 0: 4}
    assert set(zip(messages, labels)) >= {("ab-1", 1), ("ab-2", 1), ("ab-neg-1", 0)}
    assert all(m.startswith(("ab-neg", "pb-", "si-")) for m, label in zip(messages, labels) if label == 0)


def test_build_one_vs_rest_data_after_balance_data() -> None:
    data = balance_data(
        [
            _entry("ab-1", "ab", True),
            _entry("pb-1", "pb", True),
            _entry("si-1", "si", True),
        ]
    )
    messages, labels = build_one_vs_rest_data("ab", data)
    assert Counter(zip(messages, labels)) == Counter([("ab-1", 1), ("pb-1", 0), ("si-1", 0)])
