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

import pytest

from app.ml.training import TrainingEntry
from app.ml.training.train_defect_type_model import remove_conflicting_entries


def _entry(message: str, issue_type: str, is_positive: bool) -> TrainingEntry[str]:
    return TrainingEntry(data=message, project_id=None, issue_type=issue_type, is_positive=is_positive)


UNRELATED_POSITIVE = _entry("msg-unrelated", "si", True)
UNRELATED_NEGATIVE = _entry("msg-unrelated", "ab", False)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param([], [], id="empty-data"),
        pytest.param(
            [
                _entry("msg-a", "ab", True),
                UNRELATED_POSITIVE,
                _entry("msg-a", "pb", True),
                _entry("msg-a", "ab", True),
                UNRELATED_NEGATIVE,
            ],
            [UNRELATED_POSITIVE, UNRELATED_NEGATIVE],
            id="positive-for-two-labels",
        ),
        pytest.param(
            [
                _entry("msg-a", "ab", True),
                _entry("msg-a", "pb", False),
                UNRELATED_POSITIVE,
                _entry("msg-a", "ab", False),
            ],
            [UNRELATED_POSITIVE],
            id="positive-and-negative-for-same-label",
        ),
        pytest.param(
            [
                _entry("msg-a", "ab", False),
                _entry("msg-a", "pb", False),
                UNRELATED_POSITIVE,
            ],
            [
                _entry("msg-a", "ab", False),
                _entry("msg-a", "pb", False),
                UNRELATED_POSITIVE,
            ],
            id="negative-for-several-labels",
        ),
        pytest.param(
            [
                _entry("msg-a", "ab", True),
                _entry("msg-a", "ab", True),
                _entry("msg-a", "pb", False),
                UNRELATED_POSITIVE,
            ],
            [
                _entry("msg-a", "ab", True),
                _entry("msg-a", "ab", True),
                _entry("msg-a", "pb", False),
                UNRELATED_POSITIVE,
            ],
            id="same-label-repeated-with-history-negative",
        ),
    ],
)
def test_remove_conflicting_entries(data: list[TrainingEntry[str]], expected: list[TrainingEntry[str]]) -> None:
    assert remove_conflicting_entries(data) == expected
