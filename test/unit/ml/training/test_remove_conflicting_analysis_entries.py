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

from typing import Optional

import pytest

from app.commons.model.test_item_index import LogData, TestItemHistoryData, TestItemIndexData
from app.ml.training import TrainingEntry
from app.ml.training.train_analysis_model import remove_conflicting_entries

STACKTRACE = "at com.example.ApiTest.checkStatus(ApiTest.java)"


def _entry(
    test_item_id: str,
    history: list[str],
    message: str = "expected HTTP code 400, but was 503",
    test_case_hash: Optional[int] = 1,
    project_id: int = 1,
    stacktrace: str = STACKTRACE,
) -> TrainingEntry[TestItemIndexData]:
    test_item = TestItemIndexData(
        test_item_id=test_item_id,
        test_item_name="checkStatus",
        test_case_hash=test_case_hash,
        launch_id="1",
        issue_type=history[-1],
        logs=[
            LogData(
                log_id=f"{test_item_id}1",
                log_order=0,
                log_level=40000,
                detected_message_with_numbers=message,
                stacktrace=stacktrace,
            )
        ],
        issue_history=[
            TestItemHistoryData(is_auto_analyzed=False, issue_type=issue_type, timestamp="2026-01-01")
            for issue_type in history
        ],
    )
    return TrainingEntry(data=test_item, project_id=project_id, issue_type=history[-1][:2], is_positive=True)


UNRELATED = _entry("100", ["si001"], message="Connection refused")


@pytest.mark.parametrize(
    ("data", "expected_ids", "expected_removed"),
    [
        pytest.param([], [], {}, id="empty-data"),
        pytest.param(
            [_entry("1", ["pb001"]), UNRELATED, _entry("2", ["ab001"]), _entry("3", ["pb001"])],
            ["100"],
            {1: {"1", "2", "3"}},
            id="same-test-and-logs-different-issue-types",
        ),
        pytest.param(
            [_entry("1", ["pb001"]), _entry("2", ["pb001"]), UNRELATED],
            ["1", "2", "100"],
            {},
            id="same-issue-type-is-not-a-conflict",
        ),
        pytest.param(
            [_entry("1", ["pb001"]), _entry("2", ["ab001"], message="expected HTTP code 401, but was 503")],
            ["1", "2"],
            {},
            id="different-numbers-are-different-failures",
        ),
        pytest.param(
            [
                _entry("1", ["pb001"]),
                _entry("2", ["ab001"], stacktrace="at com.example.AuthTest.login(AuthTest.java)"),
            ],
            ["1", "2"],
            {},
            id="different-stacktraces-are-different-failures",
        ),
        pytest.param(
            [_entry("1", ["pb001"]), _entry("2", ["ab001"], test_case_hash=2)],
            ["1", "2"],
            {},
            id="different-tests-are-not-a-conflict",
        ),
        pytest.param(
            [_entry("1", ["pb001"]), _entry("2", ["ab001"], project_id=2)],
            ["1", "2"],
            {},
            id="different-projects-are-not-a-conflict",
        ),
        pytest.param(
            [_entry("1", ["ab001", "pb001"]), _entry("2", ["ab001"]), UNRELATED],
            ["100"],
            {1: {"1", "2"}},
            id="history-negative-is-positive-for-another-item",
        ),
        pytest.param(
            [_entry("1", ["ab001", "pb001"]), _entry("2", ["ab001", "pb001"])],
            ["1", "2"],
            {},
            id="same-history-is-not-a-conflict",
        ),
    ],
)
def test_remove_conflicting_entries(
    data: list[TrainingEntry[TestItemIndexData]],
    expected_ids: list[str],
    expected_removed: dict[int, set[str]],
) -> None:
    result, removed = remove_conflicting_entries(data)
    assert [entry.data.test_item_id for entry in result] == expected_ids
    assert dict(removed) == expected_removed
