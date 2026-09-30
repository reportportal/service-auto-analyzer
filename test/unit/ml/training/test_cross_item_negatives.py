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

from app.commons.model.test_item_index import LogData, TestItemHistoryData, TestItemIndexData
from app.ml.training import TrainingEntry
from app.ml.training.train_analysis_model import (
    HIT_LOG_FIELDS,
    SYNTHETIC_HIT_SCORE,
    build_negative_hits,
    select_negative_items,
    to_hit_source,
)


def _item(test_item_id: str, issue_type: str) -> TestItemIndexData:
    return TestItemIndexData(
        test_item_id=test_item_id,
        test_item_name=f"test {test_item_id}",
        launch_id="1",
        launch_number="7",
        issue_type=issue_type,
        log_count=1,
        logs=[
            LogData(
                log_id=f"{test_item_id}1",
                log_order=0,
                log_level=40000,
                message="error",
                message_for_clustering="error for clustering",
                original_message="ERROR error",
            )
        ],
        issue_history=[TestItemHistoryData(is_auto_analyzed=False, issue_type=issue_type, timestamp="2026-01-01")],
    )


def _entries(*items: tuple[str, str, int]) -> list[TrainingEntry[TestItemIndexData]]:
    return [
        TrainingEntry(
            data=_item(test_item_id, issue_type), project_id=project_id, issue_type=issue_type[:2], is_positive=True
        )
        for test_item_id, issue_type, project_id in items
    ]


def test_to_hit_source_keeps_only_found_item_fields() -> None:
    source = to_hit_source(_item("1", "pb001"))
    assert source.launch_number == "7"
    assert source.log_count == 1
    assert source.issue_history
    assert source.launch_start_time is None
    log = source.logs[0]
    assert log.message == "error"
    assert "message_for_clustering" not in HIT_LOG_FIELDS
    assert log.message_for_clustering is None
    assert log.original_message is None


ENTRIES = _entries(
    ("1", "pb001", 1),
    ("2", "ab001", 1),
    ("3", "si001", 1),
    ("4", "nd001", 1),
    ("5", "pb001", 1),
    ("6", "ab_custom", 2),
)


@pytest.mark.parametrize(
    ("filter_no_defect", "test_item_id", "expected_types"),
    [
        pytest.param(True, "1", {"ab001", "si001"}, id="other-types-of-project"),
        pytest.param(False, "1", {"ab001", "si001", "nd001"}, id="no-defect-kept"),
        pytest.param(True, "2", {"pb001", "si001"}, id="another-positive-type"),
        pytest.param(True, "6", set(), id="single-item-project"),
    ],
)
def test_select_negative_items(filter_no_defect: bool, test_item_id: str, expected_types: set[str]) -> None:
    negative_items = select_negative_items(ENTRIES, filter_no_defect)
    assert set(negative_items) == {entry.data.test_item_id for entry in ENTRIES}
    items = negative_items[test_item_id]
    issue_types = [item.issue_type for item in items]
    assert set(issue_types) == expected_types
    assert len(issue_types) == len(expected_types)
    for item in items:
        assert item.test_item_id != test_item_id
        assert item.logs[0].original_message is None


def test_select_negative_items_is_deterministic() -> None:
    first = select_negative_items(ENTRIES, True)
    second = select_negative_items(ENTRIES, True)
    assert {k: [i.test_item_id for i in v] for k, v in first.items()} == {
        k: [i.test_item_id for i in v] for k, v in second.items()
    }


NEGATIVES = [_item("2", "ab001"), _item("3", "si001"), _item("7", "ti001")]


@pytest.mark.parametrize(
    ("found_types", "positive_type", "expected_ids"),
    [
        pytest.param({"pb001"}, "pb001", ["2", "3"], id="two-negatives-lacking"),
        pytest.param({"pb001", "ab001"}, "pb001", ["3"], id="found-type-skipped"),
        pytest.param({"pb001", "ab001", "si001"}, "pb001", [], id="enough-negatives"),
        pytest.param({"ab001"}, "pb001", [], id="positive-not-found"),
    ],
)
def test_build_negative_hits(found_types: set[str], positive_type: str, expected_ids: list[str]) -> None:
    hits = build_negative_hits(found_types, positive_type, NEGATIVES)
    assert [hit.source.test_item_id for hit in hits] == expected_ids
    assert all(hit.score == SYNTHETIC_HIT_SCORE for hit in hits)
