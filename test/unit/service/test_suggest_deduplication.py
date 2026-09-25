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

from app.commons.model.db import Hit
from app.commons.model.launch_objects import RelevantItem
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.ml.predictor import PredictionResult
from app.service.suggest_service import deduplicate_results


def build_result(test_item_id: str, issue_type: str, messages: list[str], position: int) -> PredictionResult:
    logs = [
        LogData(
            log_id=f"{test_item_id}{idx}",
            log_order=idx,
            log_level=40000,
            detected_message_with_numbers=message,
            whole_message=message,
        )
        for idx, message in enumerate(messages)
    ]
    source = TestItemIndexData(test_item_id=test_item_id, launch_id="1", issue_type=issue_type, logs=logs)
    hit = Hit[TestItemIndexData].from_dict({"_id": test_item_id, "_score": 1.0, "_source": source.model_dump()})
    return PredictionResult(
        label=1,
        probability=[0.1, 0.9],
        data=RelevantItem(mrHit=hit, compared_item=TestItemIndexData(test_item_id="100", launch_id="1")),
        identity=test_item_id,
        feature_info=None,
        model_info_tags=[],
        original_position=position,
    )


def test_duplicates_of_the_same_issue_type_are_removed():
    results = [
        build_result("1", "pb001", ["database timeout", "connection refused"], 0),
        build_result("2", "pb001", ["database timeout", "connection refused"], 1),
        build_result("3", "ab001", ["database timeout", "connection refused"], 2),
        build_result("4", "pb001", ["assertion failed"], 3),
    ]

    unique_results = deduplicate_results(results)

    assert [result.identity for result in unique_results] == ["1", "3", "4"]


def test_items_with_different_logs_are_kept():
    results = [
        build_result("1", "pb001", ["database timeout"], 0),
        build_result("2", "pb001", ["database timeout", "connection refused"], 1),
    ]

    assert [result.identity for result in deduplicate_results(results)] == ["1", "2"]
