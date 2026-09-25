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

from unittest.mock import Mock

import pytest

from app.commons.model.db import Hit
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.ml.boosting_featurizer import BoostingFeaturizer, Feature
from app.ml.suggest_boosting_featurizer import SuggestBoostingFeaturizer


class PositionIndexFeature:
    """Returns hit position divided by 10, to check which rows are selected."""

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return [position / 10 for position in range(len(hits))]


class ConstantFeature:
    value: object
    length_delta: int

    def __init__(self, value: object, length_delta: int = 0) -> None:
        self.value = value
        self.length_delta = length_delta

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return [self.value] * (len(hits) + self.length_delta)  # type: ignore[list-item]


def build_request_item() -> TestItemIndexData:
    return TestItemIndexData(
        test_item_id="100",
        launch_id="1",
        logs=[LogData(log_id="1", log_level=40000, detected_message_without_params_extended="error text")],
    )


def build_hit(test_item_id: str, score: float, issue_type: str = "pb001") -> Hit[TestItemIndexData]:
    return Hit[TestItemIndexData].from_dict(
        {
            "_id": test_item_id,
            "_score": score,
            "_source": {"test_item_id": test_item_id, "launch_id": "2", "issue_type": issue_type},
        }
    )


def build_hits() -> list[Hit[TestItemIndexData]]:
    return [
        build_hit("1", 4.0, "pb001"),
        build_hit("2", 3.0, "ab001"),
        build_hit("3", 2.0, "pb001"),
        build_hit("4", 1.0, "si001"),
    ]


def create_featurizer(
    featurizer_class: type[BoostingFeaturizer],
    features: dict[int, Feature],
    feature_ids: list[int],
    hits: list[Hit[TestItemIndexData]],
) -> BoostingFeaturizer:
    featurizer = featurizer_class((build_request_item(), hits), {}, feature_ids)
    featurizer._create_features = Mock(return_value=features)  # type: ignore[method-assign]
    return featurizer


def test_normalized_scores_are_filled_and_duplicates_dropped():
    hits = [build_hit("1", 4.0), build_hit("2", 2.0), build_hit("1", 1.0)]

    featurizer = BoostingFeaturizer((build_request_item(), hits), {}, [1])

    request_item, normalized_hits = featurizer.all_results
    assert request_item is featurizer.raw_results[0]
    assert [hit.source.test_item_id for hit in normalized_hits] == ["1", "2"]
    assert [hit.normalized_score for hit in normalized_hits] == [1.0, 0.5]
    assert [hit.normalized_score for hit in hits] == [0.0, 0.0, 0.0]


def test_auto_analysis_rows_are_top_items_of_issue_types():
    featurizer = create_featurizer(BoostingFeaturizer, {5: PositionIndexFeature()}, [3, 5], build_hits())

    feature_data, identities = featurizer.gather_features_info()

    assert identities == ["pb001", "ab001", "si001"]
    assert feature_data == [[0.0, 0.0], [0.0, 0.1], [0.0, 0.3]]
    relevant_items = featurizer.get_relevant_items()
    assert [item.mrHit.source.test_item_id for item in relevant_items.values()] == ["1", "2", "4"]
    assert [item.original_position for item in relevant_items.values()] == [0, 1, 3]
    featurizer._create_features.assert_called_once()  # type: ignore[attr-defined]


def test_suggestion_rows_are_all_items():
    featurizer = create_featurizer(SuggestBoostingFeaturizer, {5: PositionIndexFeature()}, [5], build_hits())

    feature_data, identities = featurizer.gather_features_info()

    assert identities == ["1", "2", "3", "4"]
    assert feature_data == [[0.0], [0.1], [0.2], [0.3]]


@pytest.mark.parametrize("featurizer_class", [BoostingFeaturizer, SuggestBoostingFeaturizer])
def test_no_hits(featurizer_class):
    featurizer = create_featurizer(featurizer_class, {1: PositionIndexFeature()}, [1], [])

    assert featurizer.gather_features_info() == ([], [])


@pytest.mark.parametrize(
    "feature",
    [ConstantFeature(0.5, length_delta=-1), ConstantFeature(1.5), ConstantFeature(-0.1), ConstantFeature(1)],
)
def test_feature_output_is_validated(feature):
    featurizer = create_featurizer(BoostingFeaturizer, {1: feature}, [1], build_hits())

    with pytest.raises(ValueError):
        featurizer.gather_features_info()


@pytest.mark.parametrize("featurizer_class", [BoostingFeaturizer, SuggestBoostingFeaturizer])
def test_all_registered_features_are_in_range(featurizer_class):
    feature_ids = list(range(1, 47))
    featurizer = featurizer_class((build_request_item(), build_hits()), {"time_weight_decay": 0.95}, feature_ids)

    feature_data, _ = featurizer.gather_features_info()

    assert set(featurizer._create_features().keys()) == set(feature_ids)
    for row in feature_data:
        assert len(row) == len(feature_ids)
        assert all(isinstance(value, float) and 0.0 <= value <= 1.0 for value in row)


def test_defect_type_model_from_constructor():
    defect_type_model = Mock()
    defect_type_model.predict.return_value = ([1], [[0.3, 0.7]])
    defect_type_model.get_model_info.return_value = ["defect_type_model"]
    featurizer = BoostingFeaturizer((build_request_item(), build_hits()), {}, [2], defect_type_model=defect_type_model)

    feature_data, identities = featurizer.gather_features_info()

    assert identities == ["pb001", "ab001", "si001"]
    assert feature_data == [[0.7], [0.7], [0.7]]
    assert featurizer.get_used_model_info() == ["defect_type_model"]
    assert defect_type_model.predict.call_count == 3


def test_no_model_info_without_defect_type_model():
    featurizer = BoostingFeaturizer((build_request_item(), build_hits()), {}, [2])

    feature_data, _ = featurizer.gather_features_info()

    assert feature_data == [[0.0], [0.0], [0.0]]
    assert featurizer.get_used_model_info() == []


def test_unsupported_features_are_zeros():
    featurizer = BoostingFeaturizer((build_request_item(), build_hits()), {}, [0, 1, 100])

    feature_data, _ = featurizer.gather_features_info()

    assert feature_data == [[0.0, 1.0, 0.0], [0.0, 1.0 - 1 / 3, 0.0], [0.0, 0.0, 0.0]]
