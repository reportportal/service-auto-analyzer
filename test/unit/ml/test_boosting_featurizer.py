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


class ScoreFeature:
    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return [hit.normalized_score for hit in hits]


class BrokenFeature:
    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return [1.0]


def build_request_item() -> TestItemIndexData:
    return TestItemIndexData(
        test_item_id="100", launch_id="1", logs=[LogData(log_id="1", log_level=40000, message="error")]
    )


def build_hit(test_item_id: str, score: float) -> Hit[TestItemIndexData]:
    return Hit[TestItemIndexData].from_dict(
        {"_id": test_item_id, "_score": score, "_source": {"test_item_id": test_item_id, "launch_id": "2"}}
    )


def create_featurizer(
    features: dict[int, Feature], feature_ids: list[int], hits: list[Hit[TestItemIndexData]]
) -> BoostingFeaturizer:
    featurizer = BoostingFeaturizer((build_request_item(), hits), {}, feature_ids)
    featurizer._create_features = Mock(return_value=features)  # type: ignore[method-assign]
    return featurizer


def test_normalized_scores_are_filled():
    hits = [build_hit("1", 4.0), build_hit("2", 2.0)]

    featurizer = BoostingFeaturizer((build_request_item(), hits), {}, [0])

    request_item, normalized_hits = featurizer.all_results
    assert request_item is featurizer.raw_results[0]
    assert [hit.normalized_score for hit in normalized_hits] == [1.0, 0.5]
    assert [hit.normalized_score for hit in hits] == [0.0, 0.0]


def test_one_row_per_hit_and_unsupported_features_are_zeros():
    hits = [build_hit("1", 4.0), build_hit("2", 2.0)]
    featurizer = create_featurizer({5: ScoreFeature()}, [3, 5], hits)

    feature_data, identities = featurizer.gather_features_info()

    assert identities == ["1", "2"]
    assert feature_data == [[0.0, 1.0], [0.0, 0.5]]
    featurizer._create_features.assert_called_once()  # type: ignore[attr-defined]


def test_relevant_items_keep_original_positions():
    hits = [build_hit("1", 4.0), build_hit("2", 2.0), build_hit("1", 1.0)]
    featurizer = BoostingFeaturizer((build_request_item(), hits), {}, [0])

    relevant_items = featurizer.get_relevant_items()

    assert list(relevant_items.keys()) == ["1", "2"]
    assert [item.original_position for item in relevant_items.values()] == [0, 1]
    assert [item.score for item in relevant_items.values()] == [1.0, 0.5]
    assert all(item.compared_item is featurizer.all_results[0] for item in relevant_items.values())


def test_no_hits():
    featurizer = create_featurizer({0: ScoreFeature()}, [0], [])

    assert featurizer.gather_features_info() == ([], [])


def test_feature_output_length_is_validated():
    featurizer = create_featurizer({0: BrokenFeature()}, [0], [build_hit("1", 4.0), build_hit("2", 2.0)])

    with pytest.raises(ValueError):
        featurizer.gather_features_info()


@pytest.mark.parametrize("featurizer_class", [BoostingFeaturizer, SuggestBoostingFeaturizer])
def test_feature_registry_is_empty(featurizer_class):
    featurizer = featurizer_class((build_request_item(), [build_hit("1", 4.0)]), {}, [0, 1])

    feature_data, identities = featurizer.gather_features_info()

    assert identities == ["1"]
    assert feature_data == [[0.0, 0.0]]
