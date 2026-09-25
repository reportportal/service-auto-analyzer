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

from typing import Any, Optional, Protocol

from app.commons import logging
from app.commons.model.db import Hit
from app.commons.model.launch_objects import RelevantItem
from app.commons.model.test_item_index import TestItemIndexData
from app.ml.models.defect_type_model import DefectTypeModel
from app.utils import text_processing, utils

logger = logging.getLogger("analyzerApp.boosting_featurizer")


class Feature(Protocol):
    """Gradient Boosting feature, calculated for every found Test Item against the request Test Item."""

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        """Calculate the feature value for every found Test Item.

        :param request: Request Test Item
        :param hits: Found Test Items
        :return: Feature values, one per hit, in the order of hits
        """
        ...


class BoostingFeaturizer:
    """Gather Gradient Boosting features for Test Items found by the request Test Item.

    Every found Test Item is a separate candidate, so features are gathered into one row per found Test Item.
    """

    config: dict[str, Any]
    feature_ids: list[int]
    defect_type_predict_model: Optional[DefectTypeModel]
    raw_results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]
    all_results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]
    used_model_info: set[str]
    _features: Optional[dict[int, Feature]]
    _relevant_items: Optional[dict[str, RelevantItem]]

    def __init__(
        self,
        results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]],
        config: dict[str, Any],
        feature_ids: str | list[int],
        **_: Any,
    ) -> None:
        self.config = config
        if isinstance(feature_ids, str):
            self.feature_ids = text_processing.transform_string_feature_range_into_list(feature_ids)
        else:
            self.feature_ids = feature_ids
        self.defect_type_predict_model = None
        self.used_model_info = set()
        self._features = None
        self._relevant_items = None
        self.raw_results = results
        self.all_results = self.normalize_results(results)

    @staticmethod
    def normalize_results(
        results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]],
    ) -> tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]:
        """Copy found Test Items and fill in their scores normalized by the maximum score.

        :param results: Request Test Item and found Test Items
        :return: Request Test Item and copies of found Test Items with normalized scores
        """
        request, hits = results
        normalized_hits = [hit.model_copy(deep=True) for hit in hits]
        max_score = max((hit.score or 0.0 for hit in normalized_hits), default=0.0)
        for hit in normalized_hits:
            hit.normalized_score = (hit.score or 0.0) / (max_score if max_score > 0.0 else 1.0)
        return request, normalized_hits

    def set_defect_type_model(self, defect_type_model: Optional[DefectTypeModel]) -> None:
        self.defect_type_predict_model = defect_type_model

    def get_used_model_info(self) -> list[str]:
        return list(self.used_model_info)

    def _create_features(self) -> dict[int, Feature]:
        """Create features supported by the featurizer.

        Called on the first feature gathering, so features can use the config and the defect type model.

        :return: Feature ID mapped to the feature
        """
        return {}

    def _get_features(self) -> dict[int, Feature]:
        if self._features is None:
            self._features = self._create_features()
        return self._features

    def get_relevant_items(self) -> dict[str, RelevantItem]:
        """Get found Test Items with their comparison metadata.

        :return: Found Test Item ID mapped to its relevant item, in the order of search results
        """
        if self._relevant_items is not None:
            return self._relevant_items
        request, hits = self.all_results
        relevant_items: dict[str, RelevantItem] = {}
        for position, hit in enumerate(hits):
            identity = str(hit.source.test_item_id)
            if identity in relevant_items:
                continue
            relevant_items[identity] = RelevantItem(
                mrHit=hit, score=hit.normalized_score, compared_item=request, original_position=position
            )
        self._relevant_items = relevant_items
        return relevant_items

    def gather_features_info(self) -> tuple[list[list[float]], list[str]]:
        """Gather features for every found Test Item.

        Features which are not supported by the featurizer are filled with zeros.

        :return: Feature rows, one per found Test Item, and found Test Item IDs aligned with the rows
        """
        relevant_items = self.get_relevant_items()
        if not relevant_items:
            return [], []
        identities = list(relevant_items.keys())
        hits = [relevant_item.mrHit for relevant_item in relevant_items.values()]
        request = self.all_results[0]
        features = self._get_features()

        gathered_data_dict: dict[int, list[list[float]]] = {}
        for feature_id in self.feature_ids:
            feature = features.get(feature_id)
            if feature is None:
                gathered_data_dict[feature_id] = [[0.0] for _ in hits]
                continue
            values = feature.calculate(request, hits)
            if len(values) != len(hits):
                raise ValueError(
                    f"Feature {feature_id} returned {len(values)} values for {len(hits)} found Test Items"
                )
            gathered_data_dict[feature_id] = [[value] for value in values]
        gathered_data = utils.gather_feature_list(gathered_data_dict, self.feature_ids)
        return gathered_data, identities
