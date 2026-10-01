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

import math
from typing import Any, Optional, Protocol

from app.commons import logging
from app.commons.model.db import Hit
from app.commons.model.launch_objects import RelevantItem
from app.commons.model.test_item_index import TestItemIndexData
from app.commons.similarity_calculator import SimilarityCalculator
from app.ml.features import (
    DEFAULT_TIME_WEIGHT_DECAY,
    BaseIssueTypeFeature,
    DefectTypeFeature,
    HistoryStabilityFeature,
    HistoryUnchangedFeature,
    IdentifierJaccardFeature,
    IssueTypeConsensusFeature,
    IssueTypeGroupEqualityFeature,
    IssueTypeShareFeature,
    ItemFieldEqualityFeature,
    ItemFieldSimilarityFeature,
    LaunchNumberDistanceFeature,
    LogCountFeature,
    LogCoverageFeature,
    LogFieldEqualityFeature,
    LogFieldSimilarityFeature,
    ManuallyAnalyzedFeature,
    RequestFieldPresentFeature,
    RequestIdentifiersPresentFeature,
    SeveralLogsFeature,
    StacktraceFeature,
    TimeDecayFeature,
    ValuesSimilarityFeature,
)
from app.ml.models.defect_type_model import DefectTypeModel
from app.utils import text_processing, utils
from app.utils.utils import normalize_issue_type

logger = logging.getLogger("analyzerApp.boosting_featurizer")


class Feature(Protocol):
    """Gradient Boosting feature, calculated for every found Test Item against the request Test Item."""

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        """Calculate the feature value for every found Test Item.

        :param request: Request Test Item
        :param hits: Found Test Items
        :return: Feature values in the range [0.0, 1.0], one per hit, in the order of hits
        """
        ...


class BoostingFeaturizer:
    """Gather Gradient Boosting features by issue types.

    The most relevant found Test Item of every issue type, the top one in search results, represents the issue type,
    so features are gathered into one row per issue type. Features are calculated over all found Test Items, so
    features of issue type groups see every Test Item of a group.
    """

    config: dict[str, Any]
    feature_ids: list[int]
    defect_type_predict_model: Optional[DefectTypeModel]
    similarity_calculator: SimilarityCalculator
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
        defect_type_model: Optional[DefectTypeModel] = None,
        **_: Any,
    ) -> None:
        self.config = config
        if isinstance(feature_ids, str):
            self.feature_ids = text_processing.transform_string_feature_range_into_list(feature_ids)
        else:
            self.feature_ids = feature_ids
        self.defect_type_predict_model = defect_type_model
        self.similarity_calculator = SimilarityCalculator()
        self.used_model_info = set(defect_type_model.get_model_info()) if defect_type_model else set()
        self._features = None
        self._relevant_items = None
        self.raw_results = results
        self.all_results = self.normalize_results(results)

    @staticmethod
    def normalize_results(
        results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]],
    ) -> tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]:
        """Copy unique found Test Items and fill in their scores normalized by the maximum score.

        :param results: Request Test Item and found Test Items
        :return: Request Test Item and copies of found Test Items with normalized scores, one per Test Item
        """
        request, hits = results
        seen_test_items: set[str] = set()
        normalized_hits = []
        for hit in hits:
            test_item_id = str(hit.source.test_item_id)
            if test_item_id in seen_test_items:
                continue
            seen_test_items.add(test_item_id)
            normalized_hits.append(hit.model_copy(deep=True))
        max_score = max((hit.score or 0.0 for hit in normalized_hits), default=0.0)
        for hit in normalized_hits:
            hit.normalized_score = (hit.score or 0.0) / (max_score if max_score > 0.0 else 1.0)
        return request, normalized_hits

    def get_used_model_info(self) -> list[str]:
        return list(self.used_model_info)

    @classmethod
    def get_supported_feature_ids(cls) -> list[int]:
        """Get IDs of features supported by the featurizer.

        :return: Sorted feature IDs
        """
        return sorted(cls.create_features({}, None, SimilarityCalculator()).keys())

    def _create_features(self) -> dict[int, Feature]:
        return self.create_features(self.config, self.defect_type_predict_model, self.similarity_calculator)

    @staticmethod
    def create_features(
        config: dict[str, Any],
        defect_type_model: Optional[DefectTypeModel],
        calculator: SimilarityCalculator,
    ) -> dict[int, Feature]:
        """Create features supported by the featurizer.

        Features depend on the found Test Items only, never on their order in search results: the order is decided by
        a search engine, and training data can not reproduce it for negative cases.

        :param config: Featurizer configuration
        :param defect_type_model: Defect Type model for the Defect Type feature
        :param calculator: Text similarity calculator
        :return: Feature ID mapped to the feature
        """
        return {
            1: DefectTypeFeature(defect_type_model),
            2: IssueTypeShareFeature(),
            3: LogFieldSimilarityFeature("message", calculator),
            4: LogFieldSimilarityFeature("detected_message", calculator),
            5: LogFieldSimilarityFeature("detected_message_with_numbers", calculator),
            6: LogFieldSimilarityFeature("stacktrace", calculator),
            7: LogFieldSimilarityFeature("only_numbers", calculator),
            8: SeveralLogsFeature(of_request=False),
            9: SeveralLogsFeature(of_request=True),
            10: ValuesSimilarityFeature("only_numbers"),
            11: LogFieldSimilarityFeature("message_params", calculator),
            12: LogFieldSimilarityFeature("found_exceptions", calculator),
            13: ManuallyAnalyzedFeature(),
            14: LogFieldSimilarityFeature("detected_message_extended", calculator),
            15: LogFieldSimilarityFeature("detected_message_without_params_extended", calculator),
            16: LogFieldSimilarityFeature("stacktrace_extended", calculator),
            17: LogFieldSimilarityFeature("message_without_params_extended", calculator),
            18: LogFieldSimilarityFeature("message_extended", calculator),
            19: ItemFieldEqualityFeature("test_case_hash"),
            20: IssueTypeGroupEqualityFeature("test_case_hash"),
            21: BaseIssueTypeFeature("ab"),
            22: BaseIssueTypeFeature("pb"),
            23: BaseIssueTypeFeature("si"),
            24: LogFieldEqualityFeature("urls"),
            25: LogFieldSimilarityFeature("detected_message_without_params_and_brackets", calculator),
            26: LogFieldEqualityFeature("potential_status_codes"),
            27: ItemFieldEqualityFeature("launch_name", lowercase=True),
            28: ItemFieldEqualityFeature("launch_id"),
            29: LogFieldSimilarityFeature("found_tests_and_methods", calculator),
            30: ItemFieldSimilarityFeature("test_item_name", calculator),
            31: TimeDecayFeature(config.get("time_weight_decay", DEFAULT_TIME_WEIGHT_DECAY)),
            32: LogCountFeature(),
            33: LaunchNumberDistanceFeature(),
            34: HistoryStabilityFeature(),
            35: StacktraceFeature(),
            36: HistoryUnchangedFeature(),
            37: IssueTypeConsensusFeature(),
            38: IdentifierJaccardFeature("message"),
            39: RequestIdentifiersPresentFeature("message"),
            40: RequestFieldPresentFeature("potential_status_codes"),
            41: LogCoverageFeature("message", reverse=False),
            42: LogCoverageFeature("message", reverse=True),
        }

    def _get_features(self) -> dict[int, Feature]:
        if self._features is None:
            self._features = self._create_features()
        return self._features

    def _get_identity(self, hit: Hit[TestItemIndexData]) -> str:
        """Get the identity a found Test Item represents in the output.

        :param hit: Found Test Item
        :return: Issue type of the found Test Item
        """
        return normalize_issue_type(hit.source.issue_type)

    def get_relevant_items(self) -> dict[str, RelevantItem]:
        """Get the most relevant found Test Item for every identity, with its comparison metadata.

        :return: Identity mapped to its relevant item, in the order of search results
        """
        if self._relevant_items is not None:
            return self._relevant_items
        request, hits = self.all_results
        relevant_items: dict[str, RelevantItem] = {}
        for position, hit in enumerate(hits):
            identity = self._get_identity(hit)
            if identity in relevant_items:
                continue
            relevant_items[identity] = RelevantItem(
                mrHit=hit, score=hit.normalized_score, compared_item=request, original_position=position
            )
        self._relevant_items = relevant_items
        return relevant_items

    @staticmethod
    def _validate_values(feature_id: int, values: list[float], hits_number: int) -> None:
        if len(values) != hits_number:
            raise ValueError(f"Feature {feature_id} returned {len(values)} values for {hits_number} found Test Items")
        for value in values:
            if not isinstance(value, float) or math.isnan(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Feature {feature_id} returned value {value!r} out of the range [0.0, 1.0]")

    def gather_features_info(self) -> tuple[list[list[float]], list[str]]:
        """Gather features for the most relevant found Test Item of every identity.

        Features which are not supported by the featurizer are filled with zeros.

        :return: Feature rows, one per identity, and identities aligned with the rows
        """
        relevant_items = self.get_relevant_items()
        if not relevant_items:
            return [], []
        identities = list(relevant_items.keys())
        positions = [relevant_item.original_position for relevant_item in relevant_items.values()]
        request, hits = self.all_results
        features = self._get_features()

        gathered_data_dict: dict[int, list[list[float]]] = {}
        for feature_id in self.feature_ids:
            feature = features.get(feature_id)
            if feature is None:
                gathered_data_dict[feature_id] = [[0.0] for _ in positions]
                continue
            values = feature.calculate(request, hits)
            self._validate_values(feature_id, values, len(hits))
            gathered_data_dict[feature_id] = [[values[position]] for position in positions]
        gathered_data = utils.gather_feature_list(gathered_data_dict, self.feature_ids)
        return gathered_data, identities
