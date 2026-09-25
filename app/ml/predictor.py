#  Copyright 2025 EPAM Systems
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

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union, override

from app.commons import logging
from app.commons.model.db import Hit
from app.commons.model.launch_objects import RelevantItem
from app.commons.model.ml import ModelType
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.query_builder import extract_log_matches
from app.ml.boosting_featurizer import BoostingFeaturizer
from app.ml.models import BoostingDecisionMaker, DefectTypeModel
from app.ml.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from app.utils.text_processing import calculate_text_similarity

LOGGER = logging.getLogger("analyzerApp.predictor")


@dataclass
class FeatureInfo:
    """Container for data about features.

    Attributes:
        feature_ids: List of feature IDs used in the featurizer
        feature_data: Feature vector for the identity gathered from the featurizer
    """

    feature_ids: list[int]
    feature_data: list[float]


@dataclass
class PredictionResult:
    """Result container for prediction workflows.

    Attributes:
        label: Binary prediction label from the decision maker
        probability: Prediction probability from the decision maker
        data: Found Test Item and its metadata for the result
        identity: Identity for the gathered features, the found Test Item ID
        feature_info: Data about features if any
        model_info_tags: List of model information tags
        original_position: Original position of the found Test Item in the search results
    """

    label: int
    probability: list[float]
    data: RelevantItem
    identity: str
    feature_info: Optional[FeatureInfo]
    model_info_tags: list[str]
    original_position: int


class Predictor(metaclass=ABCMeta):
    """Abstract base class for prediction workflows"""

    def __init__(self, **_) -> None:
        """Initialize the predictor."""
        pass

    @abstractmethod
    def predict(
        self,
        search_results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]],
    ) -> list[PredictionResult]:
        """Execute the full prediction workflow.

        :param search_results: Request Test Item and Test Items found for it in OpenSearch
        :return: List of PredictionResult objects, one for each prediction
        """
        ...


class MlPredictor(Predictor, metaclass=ABCMeta):
    """Abstract base class for prediction workflows using BoostingDecisionMaker and BoostingFeaturizer.

    This class encapsulates the common pattern used in both auto analysis and suggestion prediction:
    1. Acquire models from model chooser
    2. Create and configure featurizer
    3. Extract features and find most relevant items
    4. Make predictions with boosting decision maker
    5. Return prediction results
    """

    boosting_config: dict[str, Any]
    boosting_decision_maker: BoostingDecisionMaker
    defect_type_model: DefectTypeModel

    def __init__(
        self,
        *,
        model_chooser: ModelChooser,
        project_id: int,
        boosting_config: dict[str, Any],
        custom_model_prob: float = 0.0,
        hash_source: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize the predictor with required dependencies.

        :param ModelChooser model_chooser: Service for choosing appropriate ML models
        :param int project_id: Project identifier for model selection
        :param dict[str, Any] boosting_config: Configuration for the boosting featurizer
        :param float custom_model_prob: Probability to use custom model instead of global
        :param Optional[Union[int, str]] hash_source: Source for hash-based model selection
        """
        super().__init__()
        self.boosting_config = boosting_config

        # Acquire models
        self.boosting_decision_maker = model_chooser.choose_model(  # type: ignore[assignment]
            project_id,
            self.model_type,
            custom_model_prob=custom_model_prob,
            hash_source=hash_source,
        )
        use_custom_defect_model = 1.0 if self.boosting_decision_maker.is_custom else 0.0
        self.defect_type_model = model_chooser.choose_model(  # type: ignore[assignment]
            project_id, ModelType.defect_type, custom_model_prob=use_custom_defect_model
        )

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Return the type of model used by this predictor."""
        ...

    @abstractmethod
    def create_featurizer(
        self, search_results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]
    ) -> BoostingFeaturizer:
        """Create the appropriate featurizer for this prediction type.

        :param search_results: Request Test Item and Test Items found for it in OpenSearch
        :return: Configured featurizer instance
        """
        ...

    @override
    def predict(
        self,
        search_results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]],
    ) -> list[PredictionResult]:
        """Execute the full prediction workflow.

        :param search_results: Request Test Item and Test Items found for it in OpenSearch
        :return: List of PredictionResult objects, one for each prediction
        """
        # Create and configure featurizer
        featurizer = self.create_featurizer(search_results)

        # Extract features, one row per found Test Item
        feature_data, identifiers = featurizer.gather_features_info()

        # Get model info tags
        model_info_tags = featurizer.get_used_model_info() + self.boosting_decision_maker.get_model_info()

        # If no feature data, return empty result
        if feature_data:
            LOGGER.debug(f"Feature data extracted for {len(feature_data)} items.")
        else:
            LOGGER.debug("No feature data extracted, skipping prediction.")
            return []

        # Make predictions
        predicted_labels, predicted_labels_probability = self.boosting_decision_maker.predict(feature_data)

        if not predicted_labels or not predicted_labels_probability:
            LOGGER.debug("No predictions made, skipping result generation.")
            return []

        relevant_items = featurizer.get_relevant_items()

        # Create list of PredictionResult objects, one for each found Test Item
        results = []
        for idx, identity in enumerate(identifiers):
            relevant_item = relevant_items[identity]
            result = PredictionResult(
                label=predicted_labels[idx],
                probability=predicted_labels_probability[idx],
                data=relevant_item,
                identity=identity,
                feature_info=FeatureInfo(
                    feature_ids=self.boosting_decision_maker.feature_ids, feature_data=feature_data[idx]
                ),
                model_info_tags=model_info_tags,
                original_position=(relevant_item.original_position if relevant_item.original_position >= 0 else idx),
            )
            results.append(result)

        return results


class AutoAnalysisPredictor(MlPredictor):
    """Concrete predictor implementation for auto analysis workflow.

    Uses BoostingFeaturizer for feature extraction and prediction.
    """

    def __init__(
        self,
        model_chooser: ModelChooser,
        project_id: int,
        boosting_config: dict[str, Any],
        custom_model_prob: float = 0.0,
        hash_source: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize auto analysis predictor.

        :param ModelChooser model_chooser: Service for choosing appropriate ML models
        :param int project_id: Project identifier for model selection
        :param dict[str, Any] boosting_config: Configuration for the boosting featurizer
        :param float custom_model_prob: Probability to use custom model instead of global
        :param Optional[Union[int, str]] hash_source: Source for hash-based model selection
        """
        super().__init__(
            model_chooser=model_chooser,
            project_id=project_id,
            boosting_config=boosting_config,
            custom_model_prob=custom_model_prob,
            hash_source=hash_source,
        )

    @property
    @override
    def model_type(self) -> ModelType:
        """Return the type of model used by this predictor."""
        return ModelType.auto_analysis

    @override
    def create_featurizer(
        self, search_results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]
    ) -> BoostingFeaturizer:
        """Create a BoostingFeaturizer for auto analysis.

        :param search_results: Request Test Item and Test Items found for it in OpenSearch
        :return: Configured BoostingFeaturizer instance
        """
        featurizer = BoostingFeaturizer(
            search_results, self.boosting_config, feature_ids=self.boosting_decision_maker.feature_ids
        )
        featurizer.set_defect_type_model(self.defect_type_model)
        return featurizer


class SuggestionPredictor(MlPredictor):
    """Concrete predictor implementation for suggestion workflow.

    Uses SuggestBoostingFeaturizer for feature extraction and prediction.
    """

    def __init__(
        self,
        model_chooser: ModelChooser,
        project_id: int,
        boosting_config: dict[str, Any],
        custom_model_prob: float = 0.0,
        hash_source: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize suggestion predictor.

        :param ModelChooser model_chooser: Service for choosing appropriate ML models
        :param int project_id: Project identifier for model selection
        :param dict[str, Any] boosting_config: Configuration for the boosting featurizer
        :param float custom_model_prob: Probability to use custom model instead of global
        :param Optional[Union[int, str]] hash_source: Source for hash-based model selection
        """
        super().__init__(
            model_chooser=model_chooser,
            project_id=project_id,
            boosting_config=boosting_config,
            custom_model_prob=custom_model_prob,
            hash_source=hash_source,
        )

    @property
    @override
    def model_type(self) -> ModelType:
        """Return the type of model used by this predictor."""
        return ModelType.suggestion

    @override
    def create_featurizer(
        self, search_results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]
    ) -> BoostingFeaturizer:
        """Create a SuggestBoostingFeaturizer for suggestions.

        :param search_results: Request Test Item and Test Items found for it in OpenSearch
        :return: Configured SuggestBoostingFeaturizer instance
        """
        featurizer = SuggestBoostingFeaturizer(
            search_results,
            self.boosting_config,
            feature_ids=self.boosting_decision_maker.feature_ids,
        )
        featurizer.set_defect_type_model(self.defect_type_model)
        return featurizer


def extract_text_fields_for_comparison(logs: list[LogData]) -> str:
    """Combine log messages into one text for similarity comparison.

    :param logs: Logs to combine
    :return: Combined text
    """
    return " ".join([message.strip() for message in (log.message or "" for log in logs) if message.strip()])


def extract_matched_logs(hit: Hit[TestItemIndexData]) -> list[LogData]:
    """Get the best matched found log for every request log matched in the found Test Item.

    :param hit: Found Test Item hit
    :return: Found logs in the order of request logs they matched
    """
    return [log_hits[0].source for log_hits in extract_log_matches(hit).values()]


class SimilarityPredictor(Predictor):
    """Predictor implementation using text similarity calculation.

    Uses `calculate_text_similarity` function to compute similarity scores
    between the request Test Item logs and matched logs of found Test Items.
    """

    similarity_threshold: float

    def __init__(self, **kwargs) -> None:
        """Initialize similarity predictor.

        :param similarity_threshold: Threshold for binary classification (default 0.5)
        Other parameters are accepted for compatibility but not used.
        """
        super().__init__()
        self.similarity_threshold = kwargs.get("similarity_threshold", 0.5)

    @override
    def predict(
        self,
        search_results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]],
    ) -> list[PredictionResult]:
        """Execute similarity-based prediction workflow.

        :param search_results: Request Test Item and Test Items found for it in OpenSearch
        :return: List of PredictionResult objects, one for each found Test Item with matched logs
        """
        request_item, hits = search_results
        query_text = extract_text_fields_for_comparison(request_item.logs or [])
        if not query_text:
            return []

        valid_hits: list[tuple[int, Hit[TestItemIndexData]]] = []
        hit_texts: list[str] = []
        seen_test_items: set[str] = set()
        for position, hit in enumerate(hits):
            test_item_id = str(hit.source.test_item_id)
            if test_item_id in seen_test_items:
                continue
            hit_text = extract_text_fields_for_comparison(extract_matched_logs(hit))
            if not hit_text:
                continue
            seen_test_items.add(test_item_id)
            valid_hits.append((position, hit))
            hit_texts.append(hit_text)

        if not valid_hits:
            return []

        similarity_scores = calculate_text_similarity(query_text, hit_texts)
        results = []
        for (position, hit), sim_result in zip(valid_hits, similarity_scores):
            similarity = sim_result.similarity
            results.append(
                PredictionResult(
                    label=1 if similarity >= self.similarity_threshold else 0,
                    # Probability format: [1-similarity, similarity] to match other predictors
                    probability=[1.0 - similarity, similarity],
                    data=RelevantItem(mrHit=hit, compared_item=request_item, original_position=position),
                    identity=str(hit.source.test_item_id),
                    feature_info=FeatureInfo(feature_ids=[0], feature_data=[similarity]),
                    model_info_tags=["similarity_predictor"],
                    original_position=position,
                )
            )
        return results


PREDICTION_CLASSES: dict[str, type[Predictor]] = {
    ModelType.auto_analysis.name: AutoAnalysisPredictor,
    ModelType.suggestion.name: SuggestionPredictor,
    "similarity": SimilarityPredictor,
}
