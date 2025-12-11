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
from typing import Any, Optional, Union

from typing_extensions import override

from app.commons import logging
from app.commons.model.ml import ModelType
from app.commons.model_chooser import ModelChooser
from app.machine_learning.boosting_featurizer import BoostingFeaturizer
from app.machine_learning.models import BoostingDecisionMaker, DefectTypeModel
from app.machine_learning.suggest_boosting_featurizer import SuggestBoostingFeaturizer
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
        data: Most relevant log and its metadata for the result
        identity: Identity for the gathered features
        feature_info: Data about features if any
        model_info_tags: List of model information tags
        original_position: Original position of the log in the search results
    """

    label: int
    probability: list[float]
    data: dict[str, Any]
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
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[PredictionResult]:
        """Execute the full prediction workflow.

        :param list[tuple[dict[str, Any], dict[str, Any]]] search_results: List of (log_info, search_results) tuples
                                                                           from OpenSearch
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
    def create_featurizer(self, search_results: list[tuple[dict[str, Any], dict[str, Any]]]) -> BoostingFeaturizer:
        """Create the appropriate featurizer for this prediction type.

        :param list[tuple[dict[str, Any], dict[str, Any]]] search_results: List of (log_info, search_results) tuples
        :return: Configured featurizer instance
        """
        ...

    @override
    def predict(
        self,
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[PredictionResult]:
        """Execute the full prediction workflow.

        :param list[tuple[dict[str, Any], dict[str, Any]]] search_results: List of (log_info, search_results) tuples
                                                                           from OpenSearch
        :return: List of PredictionResult objects, one for each prediction
        """
        # Create and configure featurizer
        featurizer = self.create_featurizer(search_results)

        # Extract features and find most relevant items
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

        # Get scores by identity
        scores_by_identity = featurizer.find_most_relevant_by_type()

        # Create list of PredictionResult objects, one for each prediction
        results = []
        for idx, identity in enumerate(identifiers):
            result = PredictionResult(
                label=predicted_labels[idx],
                probability=predicted_labels_probability[idx],
                data=scores_by_identity[identity],
                identity=identity,
                feature_info=FeatureInfo(
                    feature_ids=self.boosting_decision_maker.feature_ids, feature_data=feature_data[idx]
                ),
                model_info_tags=model_info_tags,
                original_position=scores_by_identity[identity].get("original_position", idx),
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
    def create_featurizer(self, search_results: list[tuple[dict[str, Any], dict[str, Any]]]) -> BoostingFeaturizer:
        """Create a BoostingFeaturizer for auto analysis.

        :param list[tuple[dict[str, Any], dict[str, Any]]] search_results: List of (log_info, search_results) tuples
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
    def create_featurizer(self, search_results: list[tuple[dict[str, Any], dict[str, Any]]]) -> BoostingFeaturizer:
        """Create a SuggestBoostingFeaturizer for suggestions.

        :param list[tuple[dict[str, Any], dict[str, Any]]] search_results: List of (log_info, search_results) tuples
        :return: Configured SuggestBoostingFeaturizer instance
        """
        featurizer = SuggestBoostingFeaturizer(
            search_results,
            self.boosting_config,
            feature_ids=self.boosting_decision_maker.feature_ids,
        )
        featurizer.set_defect_type_model(self.defect_type_model)
        return featurizer


def extract_text_fields_for_comparison(search_request: dict[str, Any]) -> str:
    query_message = search_request.get("_source", {}).get("message", "") or ""
    query_merged_logs = search_request.get("_source", {}).get("merged_small_logs", "") or ""
    # Combine query text fields
    query_text = " ".join([text.strip() for text in [query_message, query_merged_logs] if text.strip()])
    return query_text


class SimilarityPredictor(Predictor):
    """Predictor implementation using text similarity calculation.

    Uses calculate_text_similarity function to compute similarity scores
    between the current test item logs and candidate logs from search results.
    """

    similarity_threshold: float

    def __init__(self, **kwargs) -> None:
        """Initialize similarity predictor.

        :param similarity_threshold: Threshold for binary classification (default 0.5)
        Other parameters are accepted for compatibility but not used.
        """
        super().__init__()
        self.similarity_threshold = kwargs.get("similarity_threshold", 0.5)

    def __do_prediction_for_request(
        self, query_text: str, search_request: dict[str, Any], valid_hits: list[Any], hit_texts: list[str]
    ) -> list[PredictionResult]:
        # Calculate similarities for all hits at once
        similarity_scores = calculate_text_similarity(query_text, *hit_texts)

        # Group results by test_item to find most relevant for each
        results_by_test_item = {}

        for idx, (hit, sim_result) in enumerate(zip(valid_hits, similarity_scores)):
            # Get test_item identifier
            test_item = str(hit.get("_source", {}).get("test_item", "unknown"))

            # Track the best hit for each test_item
            if (
                test_item not in results_by_test_item
                or sim_result.similarity > results_by_test_item[test_item]["similarity"]
            ):
                results_by_test_item[test_item] = {
                    "similarity": sim_result.similarity,
                    "mrHit": hit,
                    "compared_log": search_request,
                    "original_position": idx,  # Add position info to hit
                }

        results = []
        # Create PredictionResult objects for each test_item
        for test_item, result_data in results_by_test_item.items():
            similarity = result_data["similarity"]

            # Binary classification based on threshold
            label = 1 if similarity >= self.similarity_threshold else 0

            # Probability format: [1-similarity, similarity] to match other predictors
            probability = [1.0 - similarity, similarity]

            # Create data structure matching expected format
            data = {"mrHit": result_data["mrHit"], "compared_log": result_data["compared_log"]}

            # Create PredictionResult
            prediction_result = PredictionResult(
                label=label,
                probability=probability,
                data=data,
                identity=test_item,
                feature_info=FeatureInfo(feature_ids=[0], feature_data=[similarity]),
                model_info_tags=["similarity_predictor"],
                original_position=result_data["original_position"],
            )
            results.append(prediction_result)
        return results

    @override
    def predict(
        self,
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[PredictionResult]:
        """Execute similarity-based prediction workflow.

        :param list[tuple[dict[str, Any], dict[str, Any]]] search_results: List of (search_request, search_results)
                                                                           tuples
        :return: List of PredictionResult objects, one for each prediction
        """
        if not search_results:
            return []

        results = []

        for search_request, search_result in search_results:
            # Get all candidate logs from search hits
            hits = search_result.get("hits", {}).get("hits", [])

            if not hits:
                continue

            # Extract and combine message and merged_small_logs from query log
            query_text = extract_text_fields_for_comparison(search_request)

            # If query text is empty, skip this search request
            if not query_text.strip():
                continue

            # Collect valid hits and their texts for batch processing
            valid_hits = []
            hit_texts = []

            for idx, hit in enumerate(hits):
                # Extract and combine message and merged_small_logs from hit
                hit_text = extract_text_fields_for_comparison(hit)

                # If hit text is empty, skip this hit
                if not hit_text.strip():
                    continue

                valid_hits.append(hit)
                hit_texts.append(hit_text)

            if not valid_hits:
                continue

            results.extend(self.__do_prediction_for_request(query_text, search_request, valid_hits, hit_texts))

        return results


PREDICTION_CLASSES: dict[str, type[Predictor]] = {
    ModelType.auto_analysis.name: AutoAnalysisPredictor,
    ModelType.suggestion.name: SuggestionPredictor,
    "similarity": SimilarityPredictor,
}
