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

from app.commons.model.ml import ModelType
from app.commons.model_chooser import ModelChooser
from app.machine_learning.boosting_featurizer import BoostingFeaturizer
from app.machine_learning.models import BoostingDecisionMaker, DefectTypeModel, WeightedSimilarityCalculator
from app.machine_learning.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from app.utils import utils


@dataclass
class PredictionResult:
    """Result container for prediction workflows.

    Attributes:
        prediction_result: The actual prediction result (varies by implementation)
        model_info_tags: List of model information tags
    """

    prediction_result: Any
    model_info_tags: list[str]


class Predictor(metaclass=ABCMeta):
    """Abstract base class for prediction workflows using BoostingDecisionMaker and BoostingFeaturizer.

    This class encapsulates the common pattern used in both auto analysis and suggestion prediction:
    1. Acquire models from model chooser
    2. Create and configure featurizer
    3. Extract features and find most relevant items
    4. Make predictions with boosting decision maker
    5. Post-process results
    """

    boosting_decision_maker: BoostingDecisionMaker
    defect_type_model: DefectTypeModel

    def __init__(
        self,
        model_chooser: ModelChooser,
        project_id: int,
        model_type: ModelType,
        boosting_config: dict[str, Any],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
        custom_model_prob: float = 0.0,
        hash_source: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize the predictor with required dependencies.

        Args:
            model_chooser: Service for choosing appropriate ML models
            project_id: Project identifier for model selection
            model_type: Type of model to use (auto_analysis or suggestion)
            boosting_config: Configuration for the boosting featurizer
            weighted_log_similarity_calculator: Model for calculating log similarities
            custom_model_prob: Probability to use custom model instead of global
            hash_source: Source for hash-based model selection
        """
        self.model_chooser = model_chooser
        self.project_id = project_id
        self.model_type = model_type
        self.boosting_config = boosting_config
        self.weighted_log_similarity_calculator = weighted_log_similarity_calculator
        self.custom_model_prob = custom_model_prob
        self.hash_source = hash_source

        # Acquire models
        self.boosting_decision_maker = self.model_chooser.choose_model(  # type: ignore[assignment]
            project_id,
            model_type,
            custom_model_prob=custom_model_prob,
            hash_source=hash_source,
        )
        self.defect_type_model = self.model_chooser.choose_model(  # type: ignore[assignment]
            project_id, ModelType.defect_type
        )

    @abstractmethod
    def create_featurizer(
        self,
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
        boosting_config: dict[str, Any],
        feature_ids: list[int],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
    ) -> BoostingFeaturizer:
        """Create the appropriate featurizer for this prediction type.

        Args:
            search_results: List of (log_info, search_results) tuples
            boosting_config: Configuration for the featurizer
            feature_ids: List of feature IDs to use
            weighted_log_similarity_calculator: Similarity calculator model

        Returns:
            Configured featurizer instance
        """
        pass

    @abstractmethod
    def post_process_predictions(
        self,
        predicted_labels: list[int],
        predicted_labels_probability: list[list[float]],
        feature_data: list[list[float]],
        identifiers: list[str],
        scores_by_type: dict[str, dict[str, Any]],
    ) -> Any:
        """Post-process the prediction results.

        Args:
            predicted_labels: Binary prediction labels from the decision maker
            predicted_labels_probability: Prediction probabilities from the decision maker
            feature_data: Feature vectors used for prediction
            identifiers: Issue type names (auto analysis) or test item IDs (suggestions)
            scores_by_type: Most relevant items by type from the featurizer

        Returns:
            Post-processed prediction result (varies by implementation)
        """
        pass

    def predict(
        self,
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> PredictionResult:
        """Execute the full prediction workflow.

        Args:
            search_results: List of (log_info, search_results) tuples from Elasticsearch

        Returns:
            PredictionResult containing:
            - prediction_result: Result from post_process_predictions
            - model_info_tags: List of model information tags
        """
        # Create and configure featurizer
        featurizer = self.create_featurizer(
            search_results,
            self.boosting_config,
            self.boosting_decision_maker.feature_ids,
            self.weighted_log_similarity_calculator,
        )

        # Extract features and find most relevant items
        feature_data, identifiers = featurizer.gather_features_info()
        scores_by_type = featurizer.find_most_relevant_by_type()

        # Get model info tags
        model_info_tags = featurizer.get_used_model_info() + self.boosting_decision_maker.get_model_info()

        # If no feature data, return empty result
        if not feature_data:
            return PredictionResult(prediction_result=None, model_info_tags=model_info_tags)

        # Make predictions
        predicted_labels, predicted_labels_probability = self.boosting_decision_maker.predict(feature_data)

        # Post-process results
        prediction_result = self.post_process_predictions(
            predicted_labels,
            predicted_labels_probability,
            feature_data,
            identifiers,
            scores_by_type,
        )

        return PredictionResult(prediction_result=prediction_result, model_info_tags=model_info_tags)


class AutoAnalysisPredictor(Predictor):
    """Concrete predictor implementation for auto analysis workflow.

    Uses BoostingFeaturizer and choose_issue_type post-processing logic.
    """

    def create_featurizer(
        self,
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
        boosting_config: dict[str, Any],
        feature_ids: list[int],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
    ) -> BoostingFeaturizer:
        """Create a BoostingFeaturizer for auto analysis."""
        featurizer = BoostingFeaturizer(
            search_results,
            boosting_config,
            feature_ids=feature_ids,
            weighted_log_similarity_calculator=weighted_log_similarity_calculator,
        )
        featurizer.set_defect_type_model(self.defect_type_model)
        return featurizer

    def post_process_predictions(
        self,
        predicted_labels: list[int],
        predicted_labels_probability: list[list[float]],
        feature_data: list[list[float]],
        identifiers: list[str],
        scores_by_type: dict[str, dict[str, Any]],
    ) -> tuple[str, float, int]:
        """Post-process predictions using choose_issue_type logic.

        Returns:
            Tuple of (predicted_issue_type, probability, global_index)
        """
        issue_type_names = identifiers
        scores_by_issue_type = scores_by_type

        predicted_issue_type, prob, global_idx = utils.choose_issue_type(
            predicted_labels, predicted_labels_probability, issue_type_names, scores_by_issue_type
        )

        return predicted_issue_type, prob, global_idx


class SuggestionPredictor(Predictor):
    """Concrete predictor implementation for suggestion workflow.

    Uses SuggestBoostingFeaturizer and sort_results post-processing logic.
    """

    def __init__(
        self,
        model_chooser: ModelChooser,
        project_id: int,
        model_type: ModelType,
        boosting_config: dict[str, Any],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
        custom_model_prob: float = 0.0,
        hash_source: Optional[Union[int, str]] = None,
        suggest_threshold: float = 0.4,
    ) -> None:
        """Initialize suggestion predictor with additional threshold parameter."""
        super().__init__(
            model_chooser,
            project_id,
            model_type,
            boosting_config,
            weighted_log_similarity_calculator,
            custom_model_prob,
            hash_source,
        )
        self.suggest_threshold = suggest_threshold

    def create_featurizer(
        self,
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
        boosting_config: dict[str, Any],
        feature_ids: list[int],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
    ) -> BoostingFeaturizer:
        """Create a SuggestBoostingFeaturizer for suggestions."""
        featurizer = SuggestBoostingFeaturizer(
            search_results,
            boosting_config,
            feature_ids=feature_ids,
            weighted_log_similarity_calculator=weighted_log_similarity_calculator,
        )
        featurizer.set_defect_type_model(self.defect_type_model)
        return featurizer

    def post_process_predictions(
        self,
        predicted_labels: list[int],
        predicted_labels_probability: list[list[float]],
        feature_data: list[list[float]],
        identifiers: list[str],
        scores_by_type: dict[str, dict[str, Any]],
    ) -> list[tuple[int, float, str]]:
        """Post-process predictions using sort_results logic.

        Returns:
            List of sorted results as (idx, probability, start_time) tuples
        """
        test_item_ids = identifiers
        scores_by_test_items = scores_by_type

        gathered_results = []
        for idx, prob in enumerate(predicted_labels_probability):
            test_item_id = test_item_ids[idx]
            gathered_results.append(
                (idx, round(prob[1], 4), scores_by_test_items[test_item_id]["mrHit"]["_source"]["start_time"])
            )

        # Sort by probability and start_time in descending order
        gathered_results = sorted(gathered_results, key=lambda x: (x[1], x[2]), reverse=True)

        # Apply deduplication logic (simplified version)
        return self._deduplicate_results(gathered_results, scores_by_test_items, test_item_ids)

    def _deduplicate_results(
        self,
        gathered_results: list[tuple[int, float, str]],
        scores_by_test_items: dict[str, dict[str, Any]],
        test_item_ids: list[str],
    ) -> list[tuple[int, float, str]]:
        """Deduplicate results by issue type, keeping the one with highest probability."""
        issue_type_dict = {}
        for idx, prob, start_time in gathered_results:
            test_item_id = test_item_ids[idx]
            issue_type = scores_by_test_items[test_item_id]["mrHit"]["_source"]["issue_type"]

            if issue_type not in issue_type_dict or prob > issue_type_dict[issue_type][1]:
                issue_type_dict[issue_type] = (idx, prob, start_time)

        # Convert back to list and sort by probability
        deduplicated_results = list(issue_type_dict.values())
        deduplicated_results.sort(key=lambda x: (x[1], x[2]), reverse=True)

        return deduplicated_results
