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
from typing import Any, Optional, Union

from app.commons.model.ml import ModelType
from app.commons.model_chooser import ModelChooser
from app.machine_learning.boosting_featurizer import BoostingFeaturizer
from app.machine_learning.models import BoostingDecisionMaker, DefectTypeModel, WeightedSimilarityCalculator


class Predictor(metaclass=ABCMeta):
    """Abstract base class for prediction workflows using BoostingDecisionMaker and BoostingFeaturizer.

    This class encapsulates the common pattern used in both auto analysis and suggestion prediction:
    1. Acquire models from model chooser
    2. Create and configure featurizer
    3. Extract features and find most relevant items
    4. Make predictions with boosting decision maker
    5. Post-process results
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
        # noinspection PyTypeChecker
        self.boosting_decision_maker: BoostingDecisionMaker = self.model_chooser.choose_model(
            project_id,
            model_type,
            custom_model_prob=custom_model_prob,
            hash_source=hash_source,
        )

        # noinspection PyTypeChecker
        self.defect_type_model: DefectTypeModel = self.model_chooser.choose_model(project_id, ModelType.defect_type)

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
    ) -> tuple[Any, list[str]]:
        """Execute the full prediction workflow.

        Args:
            search_results: List of (log_info, search_results) tuples from Elasticsearch

        Returns:
            Tuple of (prediction_result, model_info_tags)
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

        # Set defect type model on featurizer
        featurizer.set_defect_type_model(self.defect_type_model)

        # Extract features and find most relevant items
        feature_data, identifiers = featurizer.gather_features_info()
        scores_by_type = featurizer.find_most_relevant_by_type()

        # Get model info tags
        model_info_tags = featurizer.get_used_model_info() + self.boosting_decision_maker.get_model_info()

        # If no feature data, return empty result
        if not feature_data:
            return None, model_info_tags

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

        return prediction_result, model_info_tags
