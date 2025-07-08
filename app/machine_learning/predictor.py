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


@dataclass
class PredictionMetadata:
    """Metadata container for prediction results.

    Attributes:
        esScore: Optional[float] - Elasticsearch score for the prediction
        esPosition: Optional[int] - Position in Elasticsearch results
        modelFeatureNames: Optional[str] - Names of features used in the model
        modelFeatureValues: Optional[str] - Values of features used in the model
        modelInfo: Optional[str] - Additional information about the model used
    """

    esScore: Optional[float]
    esPosition: Optional[int]
    modelFeatureNames: Optional[str]
    modelFeatureValues: Optional[str]
    modelInfo: Optional[str]


@dataclass
class PredictionResult:
    """Result container for prediction workflows.

    Attributes:
        label: Binary prediction label from the decision maker
        probability: Prediction probability from the decision maker
        scores: Most relevant log and its metadata for the result
        identity: Identity for the gathered features
        feature_data: Feature vector for the identity gathered from the featurizer
        model_info_tags: List of model information tags
    """

    label: int
    probability: list[float]
    scores: dict[str, Any]
    identity: str
    feature_data: list[float]
    model_info_tags: list[str]


class Predictor(metaclass=ABCMeta):
    """Abstract base class for prediction workflows using BoostingDecisionMaker and BoostingFeaturizer.

    This class encapsulates the common pattern used in both auto analysis and suggestion prediction:
    1. Acquire models from model chooser
    2. Create and configure featurizer
    3. Extract features and find most relevant items
    4. Make predictions with boosting decision maker
    5. Return prediction results
    """

    boosting_decision_maker: BoostingDecisionMaker
    defect_type_model: DefectTypeModel

    def __init__(
        self,
        model_chooser: ModelChooser,
        project_id: int,
        boosting_config: dict[str, Any],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
        custom_model_prob: float = 0.0,
        hash_source: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize the predictor with required dependencies.

        :param ModelChooser model_chooser: Service for choosing appropriate ML models
        :param int project_id: Project identifier for model selection
        :param dict[str, Any] boosting_config: Configuration for the boosting featurizer
        :param WeightedSimilarityCalculator weighted_log_similarity_calculator: Model for calculating log similarities
        :param float custom_model_prob: Probability to use custom model instead of global
        :param Optional[Union[int, str]] hash_source: Source for hash-based model selection
        """
        self.model_chooser = model_chooser
        self.project_id = project_id
        self.boosting_config = boosting_config
        self.weighted_log_similarity_calculator = weighted_log_similarity_calculator
        self.custom_model_prob = custom_model_prob
        self.hash_source = hash_source

        # Acquire models
        self.boosting_decision_maker = self.model_chooser.choose_model(  # type: ignore[assignment]
            project_id,
            self.model_type,
            custom_model_prob=custom_model_prob,
            hash_source=hash_source,
        )
        self.defect_type_model = self.model_chooser.choose_model(  # type: ignore[assignment]
            project_id, ModelType.defect_type
        )

    @property
    @abstractmethod
    def model_type(self) -> ModelType:
        """Return the type of model used by this predictor."""
        ...

    @abstractmethod
    def create_featurizer(
        self,
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
        boosting_config: dict[str, Any],
        feature_ids: list[int],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
    ) -> BoostingFeaturizer:
        """Create the appropriate featurizer for this prediction type.

        :param list[tuple[dict[str, Any], dict[str, Any]]] search_results: List of (log_info, search_results) tuples
        :param dict[str, Any] boosting_config: Configuration for the featurizer
        :param list[int] feature_ids: List of feature IDs to use
        :param WeightedSimilarityCalculator weighted_log_similarity_calculator: Similarity calculator model
        :return: Configured featurizer instance
        """
        ...

    def predict(
        self,
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
    ) -> list[PredictionResult]:
        """Execute the full prediction workflow.

        :param list[tuple[dict[str, Any], dict[str, Any]]] search_results: List of (log_info, search_results) tuples
                                                                           from Elasticsearch
        :return: List of PredictionResult objects, one for each prediction
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

        # Get model info tags
        model_info_tags = featurizer.get_used_model_info() + self.boosting_decision_maker.get_model_info()

        # If no feature data, return empty result
        if not feature_data:
            return []

        # Make predictions
        predicted_labels, predicted_labels_probability = self.boosting_decision_maker.predict(feature_data)

        # Get scores by identity
        scores_by_identity = featurizer.find_most_relevant_by_type()

        # Create list of PredictionResult objects, one for each prediction
        results = []
        for idx, identity in enumerate(identifiers):
            result = PredictionResult(
                label=predicted_labels[idx],
                probability=predicted_labels_probability[idx],
                scores=scores_by_identity[identity],
                identity=identity,
                feature_data=feature_data[idx],
                model_info_tags=model_info_tags,
            )
            results.append(result)

        return results


class AutoAnalysisPredictor(Predictor):
    """Concrete predictor implementation for auto analysis workflow.

    Uses BoostingFeaturizer for feature extraction and prediction.
    """

    def __init__(
        self,
        model_chooser: ModelChooser,
        project_id: int,
        boosting_config: dict[str, Any],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
        custom_model_prob: float = 0.0,
        hash_source: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize auto analysis predictor.

        :param ModelChooser model_chooser: Service for choosing appropriate ML models
        :param int project_id: Project identifier for model selection
        :param dict[str, Any] boosting_config: Configuration for the boosting featurizer
        :param WeightedSimilarityCalculator weighted_log_similarity_calculator: Model for calculating log similarities
        :param float custom_model_prob: Probability to use custom model instead of global
        :param Optional[Union[int, str]] hash_source: Source for hash-based model selection
        """
        super().__init__(
            model_chooser,
            project_id,
            boosting_config,
            weighted_log_similarity_calculator,
            custom_model_prob,
            hash_source,
        )

    @property
    def model_type(self) -> ModelType:
        """Return the type of model used by this predictor."""
        return ModelType.auto_analysis

    def create_featurizer(
        self,
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
        boosting_config: dict[str, Any],
        feature_ids: list[int],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
    ) -> BoostingFeaturizer:
        """Create a BoostingFeaturizer for auto analysis.

        :param list[tuple[dict[str, Any], dict[str, Any]]] search_results: List of (log_info, search_results) tuples
        :param dict[str, Any] boosting_config: Configuration for the featurizer
        :param list[int] feature_ids: List of feature IDs to use
        :param WeightedSimilarityCalculator weighted_log_similarity_calculator: Similarity calculator model
        :return: Configured BoostingFeaturizer instance
        """
        featurizer = BoostingFeaturizer(
            search_results,
            boosting_config,
            feature_ids=feature_ids,
            weighted_log_similarity_calculator=weighted_log_similarity_calculator,
        )
        featurizer.set_defect_type_model(self.defect_type_model)
        return featurizer


class SuggestionPredictor(Predictor):
    """Concrete predictor implementation for suggestion workflow.

    Uses SuggestBoostingFeaturizer for feature extraction and prediction.
    """

    def __init__(
        self,
        model_chooser: ModelChooser,
        project_id: int,
        boosting_config: dict[str, Any],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
        custom_model_prob: float = 0.0,
        hash_source: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize suggestion predictor.

        :param ModelChooser model_chooser: Service for choosing appropriate ML models
        :param int project_id: Project identifier for model selection
        :param dict[str, Any] boosting_config: Configuration for the boosting featurizer
        :param WeightedSimilarityCalculator weighted_log_similarity_calculator: Model for calculating log similarities
        :param float custom_model_prob: Probability to use custom model instead of global
        :param Optional[Union[int, str]] hash_source: Source for hash-based model selection
        """
        super().__init__(
            model_chooser,
            project_id,
            boosting_config,
            weighted_log_similarity_calculator,
            custom_model_prob,
            hash_source,
        )

    @property
    def model_type(self) -> ModelType:
        """Return the type of model used by this predictor."""
        return ModelType.suggestion

    def create_featurizer(
        self,
        search_results: list[tuple[dict[str, Any], dict[str, Any]]],
        boosting_config: dict[str, Any],
        feature_ids: list[int],
        weighted_log_similarity_calculator: WeightedSimilarityCalculator,
    ) -> BoostingFeaturizer:
        """Create a SuggestBoostingFeaturizer for suggestions.

        :param list[tuple[dict[str, Any], dict[str, Any]]] search_results: List of (log_info, search_results) tuples
        :param dict[str, Any] boosting_config: Configuration for the featurizer
        :param list[int] feature_ids: List of feature IDs to use
        :param WeightedSimilarityCalculator weighted_log_similarity_calculator: Similarity calculator model
        :return: Configured SuggestBoostingFeaturizer instance
        """
        featurizer = SuggestBoostingFeaturizer(
            search_results,
            boosting_config,
            feature_ids=feature_ids,
            weighted_log_similarity_calculator=weighted_log_similarity_calculator,
        )
        featurizer.set_defect_type_model(self.defect_type_model)
        return featurizer


PREDICTION_CLASSES: dict[str, type[Predictor]] = {
    ModelType.auto_analysis.name: AutoAnalysisPredictor,
    ModelType.suggestion.name: SuggestionPredictor,
}
