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

from typing import Optional
from unittest.mock import Mock

import pytest

from app.commons.model.db import Hit
from app.commons.model.launch_objects import RelevantItem
from app.commons.model.ml import ModelType
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.query_builder import get_log_inner_hits_name
from app.ml.boosting_featurizer import BoostingFeaturizer
from app.ml.predictor import (
    PREDICTION_CLASSES,
    AutoAnalysisPredictor,
    FeatureInfo,
    PredictionResult,
    SimilarityPredictor,
    SuggestionPredictor,
    extract_text_fields_for_comparison,
)
from app.ml.suggest_boosting_featurizer import SuggestBoostingFeaturizer


def assert_prediction_result_structure(
    result: PredictionResult,
    expected_identity: str,
    expected_feature_ids: list[int],
    expected_feature_data: list[float],
    expected_model_info_tags: list[str],
    expected_original_position: int = 0,
) -> None:
    """Helper function to assert PredictionResult structure.

    :param result: The PredictionResult to validate
    :param expected_identity: Expected identity value
    :param expected_feature_ids: Expected feature IDs
    :param expected_feature_data: Expected feature data
    :param expected_model_info_tags: Expected model info tags
    :param expected_original_position: Expected original position
    """
    assert isinstance(result, PredictionResult)
    assert isinstance(result.label, int)
    assert isinstance(result.probability, list)
    assert len(result.probability) == 2
    assert result.probability[0] + result.probability[1] == pytest.approx(1.0)
    assert isinstance(result.data, RelevantItem)
    assert isinstance(result.data.mrHit, Hit)
    assert isinstance(result.data.compared_item, TestItemIndexData)
    assert result.identity == expected_identity
    assert result.feature_info is not None
    assert isinstance(result.feature_info, FeatureInfo)
    assert result.feature_info.feature_ids == expected_feature_ids
    assert result.feature_info.feature_data == expected_feature_data
    for tag in expected_model_info_tags:
        assert tag in result.model_info_tags
    assert result.original_position == expected_original_position


def build_log(log_id: str, message: str, log_order: int = 0) -> LogData:
    return LogData(log_id=log_id, log_order=log_order, log_level=40000, message=message)


def build_request_item(messages: list[str], test_item_id: str = "1") -> TestItemIndexData:
    return TestItemIndexData(
        test_item_id=test_item_id,
        launch_id="1",
        logs=[build_log(f"{test_item_id}{idx}", message, idx) for idx, message in enumerate(messages)],
    )


def build_hit(
    test_item_id: str = "456",
    matched_messages: Optional[dict[int, str]] = None,
    score: float = 0.95,
    issue_type: str = "pb001",
) -> Hit[TestItemIndexData]:
    """Build a found Test Item hit with named inner hits.

    :param test_item_id: Found Test Item ID
    :param matched_messages: Request log position mapped to the message of the found log matched to it
    :param score: Hit score
    :param issue_type: Issue type of the found Test Item
    :return: Found Test Item hit
    """
    inner_hits = {
        get_log_inner_hits_name(log_index): {
            "hits": {
                "hits": [
                    {
                        "_id": f"{test_item_id}{log_index}",
                        "_score": score,
                        "_source": build_log(f"{test_item_id}{log_index}", message, log_index).model_dump(),
                    }
                ]
            }
        }
        for log_index, message in (matched_messages or {}).items()
    }
    source = TestItemIndexData(test_item_id=test_item_id, launch_id="2", issue_type=issue_type)
    return Hit[TestItemIndexData].from_dict(
        {"_id": test_item_id, "_score": score, "_source": source.model_dump(), "inner_hits": inner_hits}
    )


def build_relevant_item(
    hit: Hit[TestItemIndexData], request_item: TestItemIndexData, original_position: int = 0
) -> RelevantItem:
    return RelevantItem(mrHit=hit, compared_item=request_item, original_position=original_position)


def create_test_search_results(
    message: str = "Error message", test_item_id: str = "456", issue_type: str = "pb001"
) -> tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]:
    return build_request_item([message]), [build_hit(test_item_id, {0: message}, issue_type=issue_type)]


class TestExtractTextFieldsForComparison:
    """Test cases for the extract_text_fields_for_comparison function."""

    @pytest.mark.parametrize(
        "messages, expected_output",
        [
            (
                ["Error: Connection timeout", "Additional context log"],
                "Error: Connection timeout Additional context log",
            ),
            (["Error: Connection timeout"], "Error: Connection timeout"),
            (["", "Only second log here"], "Only second log here"),
            (["", ""], ""),
            ([], ""),
            (["   ", "\t\n"], ""),
            (["  Error message  ", "  Context log  "], "Error message Context log"),
        ],
    )
    def test_extract_text_fields_for_comparison(self, messages, expected_output):
        """Test text extraction from logs."""
        logs = [build_log(str(idx), message) for idx, message in enumerate(messages)]
        result = extract_text_fields_for_comparison(logs)
        assert result == expected_output


class TestSimilarityPredictor:
    """Test cases for the SimilarityPredictor class."""

    def test_predictor_instantiation_with_defaults(self):
        """Test SimilarityPredictor instantiation with default parameters."""
        predictor = SimilarityPredictor()
        assert 0.499 <= predictor.similarity_threshold <= 0.501

    def test_predictor_instantiation_with_kwargs(self):
        """Test SimilarityPredictor instantiation with keyword arguments."""
        predictor = SimilarityPredictor(similarity_threshold=0.7)
        assert 0.699 <= predictor.similarity_threshold <= 0.701

    def test_predictor_in_prediction_classes(self):
        """Test that SimilarityPredictor is properly registered in PREDICTION_CLASSES."""
        assert "similarity" in PREDICTION_CLASSES
        assert PREDICTION_CLASSES["similarity"] == SimilarityPredictor

    def test_predict_with_no_hits(self):
        """Test predict method when search results have no hits."""
        predictor = SimilarityPredictor()
        result = predictor.predict((build_request_item(["Some message"]), []))
        assert result == []

    def test_predict_with_empty_query_text(self):
        """Test predict method when the request Test Item has no meaningful text."""
        predictor = SimilarityPredictor()
        search_results = (build_request_item([""]), [build_hit("456", {0: "Error message"})])
        result = predictor.predict(search_results)
        assert result == []

    def test_predict_with_empty_hit_text(self):
        """Test predict method when found logs have no meaningful text."""
        predictor = SimilarityPredictor()
        search_results = (build_request_item(["Query message"]), [build_hit("456", {0: ""})])
        result = predictor.predict(search_results)
        assert result == []

    def test_predict_with_no_matched_logs(self):
        """Test predict method when a found Test Item has no matched logs."""
        predictor = SimilarityPredictor()
        search_results = (build_request_item(["Query message"]), [build_hit("456", {})])
        result = predictor.predict(search_results)
        assert result == []

    @pytest.mark.parametrize(
        "threshold, expected_label, query_message, hit_message",
        [
            (0.3, 1, "Error: Connection timeout", "Error: Connection timeout"),
            (0.5, 1, "Error: Connection timeout", "Error: Connection timeout"),
            (0.8, 0, "Connection timeout error", "Connection failed error"),
        ],
    )
    def test_predict_threshold_behavior(self, threshold, expected_label, query_message, hit_message):
        """Test that threshold properly affects binary classification."""
        predictor = SimilarityPredictor(similarity_threshold=threshold)
        search_results = (build_request_item([query_message]), [build_hit("456", {0: hit_message})])
        results = predictor.predict(search_results)
        assert len(results) == 1
        assert results[0].label == expected_label

    def test_predict_identical_texts(self):
        """Test predict method with identical texts."""
        predictor = SimilarityPredictor(similarity_threshold=0.5)
        search_results = (build_request_item(["Exact same message"]), [build_hit("456", {0: "Exact same message"})])
        results = predictor.predict(search_results)
        assert len(results) == 1
        assert results[0].label == 1
        assert 0.999 <= results[0].probability[1] <= 1.001
        assert results[0].identity == "456"

    def test_predict_combined_logs(self):
        """Test predict method when combining several request and found logs."""
        predictor = SimilarityPredictor(similarity_threshold=0.5)
        request_item = build_request_item(["Error: Connection timeout", "Additional context"])
        hit = build_hit("456", {0: "Error: Connection timeout", 1: "Additional context"})
        results = predictor.predict((request_item, [hit]))
        assert len(results) == 1
        assert results[0].label == 1
        assert 0.999 <= results[0].probability[1] <= 1.001

    def test_predict_multiple_test_items(self):
        """Test predict method with multiple found Test Items."""
        predictor = SimilarityPredictor(similarity_threshold=0.3)
        request_item = build_request_item(["Error: Connection timeout"])
        hits = [
            build_hit("456", {0: "Error: Connection timeout"}),
            build_hit("789", {0: "Different error message"}),
        ]
        results = predictor.predict((request_item, hits))
        assert len(results) == 2
        assert [result.identity for result in results] == ["456", "789"]
        assert [result.original_position for result in results] == [0, 1]
        assert results[0].probability[1] > results[1].probability[1]

    def test_predict_deduplicates_test_items(self):
        """Test that one result is returned per found Test Item."""
        predictor = SimilarityPredictor(similarity_threshold=0.3)
        request_item = build_request_item(["Error: Connection timeout"])
        hits = [build_hit("456", {0: "Error: Connection timeout"}), build_hit("456", {0: "Other error"})]
        results = predictor.predict((request_item, hits))
        assert len(results) == 1
        assert results[0].original_position == 0

    def test_predict_result_structure(self):
        """Test that PredictionResult objects have correct structure."""
        predictor = SimilarityPredictor(similarity_threshold=0.5)
        results = predictor.predict(create_test_search_results())
        assert len(results) == 1

        result = results[0]
        assert_prediction_result_structure(result, "456", [0], [result.probability[1]], ["similarity_predictor"])

    @pytest.mark.parametrize(
        "query_messages, hit_messages, expected_similarity_range",
        [
            (["Error", "Context"], {0: "Error Context"}, (0.99, 1.01)),
            (["Connection timeout error"], {0: "Connection failed error"}, (0.3, 0.5)),
            (["Database error"], {0: "Network timeout"}, (0.0, 0.2)),
        ],
    )
    def test_predict_similarity_ranges(self, query_messages, hit_messages, expected_similarity_range):
        """Test that similarity calculations fall within expected ranges."""
        predictor = SimilarityPredictor(similarity_threshold=0.1)
        results = predictor.predict((build_request_item(query_messages), [build_hit("456", hit_messages)]))
        assert len(results) == 1
        min_sim, max_sim = expected_similarity_range
        assert min_sim <= results[0].probability[1] <= max_sim

    def test_predict_probability_format(self):
        """Test that probability format matches other predictors."""
        predictor = SimilarityPredictor(similarity_threshold=0.5)
        results = predictor.predict(create_test_search_results())
        assert len(results) == 1

        probability = results[0].probability
        assert len(probability) == 2
        assert probability[0] == pytest.approx(1.0 - probability[1])
        assert 0.0 <= probability[0] <= 1.0
        assert 0.0 <= probability[1] <= 1.0


class TestAutoAnalysisPredictor:
    """Test cases for the AutoAnalysisPredictor class."""

    def create_mock_dependencies(self):
        """Create mock dependencies for AutoAnalysisPredictor."""
        mock_model_chooser = Mock()
        mock_boosting_decision_maker = Mock()
        mock_defect_type_model = Mock()

        mock_model_chooser.choose_model.side_effect = lambda project_id, model_type, **kwargs: (
            mock_boosting_decision_maker if model_type == ModelType.auto_analysis else mock_defect_type_model
        )

        mock_boosting_decision_maker.feature_ids = [0, 1, 3]
        mock_boosting_decision_maker.predict.return_value = ([1], [[0.2, 0.8]])
        mock_boosting_decision_maker.get_model_info.return_value = ["auto_analysis_model"]
        mock_boosting_decision_maker.is_custom = False

        return {
            "model_chooser": mock_model_chooser,
            "project_id": 123,
            "boosting_config": {"test": "config"},
            "custom_model_prob": 0.1,
            "hash_source": "test_hash",
        }

    def test_predictor_instantiation_with_defaults(self):
        """Test AutoAnalysisPredictor instantiation with default parameters."""
        deps = self.create_mock_dependencies()
        del deps["custom_model_prob"]
        del deps["hash_source"]

        predictor = AutoAnalysisPredictor(**deps)

        assert predictor.boosting_config == {"test": "config"}
        deps["model_chooser"].choose_model.assert_any_call(
            123, ModelType.auto_analysis, custom_model_prob=0.0, hash_source=None
        )
        deps["model_chooser"].choose_model.assert_any_call(123, ModelType.defect_type, custom_model_prob=0.0)

    def test_predictor_instantiation_with_kwargs(self):
        """Test AutoAnalysisPredictor instantiation with keyword arguments."""
        deps = self.create_mock_dependencies()
        predictor = AutoAnalysisPredictor(**deps)

        assert predictor.boosting_config == {"test": "config"}
        deps["model_chooser"].choose_model.assert_any_call(
            123, ModelType.auto_analysis, custom_model_prob=0.1, hash_source="test_hash"
        )

    def test_predictor_in_prediction_classes(self):
        """Test that AutoAnalysisPredictor is properly registered in PREDICTION_CLASSES."""
        assert "auto_analysis" in PREDICTION_CLASSES
        assert PREDICTION_CLASSES["auto_analysis"] == AutoAnalysisPredictor

    def test_model_type_property(self):
        """Test that model_type property returns correct value."""
        predictor = AutoAnalysisPredictor(**self.create_mock_dependencies())
        assert predictor.model_type == ModelType.auto_analysis

    def test_create_featurizer(self):
        """Test create_featurizer method."""
        predictor = AutoAnalysisPredictor(**self.create_mock_dependencies())
        featurizer = predictor.create_featurizer((build_request_item(["test"]), []))
        assert type(featurizer) is BoostingFeaturizer

    def test_predict_with_no_hits(self):
        """Test predict method when search results have no hits."""
        predictor = AutoAnalysisPredictor(**self.create_mock_dependencies())
        result = predictor.predict((build_request_item(["Some message"]), []))
        assert result == []
        predictor.boosting_decision_maker.predict.assert_not_called()

    def test_predict_with_real_featurizer(self):
        """Test that one prediction is made per found Test Item, unsupported features are zeros."""
        deps = self.create_mock_dependencies()
        predictor = AutoAnalysisPredictor(**deps)
        predictor.boosting_decision_maker.predict = Mock(return_value=([1, 0], [[0.2, 0.8], [0.7, 0.3]]))
        request_item = build_request_item(["Error message"])
        hits = [build_hit("456", {0: "Error message"}), build_hit("789", {0: "Another error"}, score=0.5)]

        results = predictor.predict((request_item, hits))

        predictor.boosting_decision_maker.predict.assert_called_once_with([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        assert [result.identity for result in results] == ["456", "789"]
        assert [result.label for result in results] == [1, 0]
        assert [result.original_position for result in results] == [0, 1]
        assert [result.data.mrHit.normalized_score for result in results] == [1.0, pytest.approx(0.5 / 0.95)]

    def test_predict_result_structure(self):
        """Test that PredictionResult objects have correct structure."""
        predictor = AutoAnalysisPredictor(**self.create_mock_dependencies())
        request_item, hits = create_test_search_results()

        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([[0.1, 0.2, 0.3]], ["456"])
        mock_featurizer.get_used_model_info.return_value = ["featurizer_info"]
        mock_featurizer.get_relevant_items.return_value = {"456": build_relevant_item(hits[0], request_item)}
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        results = predictor.predict((request_item, hits))
        assert len(results) == 1
        assert_prediction_result_structure(
            results[0], "456", [0, 1, 3], [0.1, 0.2, 0.3], ["auto_analysis_model", "featurizer_info"]
        )

    def test_predict_probability_format(self):
        """Test that probability format is correct."""
        predictor = AutoAnalysisPredictor(**self.create_mock_dependencies())
        results = predictor.predict(create_test_search_results())
        assert len(results) == 1
        assert results[0].probability == [0.2, 0.8]

    def test_predict_multiple_predictions(self):
        """Test predict method with multiple predictions."""
        predictor = AutoAnalysisPredictor(**self.create_mock_dependencies())
        predictor.boosting_decision_maker.predict = Mock(return_value=([1, 0], [[0.2, 0.8], [0.7, 0.3]]))
        request_item = build_request_item(["Error message"])
        hits = [build_hit("456", {0: "Error message"}), build_hit("789", {0: "Another error"})]

        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], ["456", "789"])
        mock_featurizer.get_used_model_info.return_value = ["featurizer_info"]
        mock_featurizer.get_relevant_items.return_value = {
            "456": build_relevant_item(hits[0], request_item, 0),
            "789": build_relevant_item(hits[1], request_item, 1),
        }
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        results = predictor.predict((request_item, hits))
        assert [result.identity for result in results] == ["456", "789"]
        assert [result.label for result in results] == [1, 0]
        assert [result.original_position for result in results] == [0, 1]


class TestSuggestionPredictor:
    """Test cases for the SuggestionPredictor class."""

    def create_mock_dependencies(self):
        """Create mock dependencies for SuggestionPredictor."""
        mock_model_chooser = Mock()
        mock_boosting_decision_maker = Mock()
        mock_defect_type_model = Mock()

        mock_model_chooser.choose_model.side_effect = lambda project_id, model_type, **kwargs: (
            mock_boosting_decision_maker if model_type == ModelType.suggestion else mock_defect_type_model
        )

        mock_boosting_decision_maker.feature_ids = [0, 1, 3]
        mock_boosting_decision_maker.predict.return_value = ([1], [[0.3, 0.7]])
        mock_boosting_decision_maker.get_model_info.return_value = ["suggestion_model"]
        mock_boosting_decision_maker.is_custom = False

        return {
            "model_chooser": mock_model_chooser,
            "project_id": 123,
            "boosting_config": {"suggestion": "config"},
            "custom_model_prob": 0.2,
            "hash_source": "suggestion_hash",
        }

    def test_predictor_instantiation_with_defaults(self):
        """Test SuggestionPredictor instantiation with default parameters."""
        deps = self.create_mock_dependencies()
        del deps["custom_model_prob"]
        del deps["hash_source"]

        predictor = SuggestionPredictor(**deps)

        assert predictor.boosting_config == {"suggestion": "config"}
        deps["model_chooser"].choose_model.assert_any_call(
            123, ModelType.suggestion, custom_model_prob=0.0, hash_source=None
        )
        deps["model_chooser"].choose_model.assert_any_call(123, ModelType.defect_type, custom_model_prob=0.0)

    def test_predictor_instantiation_with_kwargs(self):
        """Test SuggestionPredictor instantiation with keyword arguments."""
        deps = self.create_mock_dependencies()
        predictor = SuggestionPredictor(**deps)

        assert predictor.boosting_config == {"suggestion": "config"}
        deps["model_chooser"].choose_model.assert_any_call(
            123, ModelType.suggestion, custom_model_prob=0.2, hash_source="suggestion_hash"
        )

    def test_predictor_in_prediction_classes(self):
        """Test that SuggestionPredictor is properly registered in PREDICTION_CLASSES."""
        assert "suggestion" in PREDICTION_CLASSES
        assert PREDICTION_CLASSES["suggestion"] == SuggestionPredictor

    def test_model_type_property(self):
        """Test that model_type property returns correct value."""
        predictor = SuggestionPredictor(**self.create_mock_dependencies())
        assert predictor.model_type == ModelType.suggestion

    def test_create_featurizer(self):
        """Test create_featurizer method."""
        predictor = SuggestionPredictor(**self.create_mock_dependencies())
        featurizer = predictor.create_featurizer((build_request_item(["test"]), []))
        assert isinstance(featurizer, SuggestBoostingFeaturizer)

    def test_predict_with_no_hits(self):
        """Test predict method when search results have no hits."""
        predictor = SuggestionPredictor(**self.create_mock_dependencies())
        result = predictor.predict((build_request_item(["Some message"]), []))
        assert result == []
        predictor.boosting_decision_maker.predict.assert_not_called()

    def test_predict_result_structure(self):
        """Test that PredictionResult objects have correct structure."""
        predictor = SuggestionPredictor(**self.create_mock_dependencies())
        request_item, hits = create_test_search_results(message="Suggestion message", test_item_id="789")

        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([[0.4, 0.5, 0.6]], ["789"])
        mock_featurizer.get_used_model_info.return_value = ["suggestion_featurizer_info"]
        mock_featurizer.get_relevant_items.return_value = {"789": build_relevant_item(hits[0], request_item)}
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        results = predictor.predict((request_item, hits))
        assert len(results) == 1
        assert_prediction_result_structure(
            results[0], "789", [0, 1, 3], [0.4, 0.5, 0.6], ["suggestion_model", "suggestion_featurizer_info"]
        )

    def test_predict_probability_format(self):
        """Test that probability format is correct."""
        predictor = SuggestionPredictor(**self.create_mock_dependencies())
        results = predictor.predict(create_test_search_results(message="Suggestion message", test_item_id="789"))
        assert len(results) == 1
        assert results[0].probability == [0.3, 0.7]

    def test_predict_multiple_predictions(self):
        """Test predict method with multiple predictions."""
        predictor = SuggestionPredictor(**self.create_mock_dependencies())
        predictor.boosting_decision_maker.predict = Mock(return_value=([0, 1], [[0.6, 0.4], [0.1, 0.9]]))
        request_item = build_request_item(["Suggestion message"])
        hits = [build_hit("789", {0: "Suggestion message"}), build_hit("101", {0: "Other message"})]

        results = predictor.predict((request_item, hits))
        assert [result.identity for result in results] == ["789", "101"]
        assert [result.label for result in results] == [0, 1]
