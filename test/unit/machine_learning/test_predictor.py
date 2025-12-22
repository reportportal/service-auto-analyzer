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

from unittest.mock import Mock

import pytest

from app.commons.model.ml import ModelType
from app.ml.predictor import (
    PREDICTION_CLASSES,
    AutoAnalysisPredictor,
    FeatureInfo,
    PredictionResult,
    SimilarityPredictor,
    SuggestionPredictor,
    extract_text_fields_for_comparison,
)


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
    assert isinstance(result.data, dict)
    assert "mrHit" in result.data
    assert "compared_log" in result.data
    assert result.identity == expected_identity
    assert result.feature_info is not None
    assert isinstance(result.feature_info, FeatureInfo)
    assert result.feature_info.feature_ids == expected_feature_ids
    assert result.feature_info.feature_data == expected_feature_data
    for tag in expected_model_info_tags:
        assert tag in result.model_info_tags
    assert result.original_position == expected_original_position


def create_test_search_results(
    message: str = "Error message", log_id: str = "log1", test_item: int = 456, issue_type: str = "pb001"
) -> list[tuple[dict, dict]]:
    """Helper function to create test search results structure.

    :param message: Message content for both query and hit
    :param log_id: Log ID for the hit
    :param test_item: Test item ID
    :param issue_type: Issue type
    :return: List of (search_request, search_results) tuples
    """
    return [
        (
            {"_source": {"message": message, "merged_small_logs": ""}},
            {
                "hits": {
                    "hits": [
                        {
                            "_id": log_id,
                            "_source": {
                                "message": message,
                                "merged_small_logs": "",
                                "test_item": test_item,
                                "issue_type": issue_type,
                            },
                        }
                    ]
                }
            },
        )
    ]


class TestExtractTextFieldsForComparison:
    """Test cases for the extract_text_fields_for_comparison function."""

    @pytest.mark.parametrize(
        "search_request, expected_output",
        [
            # Both fields present
            (
                {
                    "_source": {
                        "message": "Error: Connection timeout",
                        "merged_small_logs": "Additional context log",
                    }
                },
                "Error: Connection timeout Additional context log",
            ),
            # Only message field present
            (
                {
                    "_source": {
                        "message": "Error: Connection timeout",
                        "merged_small_logs": "",
                    }
                },
                "Error: Connection timeout",
            ),
            # Only merged_small_logs field present
            (
                {
                    "_source": {
                        "message": "",
                        "merged_small_logs": "Only merged logs here",
                    }
                },
                "Only merged logs here",
            ),
            # Both fields empty
            (
                {
                    "_source": {
                        "message": "",
                        "merged_small_logs": "",
                    }
                },
                "",
            ),
            # Fields with None values
            (
                {
                    "_source": {
                        "message": None,
                        "merged_small_logs": None,
                    }
                },
                "",
            ),
            # Missing _source
            ({}, ""),
            # Missing fields in _source
            ({"_source": {}}, ""),
            # Fields with whitespace only
            (
                {
                    "_source": {
                        "message": "   ",
                        "merged_small_logs": "\t\n",
                    }
                },
                "",
            ),
            # Mixed whitespace and content
            (
                {
                    "_source": {
                        "message": "  Error message  ",
                        "merged_small_logs": "  Context log  ",
                    }
                },
                "Error message Context log",
            ),
        ],
    )
    def test_extract_text_fields_for_comparison(self, search_request, expected_output):
        """Test text field extraction from search request."""
        result = extract_text_fields_for_comparison(search_request)
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

    def test_predict_with_empty_input(self):
        """Test predict method with empty search results."""
        predictor = SimilarityPredictor()
        result = predictor.predict([])
        assert result == []

    def test_predict_with_no_hits(self):
        """Test predict method when search results have no hits."""
        predictor = SimilarityPredictor()
        search_results = [
            (
                {"_source": {"message": "Some message", "merged_small_logs": ""}},
                {"hits": {"hits": []}},
            )
        ]
        result = predictor.predict(search_results)
        assert result == []

    def test_predict_with_empty_query_text(self):
        """Test predict method when query has no meaningful text."""
        predictor = SimilarityPredictor()
        search_results = [
            (
                {"_source": {"message": "", "merged_small_logs": ""}},  # Empty query
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": "Error message",
                                    "merged_small_logs": "",
                                    "test_item": 456,
                                },
                                "_score": 0.95,
                            }
                        ]
                    }
                },
            )
        ]
        result = predictor.predict(search_results)
        assert result == []

    def test_predict_with_empty_hit_text(self):
        """Test predict method when hits have no meaningful text."""
        predictor = SimilarityPredictor()
        search_results = [
            (
                {"_source": {"message": "Query message", "merged_small_logs": ""}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": "",  # Empty hit
                                    "merged_small_logs": "",
                                    "test_item": 456,
                                },
                                "_score": 0.95,
                            }
                        ]
                    }
                },
            )
        ]
        result = predictor.predict(search_results)
        assert result == []

    @pytest.mark.parametrize(
        "threshold, expected_label, query_message, hit_message",
        [
            (
                0.3,
                1,
                "Error: Connection timeout",
                "Error: Connection timeout",
            ),  # High similarity, should be above threshold
            (
                0.5,
                1,
                "Error: Connection timeout",
                "Error: Connection timeout",
            ),  # Moderate similarity, should be above threshold
            (
                0.8,
                0,
                "Connection timeout error",
                "Connection failed error",
            ),  # High threshold, should be below threshold
        ],
    )
    def test_predict_threshold_behavior(self, threshold, expected_label, query_message, hit_message):
        """Test that threshold properly affects binary classification."""
        predictor = SimilarityPredictor(similarity_threshold=threshold)
        search_results = [
            (
                {"_source": {"message": query_message, "merged_small_logs": ""}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": hit_message,
                                    "merged_small_logs": "",
                                    "test_item": 456,
                                },
                            }
                        ]
                    }
                },
            )
        ]
        results = predictor.predict(search_results)
        assert len(results) == 1
        assert results[0].label == expected_label

    def test_predict_identical_texts(self):
        """Test predict method with identical texts."""
        predictor = SimilarityPredictor(similarity_threshold=0.5)
        search_results = [
            (
                {"_source": {"message": "Exact same message", "merged_small_logs": ""}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": "Exact same message",
                                    "merged_small_logs": "",
                                    "test_item": 456,
                                },
                            }
                        ]
                    }
                },
            )
        ]
        results = predictor.predict(search_results)
        assert len(results) == 1
        assert results[0].label == 1
        assert 0.999 <= results[0].probability[1] <= 1.001
        assert results[0].identity == "456"

    def test_predict_combined_text_fields(self):
        """Test predict method when combining message and merged_small_logs fields."""
        predictor = SimilarityPredictor(similarity_threshold=0.5)
        search_results = [
            (
                {
                    "_source": {
                        "message": "Error: Connection timeout",
                        "merged_small_logs": "Additional context",
                    }
                },
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": "Error: Connection timeout Additional context",
                                    "merged_small_logs": "",
                                    "test_item": 456,
                                },
                            }
                        ]
                    }
                },
            )
        ]
        results = predictor.predict(search_results)
        assert len(results) == 1
        assert results[0].label == 1  # Should be high similarity
        assert 0.999 <= results[0].probability[1] <= 1.001  # Should be identical after combining

    def test_predict_multiple_test_items(self):
        """Test predict method with multiple test items."""
        predictor = SimilarityPredictor(similarity_threshold=0.3)
        search_results = [
            (
                {"_source": {"message": "Error: Connection timeout", "merged_small_logs": ""}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": "Error: Connection timeout",
                                    "merged_small_logs": "",
                                    "test_item": 456,
                                },
                            },
                            {
                                "_source": {
                                    "message": "Different error message",
                                    "merged_small_logs": "",
                                    "test_item": 789,
                                },
                            },
                        ]
                    }
                },
            )
        ]
        results = predictor.predict(search_results)
        assert len(results) == 2

        # Results should be for different test items
        test_items = {result.identity for result in results}
        assert test_items == {"456", "789"}

        # First result should have higher similarity
        result_456 = next(r for r in results if r.identity == "456")
        result_789 = next(r for r in results if r.identity == "789")
        assert result_456.probability[1] > result_789.probability[1]

    def test_predict_result_structure(self):
        """Test that PredictionResult objects have correct structure."""
        predictor = SimilarityPredictor(similarity_threshold=0.5)
        search_results = create_test_search_results()
        results = predictor.predict(search_results)
        assert len(results) == 1

        result = results[0]
        assert_prediction_result_structure(
            result,
            "456",
            [0],
            [result.probability[1]],
            ["similarity_predictor"],
        )

    def test_predict_multiple_search_requests(self):
        """Test predict method with multiple search requests."""
        predictor = SimilarityPredictor(similarity_threshold=0.5)
        search_results = [
            (
                {"_source": {"message": "First error", "merged_small_logs": ""}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": "First error",
                                    "merged_small_logs": "",
                                    "test_item": 456,
                                },
                            }
                        ]
                    }
                },
            ),
            (
                {"_source": {"message": "Second error", "merged_small_logs": ""}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": "Second error",
                                    "merged_small_logs": "",
                                    "test_item": 789,
                                },
                            }
                        ]
                    }
                },
            ),
        ]
        results = predictor.predict(search_results)
        assert len(results) == 2

        test_items = {result.identity for result in results}
        assert test_items == {"456", "789"}

    @pytest.mark.parametrize(
        "query_message, query_merged_logs, hit_message, hit_merged_logs, expected_similarity_range",
        [
            # Identical combined texts
            ("Error", "Context", "Error Context", "", (0.99, 1.01)),
            # Partially similar texts
            ("Connection timeout error", "", "Connection failed error", "", (0.3, 0.5)),
            # Completely different texts
            ("Database error", "", "Network timeout", "", (0.0, 0.2)),
            # Empty query with non-empty hit (should be skipped, but test range anyway)
            ("", "", "Some error", "", (0.0, 0.01)),
        ],
    )
    def test_predict_similarity_ranges(
        self, query_message, query_merged_logs, hit_message, hit_merged_logs, expected_similarity_range
    ):
        """Test that similarity calculations fall within expected ranges."""
        predictor = SimilarityPredictor(similarity_threshold=0.1)  # Low threshold to test all cases
        search_results = [
            (
                {
                    "_source": {
                        "message": query_message,
                        "merged_small_logs": query_merged_logs,
                    }
                },
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": hit_message,
                                    "merged_small_logs": hit_merged_logs,
                                    "test_item": 456,
                                },
                            }
                        ]
                    }
                },
            )
        ]
        results = predictor.predict(search_results)

        if query_message or query_merged_logs:  # Only if query has content
            assert len(results) == 1
            similarity = results[0].probability[1]
            min_sim, max_sim = expected_similarity_range
            assert min_sim <= similarity <= max_sim
        else:
            assert len(results) == 0  # Empty query should be skipped

    def test_predict_probability_format(self):
        """Test that probability format matches other predictors."""
        predictor = SimilarityPredictor(similarity_threshold=0.5)
        search_results = [
            (
                {"_source": {"message": "Error message", "merged_small_logs": ""}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": "Error message",
                                    "merged_small_logs": "",
                                    "test_item": 456,
                                },
                            }
                        ]
                    }
                },
            )
        ]
        results = predictor.predict(search_results)
        assert len(results) == 1

        probability = results[0].probability
        assert len(probability) == 2
        assert 1.0 - probability[1] - 0.001 <= probability[0] <= 1.0 - probability[1] + 0.001
        assert 0.0 <= probability[0] <= 1.0
        assert 0.0 <= probability[1] <= 1.0


class TestAutoAnalysisPredictor:
    """Test cases for the AutoAnalysisPredictor class."""

    def create_mock_dependencies(self):
        """Create mock dependencies for AutoAnalysisPredictor."""
        mock_model_chooser = Mock()
        mock_boosting_decision_maker = Mock()
        mock_defect_type_model = Mock()

        # Configure model chooser to return our mocks
        mock_model_chooser.choose_model.side_effect = lambda project_id, model_type, **kwargs: (
            mock_boosting_decision_maker if model_type == ModelType.auto_analysis else mock_defect_type_model
        )

        # Configure basic properties
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
        # Remove optional parameters to test defaults
        del deps["custom_model_prob"]
        del deps["hash_source"]

        predictor = AutoAnalysisPredictor(**deps)

        assert predictor.boosting_config == {"test": "config"}

        # Verify model chooser was called correctly with defaults
        deps["model_chooser"].choose_model.assert_any_call(
            123, ModelType.auto_analysis, custom_model_prob=0.0, hash_source=None
        )
        deps["model_chooser"].choose_model.assert_any_call(123, ModelType.defect_type, custom_model_prob=0.0)

    def test_predictor_instantiation_with_kwargs(self):
        """Test AutoAnalysisPredictor instantiation with keyword arguments."""
        deps = self.create_mock_dependencies()
        predictor = AutoAnalysisPredictor(**deps)

        assert predictor.boosting_config == {"test": "config"}

        # Verify model chooser was called with custom parameters
        deps["model_chooser"].choose_model.assert_any_call(
            123, ModelType.auto_analysis, custom_model_prob=0.1, hash_source="test_hash"
        )

    def test_predictor_in_prediction_classes(self):
        """Test that AutoAnalysisPredictor is properly registered in PREDICTION_CLASSES."""
        assert "auto_analysis" in PREDICTION_CLASSES
        assert PREDICTION_CLASSES["auto_analysis"] == AutoAnalysisPredictor

    def test_model_type_property(self):
        """Test that model_type property returns correct value."""
        deps = self.create_mock_dependencies()
        predictor = AutoAnalysisPredictor(**deps)
        assert predictor.model_type == ModelType.auto_analysis

    def test_create_featurizer(self):
        """Test create_featurizer method."""
        deps = self.create_mock_dependencies()
        predictor = AutoAnalysisPredictor(**deps)

        search_results = [({"_source": {"message": "test"}}, {"hits": {"hits": []}})]

        featurizer = predictor.create_featurizer(search_results)

        # Verify it's the right type and has expected configuration
        from app.ml.boosting_featurizer import BoostingFeaturizer

        assert isinstance(featurizer, BoostingFeaturizer)

    def test_predict_with_empty_input(self):
        """Test predict method with empty search results."""
        deps = self.create_mock_dependencies()
        predictor = AutoAnalysisPredictor(**deps)
        result = predictor.predict([])
        assert result == []

    def test_predict_with_no_hits(self):
        """Test predict method when search results have no hits."""
        deps = self.create_mock_dependencies()
        predictor = AutoAnalysisPredictor(**deps)

        # Mock featurizer to return empty data
        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([], [])
        mock_featurizer.get_used_model_info.return_value = ["featurizer_info"]
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        search_results = [
            (
                {"_source": {"message": "Some message", "merged_small_logs": ""}},
                {"hits": {"hits": []}},
            )
        ]
        result = predictor.predict(search_results)
        assert result == []

    def test_predict_result_structure(self):
        """Test that PredictionResult objects have correct structure."""
        deps = self.create_mock_dependencies()
        predictor = AutoAnalysisPredictor(**deps)

        # Mock featurizer to return test data
        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([[0.1, 0.2, 0.3]], ["456"])
        mock_featurizer.get_used_model_info.return_value = ["featurizer_info"]
        mock_featurizer.find_most_relevant_by_type.return_value = {
            "456": {
                "mrHit": {"_id": "log1", "_source": {"test_item": 456}},
                "compared_log": {"_source": {"message": "test"}},
                "original_position": 0,
            }
        }
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        search_results = create_test_search_results()
        results = predictor.predict(search_results)
        assert len(results) == 1

        result = results[0]
        assert_prediction_result_structure(
            result,
            "456",
            [0, 1, 3],
            [0.1, 0.2, 0.3],
            ["auto_analysis_model", "featurizer_info"],
        )

    def test_predict_probability_format(self):
        """Test that probability format is correct."""
        deps = self.create_mock_dependencies()
        predictor = AutoAnalysisPredictor(**deps)

        # Mock featurizer to return test data
        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([[0.1, 0.2, 0.3]], ["456"])
        mock_featurizer.get_used_model_info.return_value = ["featurizer_info"]
        mock_featurizer.find_most_relevant_by_type.return_value = {
            "456": {"mrHit": {"_id": "log1"}, "compared_log": {"_source": {"message": "test"}}, "original_position": 0}
        }
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        search_results = [
            (
                {"_source": {"message": "Error message", "merged_small_logs": ""}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": "Error message",
                                    "merged_small_logs": "",
                                    "test_item": 456,
                                },
                            }
                        ]
                    }
                },
            )
        ]
        results = predictor.predict(search_results)
        assert len(results) == 1

        probability = results[0].probability
        assert len(probability) == 2
        assert probability == [0.2, 0.8]  # From our mock

    def test_predict_multiple_predictions(self):
        """Test predict method with multiple predictions."""
        deps = self.create_mock_dependencies()
        predictor = AutoAnalysisPredictor(**deps)

        # Configure mocks for multiple results
        predictor.boosting_decision_maker.predict = Mock(return_value=([1, 0], [[0.2, 0.8], [0.7, 0.3]]))

        # Mock featurizer to return multiple items
        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], ["456", "789"])
        mock_featurizer.get_used_model_info.return_value = ["featurizer_info"]
        mock_featurizer.find_most_relevant_by_type.return_value = {
            "456": {
                "mrHit": {"_id": "log1"},
                "compared_log": {"_source": {"message": "test1"}},
                "original_position": 0,
            },
            "789": {
                "mrHit": {"_id": "log2"},
                "compared_log": {"_source": {"message": "test2"}},
                "original_position": 1,
            },
        }
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        search_results = [
            (
                {"_source": {"message": "Error message", "merged_small_logs": ""}},
                {"hits": {"hits": [{"_source": {"test_item": 456}}, {"_source": {"test_item": 789}}]}},
            )
        ]
        results = predictor.predict(search_results)
        assert len(results) == 2

        # Verify both results
        identities = {result.identity for result in results}
        assert identities == {"456", "789"}

        # Verify different labels
        labels = [result.label for result in results]
        assert 1 in labels and 0 in labels


class TestSuggestionPredictor:
    """Test cases for the SuggestionPredictor class."""

    def create_mock_dependencies(self):
        """Create mock dependencies for SuggestionPredictor."""
        mock_model_chooser = Mock()
        mock_boosting_decision_maker = Mock()
        mock_defect_type_model = Mock()

        # Configure model chooser to return our mocks
        mock_model_chooser.choose_model.side_effect = lambda project_id, model_type, **kwargs: (
            mock_boosting_decision_maker if model_type == ModelType.suggestion else mock_defect_type_model
        )

        # Configure basic properties
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
        # Remove optional parameters to test defaults
        del deps["custom_model_prob"]
        del deps["hash_source"]

        predictor = SuggestionPredictor(**deps)

        assert predictor.boosting_config == {"suggestion": "config"}

        # Verify model chooser was called correctly with defaults
        deps["model_chooser"].choose_model.assert_any_call(
            123, ModelType.suggestion, custom_model_prob=0.0, hash_source=None
        )
        deps["model_chooser"].choose_model.assert_any_call(123, ModelType.defect_type, custom_model_prob=0.0)

    def test_predictor_instantiation_with_kwargs(self):
        """Test SuggestionPredictor instantiation with keyword arguments."""
        deps = self.create_mock_dependencies()
        predictor = SuggestionPredictor(**deps)

        assert predictor.boosting_config == {"suggestion": "config"}

        # Verify model chooser was called with custom parameters
        deps["model_chooser"].choose_model.assert_any_call(
            123, ModelType.suggestion, custom_model_prob=0.2, hash_source="suggestion_hash"
        )

    def test_predictor_in_prediction_classes(self):
        """Test that SuggestionPredictor is properly registered in PREDICTION_CLASSES."""
        assert "suggestion" in PREDICTION_CLASSES
        assert PREDICTION_CLASSES["suggestion"] == SuggestionPredictor

    def test_model_type_property(self):
        """Test that model_type property returns correct value."""
        deps = self.create_mock_dependencies()
        predictor = SuggestionPredictor(**deps)
        assert predictor.model_type == ModelType.suggestion

    def test_create_featurizer(self):
        """Test create_featurizer method."""
        deps = self.create_mock_dependencies()
        predictor = SuggestionPredictor(**deps)

        search_results = [({"_source": {"message": "test"}}, {"hits": {"hits": []}})]

        featurizer = predictor.create_featurizer(search_results)

        # Verify it's the right type and has expected configuration
        from app.ml.suggest_boosting_featurizer import SuggestBoostingFeaturizer

        assert isinstance(featurizer, SuggestBoostingFeaturizer)

    def test_predict_with_empty_input(self):
        """Test predict method with empty search results."""
        deps = self.create_mock_dependencies()
        predictor = SuggestionPredictor(**deps)
        result = predictor.predict([])
        assert result == []

    def test_predict_with_no_hits(self):
        """Test predict method when search results have no hits."""
        deps = self.create_mock_dependencies()
        predictor = SuggestionPredictor(**deps)

        # Mock featurizer to return empty data
        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([], [])
        mock_featurizer.get_used_model_info.return_value = ["featurizer_info"]
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        search_results = [
            (
                {"_source": {"message": "Some message", "merged_small_logs": ""}},
                {"hits": {"hits": []}},
            )
        ]
        result = predictor.predict(search_results)
        assert result == []

    def test_predict_result_structure(self):
        """Test that PredictionResult objects have correct structure."""
        deps = self.create_mock_dependencies()
        predictor = SuggestionPredictor(**deps)

        # Mock featurizer to return test data
        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([[0.4, 0.5, 0.6]], ["789"])
        mock_featurizer.get_used_model_info.return_value = ["suggestion_featurizer_info"]
        mock_featurizer.find_most_relevant_by_type.return_value = {
            "789": {
                "mrHit": {"_id": "log2", "_source": {"test_item": 789}},
                "compared_log": {"_source": {"message": "suggestion_test"}},
                "original_position": 0,
            }
        }
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        search_results = create_test_search_results(
            message="Suggestion message", log_id="log2", test_item=789, issue_type="pb002"
        )
        results = predictor.predict(search_results)
        assert len(results) == 1

        result = results[0]
        assert_prediction_result_structure(
            result,
            "789",
            [0, 1, 3],
            [0.4, 0.5, 0.6],
            ["suggestion_model", "suggestion_featurizer_info"],
        )

    def test_predict_probability_format(self):
        """Test that probability format is correct."""
        deps = self.create_mock_dependencies()
        predictor = SuggestionPredictor(**deps)

        # Mock featurizer to return test data
        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([[0.4, 0.5, 0.6]], ["789"])
        mock_featurizer.get_used_model_info.return_value = ["featurizer_info"]
        mock_featurizer.find_most_relevant_by_type.return_value = {
            "789": {"mrHit": {"_id": "log2"}, "compared_log": {"_source": {"message": "test"}}, "original_position": 0}
        }
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        search_results = [
            (
                {"_source": {"message": "Suggestion message", "merged_small_logs": ""}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_source": {
                                    "message": "Suggestion message",
                                    "merged_small_logs": "",
                                    "test_item": 789,
                                },
                            }
                        ]
                    }
                },
            )
        ]
        results = predictor.predict(search_results)
        assert len(results) == 1

        probability = results[0].probability
        assert len(probability) == 2
        assert probability == [0.3, 0.7]  # From our mock

    def test_predict_multiple_predictions(self):
        """Test predict method with multiple predictions."""
        deps = self.create_mock_dependencies()
        predictor = SuggestionPredictor(**deps)

        # Configure mocks for multiple results
        predictor.boosting_decision_maker.predict = Mock(return_value=([0, 1], [[0.6, 0.4], [0.1, 0.9]]))

        # Mock featurizer to return multiple items
        mock_featurizer = Mock()
        mock_featurizer.gather_features_info.return_value = ([[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]], ["789", "101"])
        mock_featurizer.get_used_model_info.return_value = ["featurizer_info"]
        mock_featurizer.find_most_relevant_by_type.return_value = {
            "789": {
                "mrHit": {"_id": "log2"},
                "compared_log": {"_source": {"message": "test1"}},
                "original_position": 0,
            },
            "101": {
                "mrHit": {"_id": "log3"},
                "compared_log": {"_source": {"message": "test2"}},
                "original_position": 1,
            },
        }
        predictor.create_featurizer = Mock(return_value=mock_featurizer)

        search_results = [
            (
                {"_source": {"message": "Suggestion message", "merged_small_logs": ""}},
                {"hits": {"hits": [{"_source": {"test_item": 789}}, {"_source": {"test_item": 101}}]}},
            )
        ]
        results = predictor.predict(search_results)
        assert len(results) == 2

        # Verify both results
        identities = {result.identity for result in results}
        assert identities == {"789", "101"}

        # Verify different labels
        labels = [result.label for result in results]
        assert 1 in labels and 0 in labels
