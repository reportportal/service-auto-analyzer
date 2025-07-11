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

import pytest

from app.machine_learning.predictor import (
    PREDICTION_CLASSES,
    PredictionResult,
    SimilarityPredictor,
    extract_text_fields_for_comparison,
)


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
        search_results = [
            (
                {"_source": {"message": "Error message", "merged_small_logs": ""}},
                {
                    "hits": {
                        "hits": [
                            {
                                "_id": "log1",
                                "_source": {
                                    "message": "Error message",
                                    "merged_small_logs": "",
                                    "test_item": 456,
                                    "issue_type": "pb001",
                                },
                            }
                        ]
                    }
                },
            )
        ]
        results = predictor.predict(search_results)
        assert len(results) == 1

        result = results[0]
        assert isinstance(result, PredictionResult)
        assert isinstance(result.label, int)
        assert isinstance(result.probability, list)
        assert len(result.probability) == 2
        assert result.probability[0] + result.probability[1] == pytest.approx(1.0)
        assert isinstance(result.data, dict)
        assert "mrHit" in result.data
        assert "compared_log" in result.data
        assert result.identity == "456"
        assert result.feature_info is not None
        assert result.feature_info.feature_ids == [0]
        assert result.feature_info.feature_data == [result.probability[1]]
        assert result.model_info_tags == ["similarity_predictor"]
        assert result.original_position == 0

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
            ("Connection timeout error", "", "Connection failed error", "", (0.2, 0.4)),
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
