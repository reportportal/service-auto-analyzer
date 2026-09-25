#  Copyright 2026 EPAM Systems
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

from unittest import mock

from app.commons.similarity_calculator import SimilarityCalculator
from app.utils import text_processing


def test_find_similarity():
    calculator = SimilarityCalculator()

    results = calculator.find_similarity("database timeout", ["database timeout", "assertion failed", ""])

    assert [result.similarity for result in results] == [1.0, 0.0, 0.0]
    assert [result.both_empty for result in results] == [False, False, False]


def test_both_empty_texts():
    results = SimilarityCalculator().find_similarity("", [""])

    assert len(results) == 1
    assert results[0].similarity == 0.0
    assert results[0].both_empty


def test_results_are_cached_by_text_pair():
    calculator = SimilarityCalculator()

    with mock.patch.object(
        text_processing, "calculate_text_similarity", wraps=text_processing.calculate_text_similarity
    ) as similarity_mock:
        first = calculator.find_similarity("database timeout", ["database timeout", "assertion failed"])
        second = calculator.find_similarity("database timeout", ["assertion failed", "connection refused"])
        third = calculator.find_similarity("database timeout", ["database timeout", "database timeout"])

    assert similarity_mock.call_count == 2
    similarity_mock.assert_any_call("database timeout", ["database timeout", "assertion failed"])
    similarity_mock.assert_any_call("database timeout", ["connection refused"])
    assert [result.similarity for result in first] == [1.0, 0.0]
    assert [result.similarity for result in second] == [0.0, 0.0]
    assert [result.similarity for result in third] == [1.0, 1.0]
