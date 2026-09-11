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

import pytest

from app.utils import utils


@pytest.mark.parametrize("status_codes", ["", "   ", "\n"])
def test_no_clauses_without_codes(status_codes):
    assert utils.build_status_codes_exact_query(status_codes) is None
    assert utils.build_status_codes_queries(status_codes) == []


def test_exact_query_matches_the_whole_sequence():
    query = utils.build_status_codes_exact_query("400 401", field_name="logs.potential_status_codes", boost=8.0)

    assert query == {"term": {"logs.potential_status_codes.exact": {"value": "400 401", "boost": 8.0}}}


def test_exact_query_keeps_the_order_of_the_codes():
    """'expected 400, but was 401' must not match the inverse failure."""
    forward = utils.build_status_codes_exact_query("400 401")
    inverted = utils.build_status_codes_exact_query("401 400")

    assert forward != inverted


def test_exact_query_keeps_repeated_codes():
    """'expected 400, but was 400' is stored as two codes and stays distinct from one."""
    assert utils.build_status_codes_exact_query("400 400") != utils.build_status_codes_exact_query("400")


def test_queries_rank_exact_above_contained():
    exact, phrase = utils.build_status_codes_queries("400 401", boost=8.0)

    assert exact == {"term": {"potential_status_codes.exact": {"value": "400 401", "boost": 8.0}}}
    assert phrase == {"match_phrase": {"potential_status_codes": {"query": "400 401", "boost": 4.0}}}


def test_queries_use_no_term_overlap_matching():
    """`more_like_this` and `terms` score a code sequence the same as its permutations."""
    clauses = utils.build_status_codes_queries("400 401")

    assert not any("more_like_this" in clause or "terms" in clause for clause in clauses)
