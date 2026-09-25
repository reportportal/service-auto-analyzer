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

from typing import Any

import pytest

from app.commons.model.db import Hit
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.query_builder import (
    INNER_HITS_SIZE,
    MIN_TERMS_PER_LOG,
    AutoAnalysisQueryBuilder,
    SuggestQueryBuilder,
    add_start_time_decay,
    best_log_match,
    extract_log_matches,
)
from test import DEFAULT_SEARCH_CONFIG

AA_FIELD = "detected_message_without_params_extended"


def build_log(log_id: str, message: str, **kwargs: Any) -> LogData:
    return LogData(
        log_id=log_id,
        log_level=40000,
        message=message,
        detected_message_without_params_extended=message,
        detected_message_extended=message,
        message_extended=message,
        **kwargs,
    )


def build_request_item(logs: list[LogData], **kwargs: Any) -> TestItemIndexData:
    return TestItemIndexData(test_item_id="100", launch_id="1", logs=logs, **kwargs)


def build_query(request_item: TestItemIndexData, search_cfg=DEFAULT_SEARCH_CONFIG, **kwargs: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "number_of_log_lines": -1,
        "min_should_match": "80%",
        "min_logs_to_match": "1",
        "filter_no_defect": True,
    }
    params.update(kwargs)
    return AutoAnalysisQueryBuilder(search_cfg).build(request_item, **params)


def get_log_clauses(query: dict[str, Any]) -> list[dict[str, Any]]:
    return query["query"]["bool"]["must"][0]["bool"]["should"]


def get_primary_mlt(log_clause: dict[str, Any]) -> dict[str, Any]:
    return log_clause["nested"]["query"]["bool"]["must"][0]["more_like_this"]


def build_hit(inner_hits: dict[str, Any]) -> Hit[TestItemIndexData]:
    return Hit[TestItemIndexData].from_dict(
        {
            "_id": "200",
            "_score": 5.0,
            "_source": {"test_item_id": "200", "launch_id": "2"},
            "inner_hits": inner_hits,
        }
    )


def build_inner_hits_group(*log_hits: tuple[str, float]) -> dict[str, Any]:
    return {
        "hits": {
            "hits": [
                {"_id": log_id, "_score": score, "_source": {"log_id": log_id, "log_level": 40000}}
                for log_id, score in log_hits
            ]
        }
    }


def test_one_named_clause_per_log_aligned_by_position():
    request_item = build_request_item(
        [build_log("1", "first error"), build_log("2", ""), build_log("3", "third error")]
    )

    log_clauses = get_log_clauses(build_query(request_item))

    assert [clause["nested"]["inner_hits"]["name"] for clause in log_clauses] == ["log_0", "log_2"]
    assert [get_primary_mlt(clause)["like"] for clause in log_clauses] == ["first error", "third error"]
    for clause in log_clauses:
        assert clause["nested"]["path"] == "logs"
        assert clause["nested"]["score_mode"] == "max"
        assert clause["nested"]["inner_hits"]["size"] == INNER_HITS_SIZE


@pytest.mark.parametrize("min_logs_to_match", ["1", "100%"])
def test_min_logs_to_match(min_logs_to_match: str):
    request_item = build_request_item([build_log("1", "first error"), build_log("2", "second error")])

    query = build_query(request_item, min_logs_to_match=min_logs_to_match, min_should_match="90%")

    assert query["query"]["bool"]["must"][0]["bool"]["minimum_should_match"] == min_logs_to_match
    for clause in get_log_clauses(query):
        assert get_primary_mlt(clause)["minimum_should_match"] == "5<90%"


@pytest.mark.parametrize(
    "builder_class, number_of_log_lines, expected_field",
    [
        (AutoAnalysisQueryBuilder, -1, "detected_message_without_params_extended"),
        (AutoAnalysisQueryBuilder, 5, "message"),
        (SuggestQueryBuilder, -1, "detected_message_extended"),
        (SuggestQueryBuilder, 5, "message_extended"),
    ],
)
def test_primary_message_field(builder_class, number_of_log_lines: int, expected_field: str):
    request_item = build_request_item([build_log("1", "first error")])

    query = builder_class(DEFAULT_SEARCH_CONFIG).build(
        request_item,
        number_of_log_lines=number_of_log_lines,
        min_should_match="80%",
        min_logs_to_match="1",
        filter_no_defect=True,
    )

    assert get_primary_mlt(get_log_clauses(query)[0])["fields"] == [f"logs.{expected_field}"]


def test_log_clause_contains_exceptions_and_status_codes():
    request_item = build_request_item(
        [build_log("1", "status error", found_exceptions="AssertionError", potential_status_codes="400 401")]
    )

    nested_should = get_log_clauses(build_query(request_item))[0]["nested"]["query"]["bool"]["should"]

    assert len(nested_should) == 3
    assert nested_should[0]["more_like_this"]["fields"] == ["logs.found_exceptions"]
    assert nested_should[1]["term"]["logs.potential_status_codes.exact"]["value"] == "400 401"
    assert nested_should[2]["match_phrase"]["logs.potential_status_codes"]["query"] == "400 401"


@pytest.mark.parametrize(
    "logs_number, budget, expected_terms, expected_logs_number",
    [
        (2, 800, 50, 2),
        (20, 800, 40, 20),
        (20, 100, MIN_TERMS_PER_LOG, 10),
    ],
)
def test_terms_budget(logs_number: int, budget: int, expected_terms: int, expected_logs_number: int):
    search_cfg = DEFAULT_SEARCH_CONFIG.model_copy(update={"ItemQueryTermsBudget": budget, "MaxQueryTerms": 50})
    request_item = build_request_item([build_log(str(idx), f"error number {idx}") for idx in range(logs_number)])

    log_clauses = get_log_clauses(build_query(request_item, search_cfg=search_cfg))

    assert len(log_clauses) == expected_logs_number
    expected_names = [f"log_{idx}" for idx in range(logs_number - expected_logs_number, logs_number)]
    assert [clause["nested"]["inner_hits"]["name"] for clause in log_clauses] == expected_names
    assert {get_primary_mlt(clause)["max_query_terms"] for clause in log_clauses} == {expected_terms}


def test_item_level_clauses_are_added_once():
    request_item = build_request_item(
        [build_log("1", "first error"), build_log("2", "second error"), build_log("3", "third error")],
        test_case_hash=123,
        test_item_name="login test",
    )

    query = build_query(request_item, exclude_issue_type="pb001")
    bool_query = query["query"]["bool"]

    assert bool_query["must_not"].count({"term": {"test_item_id": "100"}}) == 1
    assert bool_query["must_not"].count({"term": {"issue_type": "pb001"}}) == 1
    test_case_hash_clauses = [clause for clause in bool_query["should"] if "test_case_hash" in clause.get("term", {})]
    assert len(test_case_hash_clauses) == 1
    name_clauses = [clause for clause in bool_query["should"] if "more_like_this" in clause]
    assert len(name_clauses) == 1
    assert name_clauses[0]["more_like_this"]["fields"] == ["test_item_name"]
    aa_ma_clauses = [clause for clause in bool_query["should"] if "is_auto_analyzed" in clause.get("term", {})]
    assert len(aa_ma_clauses) == 1


def test_identity_clauses_are_skipped_when_empty():
    request_item = build_request_item([build_log("1", "first error")])

    should = build_query(request_item)["query"]["bool"]["should"]

    assert len(should) == 1
    assert "is_auto_analyzed" in should[0]["term"]


@pytest.mark.parametrize("filter_no_defect, expected_restrictions", [(True, 2), (False, 1)])
def test_issue_type_restrictions(filter_no_defect: bool, expected_restrictions: int):
    request_item = build_request_item([build_log("1", "first error")])

    must_not = build_query(request_item, filter_no_defect=filter_no_defect)["query"]["bool"]["must_not"]

    assert len([clause for clause in must_not if "wildcard" in clause]) == expected_restrictions


def test_no_logs_to_search_by():
    assert build_query(build_request_item([build_log("1", "  ")])) == {}
    assert build_query(build_request_item([])) == {}


def test_add_start_time_decay():
    query = build_query(build_request_item([build_log("1", "first error")]))

    assert add_start_time_decay(query, None, 0.95) is query
    decayed = add_start_time_decay(query, "2026-01-01 00:00:00", 0.95)
    function_score = decayed["query"]["function_score"]
    assert function_score["query"] == query["query"]
    assert function_score["functions"][0]["exp"]["start_time"]["origin"] == "2026-01-01 00:00:00"
    assert decayed["_source"] == query["_source"]
    assert decayed["size"] == query["size"]


def test_extract_log_matches():
    hit = build_hit(
        {
            "log_2": build_inner_hits_group(("21", 3.0), ("22", 1.0)),
            "log_0": build_inner_hits_group(("01", 2.0)),
            "log_1": build_inner_hits_group(),
            "other": build_inner_hits_group(("99", 9.0)),
        }
    )

    matches = extract_log_matches(hit)

    assert list(matches.keys()) == [0, 2]
    assert [log_hit.source.log_id for log_hit in matches[0]] == ["01"]
    assert [log_hit.source.log_id for log_hit in matches[2]] == ["21", "22"]


def test_best_log_match():
    hit = build_hit(
        {
            "log_0": build_inner_hits_group(("01", 2.0)),
            "log_1": build_inner_hits_group(("11", 4.0), ("12", 1.0)),
        }
    )

    log_match = best_log_match(hit)

    assert log_match is not None
    log_index, log_hit = log_match
    assert log_index == 1
    assert log_hit.source.log_id == "11"


def test_best_log_match_without_matches():
    assert best_log_match(build_hit({})) is None
