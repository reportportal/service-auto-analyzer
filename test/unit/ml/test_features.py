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

from statistics import mean
from typing import Any, Optional
from unittest.mock import Mock

import pytest

from app.commons.model.db import Hit
from app.commons.model.test_item_index import LogData, TestItemHistoryData, TestItemIndexData
from app.commons.similarity_calculator import SimilarityCalculator
from app.ml import features
from app.ml.features import (
    BaseIssueTypeFeature,
    DefectTypeFeature,
    HistoryStabilityFeature,
    HistoryUnchangedFeature,
    IdentifierJaccardFeature,
    IssueTypeConsensusFeature,
    IssueTypeGroupEqualityFeature,
    IssueTypeShareFeature,
    ItemFieldEqualityFeature,
    ItemFieldSimilarityFeature,
    LaunchNumberDistanceFeature,
    LogCountFeature,
    LogCoverageCalculator,
    LogCoverageFeature,
    LogFieldEqualityFeature,
    LogFieldSimilarityFeature,
    ManuallyAnalyzedFeature,
    RequestFieldPresentFeature,
    RequestIdentifiersPresentFeature,
    SeveralLogsFeature,
    StacktraceFeature,
    TimeDecayFeature,
    ValuesSimilarityFeature,
)
from app.utils import text_processing


def build_log(log_id: str, log_order: Optional[int] = None, **fields: Any) -> LogData:
    return LogData(log_id=log_id, log_order=log_order, log_level=40000, **fields)


def build_item(test_item_id: str = "100", logs: Optional[list[LogData]] = None, **fields: Any) -> TestItemIndexData:
    fields.setdefault("launch_id", "1")
    return TestItemIndexData(test_item_id=test_item_id, logs=logs, **fields)


def build_hit(test_item: TestItemIndexData) -> Hit[TestItemIndexData]:
    return Hit[TestItemIndexData].from_dict(
        {"_id": test_item.test_item_id, "_score": 1.0, "_source": test_item.model_dump()}
    )


def build_typed_hits(*issue_types: str) -> list[Hit[TestItemIndexData]]:
    return [build_hit(build_item(str(idx), issue_type=issue_type)) for idx, issue_type in enumerate(issue_types)]


def build_history(*issue_types: str) -> list[TestItemHistoryData]:
    return [
        TestItemHistoryData(test_item_id="1", is_auto_analyzed=False, issue_type=issue_type, timestamp="2026-01-01")
        for issue_type in issue_types
    ]


def logs_with(field: str, *values: str) -> list[LogData]:
    return [build_log(str(idx), idx, **{field: value}) for idx, value in enumerate(values)]


@pytest.mark.parametrize(
    "first, second, lowercase, expected",
    [
        (None, None, False, 0.0),
        ("", "", False, 0.0),
        ("  ", "", False, 0.0),
        (None, "a", False, 0.0),
        ("a", "a", False, 1.0),
        ("a", "b", False, 0.0),
        ("Launch ", "launch", True, 1.0),
        ("Launch", "launch", False, 0.0),
        (123, 123, False, 1.0),
    ],
)
def test_are_equal(first, second, lowercase: bool, expected: float):
    assert features.are_equal(first, second, lowercase=lowercase) == expected


def test_join_log_field_uses_log_order():
    test_item = build_item(
        logs=[build_log("3", 1, message="second"), build_log("1", 0, message="first"), build_log("2", 2, message=" ")]
    )

    assert test_item.join_log_field("message") == "first\nsecond"
    assert test_item.join_log_field("message", " ") == "first second"


def test_sorted_logs_fall_back_to_log_id():
    test_item = build_item(logs=[build_log("20"), build_log("3")])

    assert [log.log_id for log in test_item.get_sorted_logs()] == ["3", "20"]


def test_defect_type_feature():
    model = Mock()
    model.predict.side_effect = lambda texts, issue_type: ([1, 0], [[0.4, 0.6], [0.9, 0.1]])
    request = build_item(
        logs=[
            build_log("2", 1, detected_message_without_params_extended="second"),
            build_log("1", 0, detected_message_without_params_extended="first"),
            build_log("3", 2, detected_message_without_params_extended=""),
        ]
    )

    values = DefectTypeFeature(model).calculate(request, build_typed_hits("pb001", "ab001", "pb001"))

    assert values == [0.6, 0.6, 0.6]
    assert model.predict.call_count == 2
    model.predict.assert_any_call(["first\nsecond"], "pb001")
    model.predict.assert_any_call(["first\nsecond"], "ab001")


def test_defect_type_feature_without_model_or_on_error():
    request = build_item(logs=[build_log("1", 0, detected_message_without_params_extended="text")])
    hits = build_typed_hits("pb001")
    model = Mock()
    model.predict.side_effect = KeyError("unknown")

    assert DefectTypeFeature(None).calculate(request, hits) == [0.0]
    assert DefectTypeFeature(model).calculate(request, hits) == [0.0]


def test_issue_type_share_feature():
    values = IssueTypeShareFeature().calculate(build_item(), build_typed_hits("pb001", "ab001", "PB001"))

    assert values == [pytest.approx(2 / 3), pytest.approx(1 / 3), pytest.approx(2 / 3)]


def test_log_field_similarity_feature():
    request = build_item(logs=logs_with("message", "database timeout", "connection refused"))
    hits = [
        build_hit(build_item("1", logs=logs_with("message", "database timeout", "connection refused"))),
        build_hit(build_item("2", logs=logs_with("message", "connection refused", "database timeout"))),
        build_hit(build_item("3", logs=logs_with("message", "assertion failed"))),
        build_hit(build_item("4", logs=[])),
    ]

    values = LogFieldSimilarityFeature("message", SimilarityCalculator()).calculate(request, hits)

    assert values[0] == 1.0
    # Logs are joined in their order, so the same logs in another order are less similar
    assert values[1] == pytest.approx(6 / 7)
    assert values[2] == 0.0
    assert values[3] == 0.0


def test_log_field_similarity_feature_both_empty():
    request = build_item(logs=logs_with("stacktrace", ""))
    hits = [build_hit(build_item("1", logs=logs_with("stacktrace", "")))]

    assert LogFieldSimilarityFeature("stacktrace", SimilarityCalculator()).calculate(request, hits) == [0.0]


def test_item_field_similarity_feature():
    request = build_item(test_item_name="login test")
    hits = [build_hit(build_item("1", test_item_name="login test")), build_hit(build_item("2"))]

    assert ItemFieldSimilarityFeature("test_item_name", SimilarityCalculator()).calculate(request, hits) == [1.0, 0.0]


@pytest.mark.parametrize("of_request, expected", [(False, [1.0, 0.0, 0.0]), (True, [1.0, 1.0, 1.0])])
def test_several_logs_feature(of_request: bool, expected: list[float]):
    request = build_item(logs=logs_with("message", "a", "b"))
    hits = [
        build_hit(build_item("1", logs=logs_with("message", "a", "b"))),
        build_hit(build_item("2", logs=logs_with("message", "a"))),
        build_hit(build_item("3")),
    ]

    assert SeveralLogsFeature(of_request).calculate(request, hits) == expected


def test_values_similarity_feature():
    request = build_item(logs=logs_with("only_numbers", "1 2", "3"))
    hits = [
        build_hit(build_item("1", logs=logs_with("only_numbers", "2 3 4"))),
        build_hit(build_item("2", logs=logs_with("only_numbers", ""))),
    ]

    assert ValuesSimilarityFeature("only_numbers").calculate(request, hits) == [pytest.approx(4 / 9), 0.0]


def test_manually_analyzed_feature():
    hits = [
        build_hit(build_item("1", is_auto_analyzed=False)),
        build_hit(build_item("2", is_auto_analyzed=True)),
        build_hit(build_item("3")),
    ]

    assert ManuallyAnalyzedFeature().calculate(build_item(), hits) == [1.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "field, request_value, hit_values, lowercase, expected",
    [
        ("test_case_hash", 123, [123, 321, 0, None], False, [1.0, 0.0, 0.0, 0.0]),
        ("test_case_hash", 0, [0, None], False, [0.0, 0.0]),
        ("launch_name", "Launch", ["launch", "other", None], True, [1.0, 0.0, 0.0]),
        ("launch_id", "10", ["10", "11"], False, [1.0, 0.0]),
    ],
)
def test_item_field_equality_feature(field, request_value, hit_values, lowercase, expected):
    request = build_item(**{field: request_value})
    hits = [build_hit(build_item(str(idx), **{field: value})) for idx, value in enumerate(hit_values)]

    assert ItemFieldEqualityFeature(field, lowercase=lowercase).calculate(request, hits) == expected


def test_issue_type_group_equality_feature():
    request = build_item(test_case_hash=123)
    hits = [
        build_hit(build_item("1", issue_type="pb001", test_case_hash=321)),
        build_hit(build_item("2", issue_type="ab001", test_case_hash=321)),
        build_hit(build_item("3", issue_type="pb001", test_case_hash=123)),
    ]

    assert IssueTypeGroupEqualityFeature("test_case_hash").calculate(request, hits) == [1.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "base_issue_type, expected", [("ab", [1.0, 0.0, 0.0, 0.0]), ("pb", [0.0, 1.0, 1.0, 0.0]), ("si", [0.0] * 4)]
)
def test_base_issue_type_feature(base_issue_type: str, expected: list[float]):
    hits = build_typed_hits("ab001", "pb_custom", "PB002", "nd001")

    assert BaseIssueTypeFeature(base_issue_type).calculate(build_item(), hits) == expected


def test_log_field_equality_feature():
    request = build_item(logs=logs_with("potential_status_codes", "400", "", "500"))
    hits = [
        build_hit(build_item("1", logs=logs_with("potential_status_codes", "400", "500"))),
        build_hit(build_item("2", logs=logs_with("potential_status_codes", "500", "400"))),
        build_hit(build_item("3", logs=logs_with("potential_status_codes", ""))),
    ]

    assert LogFieldEqualityFeature("potential_status_codes").calculate(request, hits) == [1.0, 0.0, 0.0]


def test_log_field_equality_feature_both_empty():
    request = build_item(logs=logs_with("urls", ""))
    hits = [build_hit(build_item("1", logs=logs_with("urls", "")))]

    assert LogFieldEqualityFeature("urls").calculate(request, hits) == [0.0]


def test_time_decay_feature():
    request = build_item(start_time="2026-01-15 10:00:00")
    hits = [
        build_hit(build_item("1", start_time="2026-01-15 01:00:00")),
        build_hit(build_item("2", start_time="2026-01-01 10:00:00")),
        build_hit(build_item("3", start_time="2026-01-29 10:00:00")),
        build_hit(build_item("4")),
    ]

    values = TimeDecayFeature(0.9).calculate(request, hits)

    assert values == [1.0, pytest.approx(0.81), pytest.approx(0.81), 0.0]


def test_log_count_feature():
    hits = [
        build_hit(build_item("1", logs=logs_with("message", "a", "b"))),
        build_hit(build_item("2", logs=logs_with("message", "a"))),
        build_hit(build_item("3", logs=[])),
        build_hit(build_item("4", log_count=1)),
    ]

    assert LogCountFeature().calculate(build_item(), hits) == [1.0, 0.5, 0.0, 0.5]
    assert LogCountFeature().calculate(build_item(), [build_hit(build_item("1"))]) == [0.0]


def test_launch_number_distance_feature():
    request = build_item(launch_name="Launch", launch_number="10")
    hits = [
        build_hit(build_item("1", launch_name="launch", launch_number="10")),
        build_hit(build_item("2", launch_name="Launch", launch_number="8")),
        build_hit(build_item("3", launch_name="Launch", launch_number="14")),
        build_hit(build_item("4", launch_name="Other", launch_number="100")),
        build_hit(build_item("5", launch_name="Launch")),
    ]

    values = LaunchNumberDistanceFeature().calculate(request, hits)

    assert values == [1.0, pytest.approx(0.55), pytest.approx(0.1), 0.0, 0.0]


def test_launch_number_distance_feature_same_numbers():
    request = build_item(launch_name="Launch", launch_number="10")
    hits = [build_hit(build_item("1", launch_name="Launch", launch_number="10"))]

    assert LaunchNumberDistanceFeature().calculate(request, hits) == [1.0]


@pytest.mark.parametrize(
    "history, stability, unchanged",
    [
        ([], 1.0, 1.0),
        (["pb001"], 1.0, 1.0),
        (["pb001", "pb001", "pb001"], 1.0, 1.0),
        (["pb001", "ab001", "pb001"], 0.5, 0.0),
        (["pb001", "ab001", "si001"], 0.0, 0.0),
    ],
)
def test_history_features(history: list[str], stability: float, unchanged: float):
    hits = [build_hit(build_item("1", issue_history=build_history(*history)))]

    assert HistoryStabilityFeature().calculate(build_item(), hits) == [stability]
    assert HistoryUnchangedFeature().calculate(build_item(), hits) == [unchanged]


def test_stacktrace_feature():
    hits = [
        build_hit(build_item("1", logs=[build_log("1", 0, stacktrace="at Foo.bar(Foo.java:10)")])),
        build_hit(build_item("2", logs=[build_log("1", 0, message="Traceback (most recent call last):")])),
        build_hit(build_item("3", logs=[build_log("1", 0, message="error")])),
        build_hit(build_item("4")),
    ]

    assert StacktraceFeature().calculate(build_item(), hits) == [1.0, 1.0, 0.0, 0.0]


def test_issue_type_consensus_feature():
    assert (
        IssueTypeConsensusFeature().calculate(build_item(), build_typed_hits("ab001", "pb001", "si001", "nd001"))
        == [pytest.approx(0.0)] * 4
    )
    assert (
        IssueTypeConsensusFeature().calculate(build_item(), build_typed_hits("ab001", "pb001"))
        == [pytest.approx(0.5)] * 2
    )
    assert IssueTypeConsensusFeature().calculate(build_item(), build_typed_hits("pb001", "pb002")) == [1.0, 1.0]


def test_identifier_features():
    request = build_item(logs=logs_with("message", "error at com.example.Service failed in getUser"))
    hits = [
        build_hit(build_item("1", logs=logs_with("message", "com.example.Service getUser"))),
        build_hit(build_item("2", logs=logs_with("message", "error at org.other.Client"))),
        build_hit(build_item("3", logs=logs_with("message", "plain words"))),
    ]
    plain_request = build_item(logs=logs_with("message", "plain words"))

    assert IdentifierJaccardFeature("message").calculate(request, hits) == [1.0, 0.0, 0.0]
    assert IdentifierJaccardFeature("message").calculate(plain_request, hits[2:]) == [0.0]
    assert RequestIdentifiersPresentFeature("message").calculate(request, hits) == [1.0] * 3
    assert RequestIdentifiersPresentFeature("message").calculate(plain_request, hits) == [0.0] * 3


def test_request_field_present_feature():
    hits = build_typed_hits("pb001", "ab001")

    assert RequestFieldPresentFeature("potential_status_codes").calculate(
        build_item(logs=logs_with("potential_status_codes", "", "404")), hits
    ) == [1.0, 1.0]
    assert RequestFieldPresentFeature("potential_status_codes").calculate(build_item(), hits) == [0.0, 0.0]


@pytest.mark.parametrize("reverse, expected", [(False, [0.5, 0.0]), (True, [1.0, 0.0])])
def test_log_coverage_feature(reverse: bool, expected: list[float]):
    request = build_item(logs=logs_with("message", "database timeout", "assertion failed"))
    hits = [
        build_hit(build_item("1", logs=logs_with("message", "database timeout"))),
        build_hit(build_item("2", logs=[])),
    ]

    assert LogCoverageFeature("message", reverse=reverse).calculate(request, hits) == expected


def calculate_coverage_per_hit(
    request: TestItemIndexData, hits: list[Hit[TestItemIndexData]], reverse: bool
) -> list[float]:
    """Log coverage calculated with one similarity call per found Test Item, as a reference."""
    request_texts = features.get_log_field_texts(request, "message")
    values = []
    for hit in hits:
        hit_texts = features.get_log_field_texts(hit.source, "message")
        if not request_texts or not hit_texts:
            values.append(0.0)
            continue
        base_texts, other_texts = (hit_texts, request_texts) if reverse else (request_texts, hit_texts)
        rows = text_processing.calculate_text_similarity_batch(base_texts, other_texts)
        values.append(features.clamp01(mean(max(result.similarity for result in row) for row in rows)))
    return values


COVERAGE_REQUEST_MESSAGES = (
    "java.lang.NullPointerException at com.example.Service.call",
    "Connection refused to database host",
    "Test failed with assertion error",
)
COVERAGE_HIT_MESSAGES = [
    ("java.lang.NullPointerException at com.example.Service.call",),
    ("Connection refused to database server", "Timeout waiting for element", "Test failed with assertion error"),
    (),
    ("completely unrelated text about weather",),
    ("Connection refused to database host", "Connection refused to database host"),
]


@pytest.mark.parametrize("reverse", [False, True])
def test_log_coverage_feature_matches_per_hit_calculation(reverse: bool):
    request = build_item(logs=logs_with("message", *COVERAGE_REQUEST_MESSAGES))
    hits = [
        build_hit(build_item(str(idx), logs=logs_with("message", *messages)))
        for idx, messages in enumerate(COVERAGE_HIT_MESSAGES)
    ]

    values = LogCoverageFeature("message", reverse=reverse).calculate(request, hits)

    assert values == pytest.approx(calculate_coverage_per_hit(request, hits, reverse), abs=1e-12)
    assert values[2] == 0.0, "a found Test Item without logs should have no coverage"


def test_log_coverage_features_share_one_similarity_calculation(monkeypatch):
    request = build_item(logs=logs_with("message", *COVERAGE_REQUEST_MESSAGES))
    hits = [
        build_hit(build_item(str(idx), logs=logs_with("message", *messages)))
        for idx, messages in enumerate(COVERAGE_HIT_MESSAGES)
    ]
    similarity_batch = Mock(wraps=text_processing.calculate_text_similarity_batch)
    monkeypatch.setattr(features.text_processing, "calculate_text_similarity_batch", similarity_batch)
    calculator = LogCoverageCalculator()
    forward_feature = LogCoverageFeature("message", reverse=False, coverage_calculator=calculator)
    reverse_feature = LogCoverageFeature("message", reverse=True, coverage_calculator=calculator)

    forward = forward_feature.calculate(request, hits)
    reverse = reverse_feature.calculate(request, hits)

    similarity_batch.assert_called_once()
    request_texts, hit_texts = similarity_batch.call_args.args
    assert request_texts == list(COVERAGE_REQUEST_MESSAGES)
    assert hit_texts == [message for messages in COVERAGE_HIT_MESSAGES for message in messages]
    assert len(forward) == len(hits)
    assert len(reverse) == len(hits)


def test_log_coverage_calculator_recalculates_for_another_request():
    calculator = LogCoverageCalculator()
    feature = LogCoverageFeature("message", reverse=False, coverage_calculator=calculator)
    hits = [build_hit(build_item("1", logs=logs_with("message", "database timeout")))]

    first = feature.calculate(build_item(logs=logs_with("message", "database timeout")), hits)
    second = feature.calculate(build_item(logs=logs_with("message", "assertion failed")), hits)

    assert first == [1.0]
    assert second == [0.0]


def test_log_coverage_feature_without_request_logs():
    hits = [build_hit(build_item("1", logs=logs_with("message", "database timeout")))]

    assert LogCoverageFeature("message", reverse=False).calculate(build_item(logs=[]), hits) == [0.0]
    assert LogCoverageFeature("message", reverse=True).calculate(build_item(logs=[]), hits) == [0.0]
