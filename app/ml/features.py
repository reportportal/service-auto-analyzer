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

"""Gradient Boosting features, calculated for every found Test Item against the request Test Item.

Features use only Test Item data, never search engine artifacts like scores or inner hits, so trained models do not
depend on a search engine. Every feature value is a float in the range [0.0, 1.0]. Equality of two empty values
(None or blank strings) is 0.0: nothing to compare is never a match.
"""

import math
import re
from collections import defaultdict
from datetime import datetime
from statistics import mean
from typing import Any, Callable, Optional

from app.commons import logging
from app.commons.model.db import Hit
from app.commons.model.test_item_index import TestItemIndexData
from app.commons.similarity_calculator import SimilarityCalculator
from app.ml.models.defect_type_model import DATA_FIELD, DefectTypeModel
from app.utils import text_processing
from app.utils.utils import normalize_issue_type

logger = logging.getLogger("analyzerApp.features")

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_TIME_WEIGHT_DECAY = 0.95
MIN_LAUNCH_DISTANCE_VALUE = 0.1
TRACEBACK_MARKER = "Traceback"
BASE_ISSUE_TYPES = ["ab", "pb", "si", "nd"]
IDENTIFIER_TOKEN_PATTERN = re.compile(r"\.|::|[a-z][A-Z]")


def clamp01(value: float) -> float:
    """Bring the value into the range [0.0, 1.0], NaN and infinity become 0.0.

    :param value: Value to clamp
    :return: Clamped value
    """
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def is_empty(value: Any) -> bool:
    """Check if the value has nothing to compare: None or a blank string.

    :param value: Value to check
    :return: True if the value is empty
    """
    return value is None or (isinstance(value, str) and not value.strip())


def are_equal(first: Any, second: Any, *, lowercase: bool = False) -> float:
    """Check values for equality, where empty values are never equal.

    :param first: First value
    :param second: Second value
    :param lowercase: Compare strings case-insensitively
    :return: 1.0 if values are equal and not empty, 0.0 otherwise
    """
    if is_empty(first) or is_empty(second):
        return 0.0
    if isinstance(first, str) and isinstance(second, str):
        first, second = first.strip(), second.strip()
        if lowercase:
            first, second = first.lower(), second.lower()
    return float(first == second)


def inverse_positions(hits_number: int) -> list[float]:
    """Calculate inverse relative positions: 1.0 for the top result, 0.0 for the last one.

    :param hits_number: Number of results
    :return: Inverse relative position of every result
    """
    if hits_number == 1:
        return [1.0]
    return [1.0 - position / (hits_number - 1) for position in range(hits_number)]


def group_by_issue_type(hits: list[Hit[TestItemIndexData]]) -> dict[str, list[int]]:
    """Group results by their issue types.

    :param hits: Found Test Items
    :return: Issue type mapped to positions of results with this issue type
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for position, hit in enumerate(hits):
        groups[normalize_issue_type(hit.source.issue_type)].append(position)
    return dict(groups)


def get_base_issue_type(issue_type: Optional[str]) -> str:
    """Get the generic issue type, e.g. "pb" for "pb001" or "pb_custom".

    :param issue_type: Issue type
    :return: Generic issue type
    """
    return normalize_issue_type(issue_type)[:2]


def get_logs_number(test_item: TestItemIndexData) -> int:
    """Get the number of logs of the Test Item.

    :param test_item: Test Item
    :return: Number of logs
    """
    if test_item.logs is not None:
        return len(test_item.logs)
    return test_item.log_count or 0


def get_history_issue_types(test_item: TestItemIndexData) -> list[str]:
    """Get issue types from the Test Item history.

    :param test_item: Test Item
    :return: Issue types in the history order
    """
    issue_types = [normalize_issue_type(entry.issue_type) for entry in test_item.issue_history or []]
    return [issue_type for issue_type in issue_types if issue_type]


def parse_date(value: Optional[str]) -> Optional[datetime]:
    """Parse a date in the index format.

    :param value: Date string
    :return: Parsed date, or None if the value is empty or has wrong format
    """
    if not value:
        return None
    try:
        return datetime.strptime(value, DATE_FORMAT)
    except ValueError:
        return None


def parse_int(value: Any) -> Optional[int]:
    """Parse an integer.

    :param value: Value to parse
    :return: Parsed integer, or None if the value is not an integer
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_item_value(test_item: TestItemIndexData, field: str) -> Any:
    """Get a Test Item field value, where 0 integers are treated as empty values (e.g. unset Test Case Hash).

    :param test_item: Test Item
    :param field: Field name
    :return: Field value
    """
    value = getattr(test_item, field, None)
    if isinstance(value, int) and not isinstance(value, bool) and value == 0:
        return None
    return value


def extract_identifier_tokens(text: str) -> set[str]:
    """Extract identifier-like tokens: dotted paths, scope resolutions and camelCase names.

    :param text: Text to extract from
    :return: Identifier tokens
    """
    return {token for token in text.split() if IDENTIFIER_TOKEN_PATTERN.search(token)}


class PositionFeature:
    """Inverse relative position of the found Test Item among others: 1.0 - the top one, 0.0 - the last one."""

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return inverse_positions(len(hits))


class DefectTypeFeature:
    """Maximum probability among request logs to be of the found Test Item's issue type, by the Defect Type model."""

    model: Optional[DefectTypeModel]

    def __init__(self, model: Optional[DefectTypeModel]) -> None:
        self.model = model

    def _predict(self, texts: list[str], issue_type: str) -> float:
        if not self.model or not texts or not issue_type:
            return 0.0
        try:
            _, probabilities = self.model.predict(texts, issue_type)
        except Exception as exc:
            logger.exception(f"Unable to predict defect type '{issue_type}'", exc_info=exc)
            return 0.0
        return clamp01(max((probability[1] for probability in probabilities if len(probability) == 2), default=0.0))

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        texts = [str(getattr(log, DATA_FIELD, None) or "").strip() for log in request.get_sorted_logs()]
        texts = [text for text in texts if text]
        values_by_issue_type: dict[str, float] = {}
        values = []
        for hit in hits:
            issue_type = normalize_issue_type(hit.source.issue_type)
            if issue_type not in values_by_issue_type:
                values_by_issue_type[issue_type] = self._predict(texts, issue_type)
            values.append(values_by_issue_type[issue_type])
        return values


class IssueTypeShareFeature:
    """Share of found Test Items with the same issue type, the same value for the whole issue type group."""

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        values = [0.0] * len(hits)
        for positions in group_by_issue_type(hits).values():
            for position in positions:
                values[position] = len(positions) / len(hits)
        return values


class IssueTypePositionFeature:
    """Aggregated inverse relative position of found Test Items of the same issue type.

    The same value is set for the whole issue type group: "mean" - mean position, "max" - position of the group's top
    item, "min" - position of the group's last item.
    """

    AGGREGATIONS: dict[str, Callable[[list[float]], float]] = {"mean": mean, "max": max, "min": min}

    aggregation: Callable[[list[float]], float]

    def __init__(self, aggregation: str) -> None:
        self.aggregation = self.AGGREGATIONS[aggregation]

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        positions_values = inverse_positions(len(hits))
        values = [0.0] * len(hits)
        for positions in group_by_issue_type(hits).values():
            value = self.aggregation([positions_values[position] for position in positions])
            for position in positions:
                values[position] = value
        return values


class LogFieldSimilarityFeature:
    """Text similarity by a log field, where field values of all logs of a Test Item are joined into one text."""

    field: str
    similarity_calculator: SimilarityCalculator

    def __init__(self, field: str, similarity_calculator: SimilarityCalculator) -> None:
        self.field = field
        self.similarity_calculator = similarity_calculator

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        if not hits:
            return []
        request_text = request.join_log_field(self.field)
        texts = [hit.source.join_log_field(self.field) for hit in hits]
        return [
            clamp01(result.similarity) for result in self.similarity_calculator.find_similarity(request_text, texts)
        ]


class ItemFieldSimilarityFeature:
    """Text similarity by a Test Item field."""

    field: str
    similarity_calculator: SimilarityCalculator

    def __init__(self, field: str, similarity_calculator: SimilarityCalculator) -> None:
        self.field = field
        self.similarity_calculator = similarity_calculator

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        if not hits:
            return []
        request_text = str(getattr(request, self.field, None) or "")
        texts = [str(getattr(hit.source, self.field, None) or "") for hit in hits]
        return [
            clamp01(result.similarity) for result in self.similarity_calculator.find_similarity(request_text, texts)
        ]


class SeveralLogsFeature:
    """1.0 if the Test Item has several logs: the found one, or the request one if `of_request` is set."""

    of_request: bool

    def __init__(self, of_request: bool) -> None:
        self.of_request = of_request

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        if self.of_request:
            return [float(get_logs_number(request) > 1)] * len(hits)
        return [float(get_logs_number(hit.source) > 1) for hit in hits]


class ValuesSimilarityFeature:
    """Similarity by separate values of a log field, see `text_processing.calculate_similarity_by_values`."""

    field: str

    def __init__(self, field: str) -> None:
        self.field = field

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        request_values = request.join_log_field(self.field, " ")
        return [
            clamp01(
                text_processing.calculate_similarity_by_values(
                    request_values, hit.source.join_log_field(self.field, " ")
                )
            )
            for hit in hits
        ]


class ManuallyAnalyzedFeature:
    """1.0 if the found Test Item was analyzed manually."""

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return [float(hit.source.is_auto_analyzed is False) for hit in hits]


class ItemFieldEqualityFeature:
    """1.0 if a Test Item field is equal in the request and the found Test Items."""

    field: str
    lowercase: bool

    def __init__(self, field: str, lowercase: bool = False) -> None:
        self.field = field
        self.lowercase = lowercase

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        request_value = get_item_value(request, self.field)
        return [
            are_equal(request_value, get_item_value(hit.source, self.field), lowercase=self.lowercase) for hit in hits
        ]


class IssueTypeGroupEqualityFeature:
    """1.0 if any found Test Item of the issue type group has a Test Item field equal to the request one."""

    field: str

    def __init__(self, field: str) -> None:
        self.field = field

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        request_value = get_item_value(request, self.field)
        values = [0.0] * len(hits)
        for positions in group_by_issue_type(hits).values():
            value = max(
                are_equal(request_value, get_item_value(hits[position].source, self.field)) for position in positions
            )
            for position in positions:
                values[position] = value
        return values


class BaseIssueTypeFeature:
    """1.0 if the found Test Item's issue type is of the given generic type, e.g. "pb"."""

    base_issue_type: str

    def __init__(self, base_issue_type: str) -> None:
        self.base_issue_type = base_issue_type

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return [float(get_base_issue_type(hit.source.issue_type) == self.base_issue_type) for hit in hits]


class LogFieldEqualityFeature:
    """1.0 if a log field is equal, where field values of all logs of a Test Item are joined into one text."""

    field: str

    def __init__(self, field: str) -> None:
        self.field = field

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        request_text = request.join_log_field(self.field)
        return [are_equal(request_text, hit.source.join_log_field(self.field)) for hit in hits]


class TimeDecayFeature:
    """Exponential decay by weeks passed between start times: 1.0 for the same day, going down to 0.0."""

    decay_speed: float

    def __init__(self, time_weight_decay: float) -> None:
        self.decay_speed = math.log(time_weight_decay) if time_weight_decay > 0.0 else -math.inf

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        request_time = parse_date(request.start_time)
        values = []
        for hit in hits:
            hit_time = parse_date(hit.source.start_time)
            if request_time is None or hit_time is None or math.isinf(self.decay_speed):
                values.append(0.0)
                continue
            days = abs(request_time - hit_time).days
            values.append(clamp01(math.exp(self.decay_speed * days / 7)))
        return values


class LogCountFeature:
    """Number of logs relative to the found Test Item with the most logs."""

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        logs_numbers = [get_logs_number(hit.source) for hit in hits]
        max_logs_number = max(logs_numbers, default=0)
        if max_logs_number <= 0:
            return [0.0] * len(hits)
        return [logs_number / max_logs_number for logs_number in logs_numbers]


class LaunchNumberDistanceFeature:
    """Relative distance between launch numbers of launches with the same name.

    1.0 - the same launch number, 0.1 - the maximum distance among found Test Items, 0.0 - different launch names or
    unknown launch numbers.
    """

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        request_number = parse_int(request.launch_number)
        distances: list[Optional[int]] = []
        for hit in hits:
            hit_number = parse_int(hit.source.launch_number)
            same_launch = are_equal(request.launch_name, hit.source.launch_name, lowercase=True)
            if not same_launch or request_number is None or hit_number is None:
                distances.append(None)
            else:
                distances.append(abs(request_number - hit_number))
        max_distance = max((distance for distance in distances if distance is not None), default=0)
        values = []
        for distance in distances:
            if distance is None:
                values.append(0.0)
            elif max_distance == 0:
                values.append(1.0)
            else:
                values.append(1.0 - (1.0 - MIN_LAUNCH_DISTANCE_VALUE) * distance / max_distance)
        return values


class HistoryStabilityFeature:
    """Stability of the found Test Item's issue history: 1.0 - one issue type, 0.0 - every entry has its own type."""

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        values = []
        for hit in hits:
            issue_types = get_history_issue_types(hit.source)
            if len(issue_types) <= 1:
                values.append(1.0)
            else:
                values.append(1.0 - (len(set(issue_types)) - 1) / (len(issue_types) - 1))
        return values


class HistoryUnchangedFeature:
    """1.0 if the issue type never changed in the found Test Item's history."""

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return [float(len(set(get_history_issue_types(hit.source))) <= 1) for hit in hits]


class StacktraceFeature:
    """1.0 if any log of the found Test Item has a stack trace or a traceback."""

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return [
            float(
                any(
                    (log.stacktrace or "").strip() or TRACEBACK_MARKER in (log.message or "")
                    for log in hit.source.logs or []
                )
            )
            for hit in hits
        ]


class IssueTypeConsensusFeature:
    """Consensus of generic issue types among found Test Items, the same value for every one.

    Calculated as one minus normalized entropy: 1.0 - all found Test Items have one generic issue type, 0.0 - they
    are evenly spread across generic types.
    """

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        if not hits:
            return []
        counts: dict[str, int] = defaultdict(int)
        for hit in hits:
            counts[get_base_issue_type(hit.source.issue_type)] += 1
        probabilities = [count / len(hits) for count in counts.values()]
        entropy = -sum(probability * math.log(probability) for probability in probabilities)
        return [clamp01(1.0 - entropy / math.log(len(BASE_ISSUE_TYPES)))] * len(hits)


class IdentifierJaccardFeature:
    """Jaccard similarity of identifier-like tokens of a log field, 0.0 if neither side has identifiers.

    Identifiers (dotted paths, scope resolutions, camelCase names) tell apart failures which share boilerplate text but
    happen in different code.
    """

    field: str

    def __init__(self, field: str) -> None:
        self.field = field

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        request_tokens = extract_identifier_tokens(request.join_log_field(self.field))
        values = []
        for hit in hits:
            hit_tokens = extract_identifier_tokens(hit.source.join_log_field(self.field))
            union = request_tokens | hit_tokens
            values.append(len(request_tokens & hit_tokens) / len(union) if union else 0.0)
        return values


class RequestIdentifiersPresentFeature:
    """1.0 if the request Test Item has identifier-like tokens in a log field, the same value for every found item."""

    field: str

    def __init__(self, field: str) -> None:
        self.field = field

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return [float(bool(extract_identifier_tokens(request.join_log_field(self.field))))] * len(hits)


class RequestFieldPresentFeature:
    """1.0 if the request Test Item has a non-empty log field, the same value for every found item."""

    field: str

    def __init__(self, field: str) -> None:
        self.field = field

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        return [float(not is_empty(request.join_log_field(self.field)))] * len(hits)


class LogCoverageFeature:
    """Mean best similarity of logs of one Test Item against logs of the other one, by a log field.

    Forward coverage checks how well request logs are covered by the found Test Item's logs, reverse coverage checks
    how well the found Test Item's logs are covered by request logs.
    """

    field: str
    reverse: bool

    def __init__(self, field: str, reverse: bool) -> None:
        self.field = field
        self.reverse = reverse

    def _get_texts(self, test_item: TestItemIndexData) -> list[str]:
        texts = [str(getattr(log, self.field, None) or "").strip() for log in test_item.get_sorted_logs()]
        return [text for text in texts if text]

    def calculate(self, request: TestItemIndexData, hits: list[Hit[TestItemIndexData]]) -> list[float]:
        request_texts = self._get_texts(request)
        values = []
        for hit in hits:
            hit_texts = self._get_texts(hit.source)
            if not request_texts or not hit_texts:
                values.append(0.0)
                continue
            base_texts, other_texts = (hit_texts, request_texts) if self.reverse else (request_texts, hit_texts)
            similarities = text_processing.calculate_text_similarity_batch(base_texts, other_texts)
            values.append(clamp01(mean(max(result.similarity for result in row) for row in similarities)))
        return values
