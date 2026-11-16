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

"""Filtering of found Test Items by text similarity of their matched logs to the request logs.

`more_like_this` builds its query only from request terms present in the index, so a found log may match a request
log by a single common term, e.g. "SPECIALNUMBER". The filter compares the whole texts of every matched pair of a
request log and a found log, which are taken from the named inner hits of a found Test Item.
"""

from collections import defaultdict
from typing import Any, Collection, Optional

from app.commons.model.db import Hit
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.query_builder import parse_log_inner_hits_name
from app.commons.similarity_calculator import SimilarityCalculator

UNSHORTENED_MESSAGE_FIELDS = [
    "detected_message_extended",
    "detected_message_without_params_extended",
    "detected_message_without_params_and_brackets",
]

SHORTENED_MESSAGE_FIELDS = [
    "message_extended",
    "message_without_params_extended",
    "message_without_params_and_brackets",
]


def choose_message_fields(number_of_log_lines: int) -> list[str]:
    """Choose log message fields to compare, according to the number of log lines used in the search.

    :param number_of_log_lines: Number of log lines to use, -1 means all lines
    :return: Log field names
    """
    return list(UNSHORTENED_MESSAGE_FIELDS if number_of_log_lines == -1 else SHORTENED_MESSAGE_FIELDS)


def _get_text(log: LogData, field: str) -> str:
    return (getattr(log, field, None) or "").strip()


class SimilarityFilter:
    """Filter of found Test Items by text similarity of their matched logs to the request logs.

    A matched log pair is similar if the similarity by any field (or by every field, if `all_fields` is set) reaches
    the minimum. Fields which are empty in both logs are not compared; a pair with nothing to compare is not similar.
    A found Test Item is kept if it has at least one similar pair, and only similar found logs are left in its inner
    hits.
    """

    fields: list[str]
    min_similarity: float
    all_fields: bool
    similarity_calculator: SimilarityCalculator

    def __init__(
        self,
        fields: list[str],
        min_similarity: float,
        *,
        all_fields: bool,
        similarity_calculator: Optional[SimilarityCalculator] = None,
    ) -> None:
        self.fields = fields
        self.min_similarity = min_similarity
        self.all_fields = all_fields
        self.similarity_calculator = similarity_calculator or SimilarityCalculator()

    def _get_log_matches(
        self, hit: Hit[TestItemIndexData], logs_number: int
    ) -> dict[str, tuple[int, list[tuple[dict[str, Any], LogData]]]]:
        matches: dict[str, tuple[int, list[tuple[dict[str, Any], LogData]]]] = {}
        for name, group in (hit.inner_hits or {}).items():
            log_index = parse_log_inner_hits_name(name)
            if log_index is None or log_index >= logs_number:
                continue
            raw_hits = (group or {}).get("hits", {}).get("hits", [])
            matches[name] = (log_index, [(raw_hit, Hit[LogData].from_dict(raw_hit).source) for raw_hit in raw_hits])
        return matches

    def _calculate_similarities(
        self,
        request_logs: list[LogData],
        all_matches: list[dict[str, tuple[int, list[tuple[dict[str, Any], LogData]]]]],
    ) -> None:
        """Calculate similarities of all matched pairs in one pass per request log and field, results are cached."""
        found_texts: dict[tuple[int, str], set[str]] = defaultdict(set)
        for matches in all_matches:
            for log_index, found_logs in matches.values():
                for _, found_log in found_logs:
                    for field in self.fields:
                        found_texts[(log_index, field)].add(_get_text(found_log, field))
        for (log_index, field), texts in found_texts.items():
            self.similarity_calculator.find_similarity(_get_text(request_logs[log_index], field), sorted(texts))

    def is_similar(self, request_log: LogData, found_log: LogData) -> bool:
        """Check if a found log is similar to the request log.

        :param request_log: Request log
        :param found_log: Found log
        :return: True if the logs are similar
        """
        results: list[bool] = []
        for field in self.fields:
            request_text = _get_text(request_log, field)
            found_text = _get_text(found_log, field)
            if not request_text and not found_text:
                continue
            similarity = self.similarity_calculator.find_similarity(request_text, [found_text])[0].similarity
            results.append(similarity >= self.min_similarity)
        if not results:
            return False
        return all(results) if self.all_fields else any(results)

    def filter(
        self,
        search_results: tuple[TestItemIndexData, list[Hit[TestItemIndexData]]],
        required_log_indices: Optional[Collection[int]] = None,
    ) -> tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]:
        """Remove found Test Items without logs similar to the request logs.

        :param search_results: Request Test Item and Test Items found for it
        :param required_log_indices: Positions of request logs every one of which must have a similar found log, if
                                     not set, one similar log is enough
        :return: Request Test Item and copies of the kept found Test Items, with only similar logs in inner hits
        """
        request_item, hits = search_results
        request_logs = request_item.logs or []
        all_matches = [self._get_log_matches(hit, len(request_logs)) for hit in hits]
        self._calculate_similarities(request_logs, all_matches)

        required = set(required_log_indices or [])
        filtered_hits: list[Hit[TestItemIndexData]] = []
        for hit, matches in zip(hits, all_matches):
            inner_hits = {
                name: group
                for name, group in (hit.inner_hits or {}).items()
                if parse_log_inner_hits_name(name) is None
            }
            similar_log_indices: set[int] = set()
            for name, (log_index, found_logs) in matches.items():
                similar_raw_hits = [
                    raw_hit for raw_hit, found_log in found_logs if self.is_similar(request_logs[log_index], found_log)
                ]
                if not similar_raw_hits:
                    continue
                similar_log_indices.add(log_index)
                group = (hit.inner_hits or {}).get(name) or {}
                inner_hits[name] = {**group, "hits": {**group.get("hits", {}), "hits": similar_raw_hits}}
            if not similar_log_indices or not required <= similar_log_indices:
                continue
            filtered_hits.append(hit.model_copy(update={"inner_hits": inner_hits}))
        return request_item, filtered_hits
