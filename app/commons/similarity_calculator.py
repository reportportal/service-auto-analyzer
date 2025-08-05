#  Copyright 2023 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Any

from app.utils import text_processing


class SimilarityCalculator:
    __similarity_dict: dict[str, dict]

    def __init__(self):
        self.__similarity_dict = {}

    @staticmethod
    def _find_similarity_for_field(
        all_results: list[tuple[dict[str, Any], dict[str, Any]]], field: str
    ) -> dict[tuple[str, str], dict[str, float | bool]]:
        all_results_similarity = {}
        for request, result in all_results:
            group_ids = [(str(obj["_id"]), str(request["_id"])) for obj in result["hits"]["hits"]]
            request_field = request["_source"].get(field, "")
            result_fields = [obj["_source"].get(field, "") for obj in result["hits"]["hits"]]
            similarity_results = text_processing.calculate_text_similarity(request_field, *result_fields)
            for group_id, (similarity, both_empty) in zip(group_ids, similarity_results):
                all_results_similarity[group_id] = {"similarity": similarity, "both_empty": both_empty}
        return all_results_similarity

    def find_similarity(
        self, all_results: list[tuple[dict[str, Any], dict[str, Any]]], fields: list[str]
    ) -> dict[str, dict[tuple[str, str], dict[str, float | bool]]]:
        for field in fields:
            if field in self.__similarity_dict:
                continue
            self.__similarity_dict[field] = SimilarityCalculator._find_similarity_for_field(all_results, field)
        return self.__similarity_dict
