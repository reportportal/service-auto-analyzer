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

from app.commons.model.launch_objects import SimilarityResult
from app.commons.model.test_item_index import LogData
from app.utils import text_processing


class SimilarityCalculator:
    """Calculate text similarity by log fields between request logs and logs they are compared to.

    Results are cached by field, so one instance should be used for one set of log pairs.
    """

    __similarity_dict: dict[str, dict[tuple[str, str], SimilarityResult]]

    def __init__(self):
        self.__similarity_dict = {}

    @staticmethod
    def _find_similarity_for_field(
        all_results: list[tuple[LogData, list[LogData]]], field: str
    ) -> dict[tuple[str, str], SimilarityResult]:
        all_results_similarity: dict[tuple[str, str], SimilarityResult] = {}
        for request, results in all_results:
            group_ids = [(str(result.log_id), str(request.log_id)) for result in results]
            request_field = getattr(request, field, None) or ""
            result_fields = [getattr(result, field, None) or "" for result in results]
            similarity_results = text_processing.calculate_text_similarity(request_field, result_fields)
            for group_id, sim_result in zip(group_ids, similarity_results):
                all_results_similarity[group_id] = sim_result
        return all_results_similarity

    def find_similarity(
        self, all_results: list[tuple[LogData, list[LogData]]], fields: list[str]
    ) -> dict[str, dict[tuple[str, str], SimilarityResult]]:
        """Calculate similarity by the given fields.

        :param all_results: Request logs, each with the logs to compare it to
        :param fields: Log fields to compare by
        :return: Field name mapped to (compared log ID, request log ID) pairs mapped to similarity results
        """
        for field in fields:
            if field in self.__similarity_dict:
                continue
            self.__similarity_dict[field] = SimilarityCalculator._find_similarity_for_field(all_results, field)
        return self.__similarity_dict
