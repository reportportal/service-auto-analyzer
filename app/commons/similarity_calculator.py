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
from app.utils import text_processing


class SimilarityCalculator:
    """Calculate text similarity between a base text and other texts, caching results by text pair."""

    __similarity_dict: dict[tuple[str, str], SimilarityResult]

    def __init__(self) -> None:
        self.__similarity_dict = {}

    def find_similarity(self, base_text: str, other_texts: list[str]) -> list[SimilarityResult]:
        """Calculate similarity of the base text against every other text.

        :param base_text: Text to compare from
        :param other_texts: Texts to compare against
        :return: Similarity results aligned with `other_texts`
        """
        missing_texts = list(
            dict.fromkeys(text for text in other_texts if (base_text, text) not in self.__similarity_dict)
        )
        if missing_texts:
            for text, result in zip(
                missing_texts, text_processing.calculate_text_similarity(base_text, missing_texts)
            ):
                self.__similarity_dict[(base_text, text)] = result
        return [self.__similarity_dict[(base_text, text)] for text in other_texts]
