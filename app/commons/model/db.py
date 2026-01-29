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
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Hit(BaseModel, Generic[T]):
    """Typed representation of an OpenSearch search hit."""

    index: Optional[str] = Field(default=None, validation_alias="_index")
    id: Optional[str] = Field(default=None, validation_alias="_id")
    score: Optional[float] = Field(default=None, validation_alias="_score")
    source: T = Field(validation_alias="_source")
    normalized_score: float = 0.0
    sort: Optional[list[Any]] = None
    highlight: Optional[dict[str, Any]] = None
    fields: Optional[dict[str, Any]] = None
    inner_hits: Optional[dict[str, Any]] = Field(default=None)

    @classmethod
    def from_dict(cls, hit: dict[str, Any]) -> "Hit[T]":
        """
        Build a typed Hit object from OpenSearch raw hit using pydantic validation.

        :param hit: Raw hit dictionary returned by OpenSearch
        :return: Hit instance with mapped source
        """
        return cls.model_validate(hit)
