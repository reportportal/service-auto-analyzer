#  Copyright 2024 EPAM Systems
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

from enum import Enum, auto
from typing import Iterable, Optional

from pydantic import BaseModel


class ModelType(Enum):
    defect_type = auto()
    suggestion = auto()
    auto_analysis = auto()


class ModelInfo(BaseModel):
    model_type: ModelType
    project: int


class TrainInfo(ModelInfo):
    additional_projects: Optional[Iterable[int]] = None
    gathered_metric_total: int = 0


class QueryResult(BaseModel):
    result: list[tuple[str, str, str]]
    error_count: int
    errors: list[str]
