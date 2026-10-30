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

"""Unit tests for utils module."""

import pytest

from app.commons.model.launch_objects import AnalyzerConf
from app.utils.utils import find_min_should_match_threshold
from test import DEFAULT_SEARCH_CONFIG


@pytest.mark.parametrize(
    "project_min_should_match, global_min_should_match, expected",
    [(95, "80%", 95), (0, "80%", 80), (0, "60%", 60)],
    ids=["project_setting", "global_default", "other_global_default"],
)
def test_find_min_should_match_threshold(project_min_should_match: int, global_min_should_match: str, expected: int):
    search_config = DEFAULT_SEARCH_CONFIG.model_copy(update={"MinShouldMatch": global_min_should_match})

    assert (
        find_min_should_match_threshold(AnalyzerConf(minShouldMatch=project_min_should_match), search_config)
        == expected
    )
