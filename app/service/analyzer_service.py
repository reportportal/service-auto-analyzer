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

from app.commons import logging
from app.commons.model.launch_objects import AnalyzerConf, Launch, SearchConfig, TestItemInfo
from app.utils import utils

LOGGER = logging.getLogger("analyzerApp.analyzerService")


def _add_launch_name_boost(query: dict, launch_name: str, launch_boost: float) -> None:
    should = utils.create_path(query, ("query", "bool", "should"), [])
    should.append({"term": {"launch_name": {"value": launch_name, "boost": launch_boost}}})


def _add_launch_id_boost(query: dict, launch_id: int, launch_boost: float) -> None:
    should = utils.create_path(query, ("query", "bool", "should"), [])
    should.append({"term": {"launch_id": {"value": launch_id, "boost": launch_boost}}})


def _add_launch_name_and_id_boost(query: dict, launch_name: str, launch_id: int, launch_boost: float) -> None:
    _add_launch_id_boost(query, launch_id, launch_boost)
    _add_launch_name_boost(query, launch_name, launch_boost)


def add_constraints_for_launches_into_query(query: dict, launch: Launch, launch_boost: float) -> dict:
    previous_launch_id = getattr(launch, "previousLaunchId", 0) or 0
    previous_launch_id = int(previous_launch_id)
    analyzer_mode = launch.analyzerConfig.analyzerMode
    launch_name = launch.launchName
    launch_id = launch.launchId
    if analyzer_mode == "LAUNCH_NAME":
        # Previous launches with the same name
        must = utils.create_path(query, ("query", "bool", "must"), [])
        must_not = utils.create_path(query, ("query", "bool", "must_not"), [])
        must.append({"term": {"launch_name": launch_name}})
        must_not.append({"term": {"launch_id": launch_id}})
    elif analyzer_mode == "CURRENT_AND_THE_SAME_NAME":
        # All launches with the same name
        must = utils.create_path(query, ("query", "bool", "must"), [])
        must.append({"term": {"launch_name": launch_name}})
        _add_launch_id_boost(query, launch_id, launch_boost)
    elif analyzer_mode == "CURRENT_LAUNCH":
        # Just current launch
        must = utils.create_path(query, ("query", "bool", "must"), [])
        must.append({"term": {"launch_id": launch_id}})
    elif analyzer_mode == "PREVIOUS_LAUNCH":
        # Just previous launch
        must = utils.create_path(query, ("query", "bool", "must"), [])
        must.append({"term": {"launch_id": previous_launch_id}})
    elif analyzer_mode == "ALL":
        # All previous launches
        must_not = utils.create_path(query, ("query", "bool", "must_not"), [])
        must_not.append({"term": {"launch_id": launch_id}})
    else:
        # Boost launches with the same name and ID, but do not ignore any
        _add_launch_name_and_id_boost(query, launch_name, launch_id, launch_boost)
    return query


def add_constraints_for_launches_into_query_suggest(
    query: dict, test_item_info: TestItemInfo, launch_boost: float
) -> dict:
    previous_launch_id = getattr(test_item_info, "previousLaunchId", 0) or 0
    previous_launch_id = int(previous_launch_id)
    analyzer_mode = test_item_info.analyzerConfig.analyzerMode
    launch_name = test_item_info.launchName
    launch_id = test_item_info.launchId
    if analyzer_mode in {"LAUNCH_NAME", "ALL"}:
        # Previous launches with the same name
        _add_launch_name_boost(query, launch_name, launch_boost)
        should = utils.create_path(query, ("query", "bool", "should"), [])
        should.append({"term": {"launch_id": {"value": launch_id, "boost": 1 / launch_boost}}})
    elif analyzer_mode == "PREVIOUS_LAUNCH":
        # Just previous launch
        if previous_launch_id:
            _add_launch_id_boost(query, previous_launch_id, launch_boost)
    else:
        # For:
        # * CURRENT_LAUNCH
        # * CURRENT_AND_THE_SAME_NAME
        # Boost launches with the same name, but do not ignore any
        _add_launch_name_and_id_boost(query, launch_name, launch_id, launch_boost)
    return query


class AnalyzerService:
    search_cfg: SearchConfig
    launch_boost: float

    def __init__(self, search_cfg: SearchConfig):
        self.search_cfg = search_cfg
        self.launch_boost = abs(self.search_cfg.BoostLaunch)

    def find_min_should_match_threshold(self, analyzer_config: AnalyzerConf) -> int:
        if analyzer_config.minShouldMatch > 0:
            return analyzer_config.minShouldMatch
        return int(self.search_cfg.MinShouldMatch.rstrip("%"))

    def add_constraints_for_launches_into_query(self, query: dict, launch: Launch) -> dict:
        return add_constraints_for_launches_into_query(query, launch, self.launch_boost)

    def add_constraints_for_launches_into_query_suggest(self, query: dict, test_item_info: TestItemInfo) -> dict:
        return add_constraints_for_launches_into_query_suggest(query, test_item_info, self.launch_boost)

    def add_query_with_start_time_decay(self, main_query: dict, start_time: str) -> dict:
        return {
            "size": main_query["size"],
            "sort": main_query["sort"],
            "query": {
                "function_score": {
                    "query": main_query["query"],
                    "functions": [
                        {
                            "exp": {
                                "start_time": {
                                    "origin": start_time,
                                    "scale": "7d",
                                    "offset": "1d",
                                    "decay": self.search_cfg.TimeWeightDecay,
                                }
                            }
                        },
                        {"script_score": {"script": {"source": "0.6"}}},
                    ],
                    "score_mode": "max",
                    "boost_mode": "multiply",
                }
            },
        }
