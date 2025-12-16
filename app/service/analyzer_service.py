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

import re
from typing import Any, Optional

from app.commons import logging
from app.commons.model.launch_objects import AnalyzerConf, Launch, SearchConfig, TestItemInfo
from app.commons.model.ml import ModelInfo
from app.commons.model_chooser import ModelChooser
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
    model_chooser: ModelChooser

    def __init__(self, model_chooser: ModelChooser, search_cfg: SearchConfig):
        self.search_cfg = search_cfg
        self.launch_boost = abs(self.search_cfg.BoostLaunch)
        self.model_chooser = model_chooser

    def find_min_should_match_threshold(self, analyzer_config: AnalyzerConf) -> int:
        return (
            analyzer_config.minShouldMatch
            if analyzer_config.minShouldMatch > 0
            else int(re.search(r"\d+", self.search_cfg.MinShouldMatch).group(0))
        )

    def add_constraints_for_launches_into_query(self, query: dict, launch: Launch) -> dict:
        return add_constraints_for_launches_into_query(query, launch, self.launch_boost)

    def add_constraints_for_launches_into_query_suggest(self, query: dict, test_item_info: TestItemInfo) -> dict:
        return add_constraints_for_launches_into_query_suggest(query, test_item_info, self.launch_boost)

    def build_more_like_this_query(
        self,
        min_should_match: str,
        log_message: str,
        field_name: str = "message",
        boost: float = 1.0,
        override_min_should_match: Optional[str] = None,
    ) -> dict:
        """Build more like this query"""
        return utils.build_more_like_this_query(
            min_should_match=min_should_match,
            log_message=log_message,
            field_name=field_name,
            boost=boost,
            override_min_should_match=override_min_should_match,
            max_query_terms=self.search_cfg.MaxQueryTerms,
        )

    @staticmethod
    def prepare_restrictions_by_issue_type(filter_no_defect: bool = True) -> list[dict]:
        if filter_no_defect:
            return [{"wildcard": {"issue_type": "ti*"}}, {"wildcard": {"issue_type": "nd*"}}]
        return [{"term": {"issue_type": "ti001"}}]

    def build_common_query(self, log: dict[str, Any], size=10, filter_no_defect=True) -> dict[str, Any]:
        issue_type_conditions = self.prepare_restrictions_by_issue_type(filter_no_defect=filter_no_defect)
        common_query = {
            "size": size,
            "sort": [
                "_score",
                {"start_time": "desc"},
            ],
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}},
                    ],
                    "must_not": issue_type_conditions + [{"term": {"test_item": log["_source"]["test_item"]}}],
                    "should": [
                        {
                            "term": {
                                "test_case_hash": {
                                    "value": log["_source"]["test_case_hash"],
                                    "boost": abs(self.search_cfg.BoostTestCaseHash),
                                }
                            }
                        },
                    ],
                }
            },
        }

        utils.append_aa_ma_boosts(common_query, self.search_cfg)
        return common_query

    def add_query_with_start_time_decay(self, main_query: dict, start_time: int) -> dict:
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

    def remove_models(self, model_info: ModelInfo) -> int:
        try:
            LOGGER.info("Started removing %s models from project %d", model_info.model_type.name, model_info.project)
            deleted_models = self.model_chooser.delete_old_model(model_info.model_type, model_info.project)
            LOGGER.info("Finished removing %s models from project %d", model_info.model_type.name, model_info.project)
            return deleted_models
        except Exception as err:
            LOGGER.exception("Error while removing models.", exc_info=err)
            return 0

    def get_model_info(self, model_info: ModelInfo) -> dict:
        try:
            LOGGER.info(
                "Started getting info for %s model from project %d", model_info.model_type.name, model_info.project
            )
            model_folder = self.model_chooser.get_model_info(model_info.model_type, model_info.project)
            LOGGER.info(
                "Finished getting info for %s model from project %d", model_info.model_type.name, model_info.project
            )
            return {"model_folder": model_folder}
        except Exception as err:
            LOGGER.exception("Error while getting info for models.", exc_info=err)
            return {}
