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
from typing import Any

from app.commons import logging, object_saving
from app.commons.log_merger import LogMerger
from app.commons.log_preparation import LogPreparation
from app.machine_learning.models import weighted_similarity_calculator
from app.utils import utils

logger = logging.getLogger("analyzerApp.analyzerService")


class AnalyzerService:
    launch_boost: float

    def __init__(self, model_chooser, search_cfg=None):
        self.search_cfg = search_cfg or {}
        self.launch_boost = abs(self.search_cfg['BoostLaunch'])
        self.log_preparation = LogPreparation()
        self.log_merger = LogMerger()
        self.model_chooser = model_chooser
        self.weighted_log_similarity_calculator = None
        weights_folder = self.search_cfg.get('SimilarityWeightsFolder', 'res/model/weights_24.11.20').strip()
        if weights_folder:
            self.weighted_log_similarity_calculator = (
                weighted_similarity_calculator.WeightedSimilarityCalculator(
                    object_saving.create_filesystem(weights_folder)))
            self.weighted_log_similarity_calculator.load_model()

    def find_min_should_match_threshold(self, analyzer_config):
        return analyzer_config.minShouldMatch if analyzer_config.minShouldMatch > 0 else \
            int(re.search(r"\d+", self.search_cfg["MinShouldMatch"]).group(0))

    def create_path(self, query: dict, path: tuple[str, ...], value: Any) -> Any:
        path_length = len(path)
        last_element = path[path_length - 1]
        current_node = query
        for i in range(path_length - 1):
            element = path[i]
            if element not in current_node:
                current_node[element] = {}
            current_node = current_node[element]
        if last_element not in current_node:
            current_node[last_element] = value
        return current_node[last_element]

    def _add_launch_name_boost(self, query: dict, launch_name: str) -> None:
        should = self.create_path(query, ('query', 'bool', 'should'), [])
        should.append({'term': {'launch_name': {'value': launch_name, 'boost': self.launch_boost}}})

    def _add_launch_id_boost(self, query: dict, launch_id: int) -> None:
        should = self.create_path(query, ('query', 'bool', 'should'), [])
        should.append({'term': {'launch_id': {'value': launch_id, 'boost': self.launch_boost}}})

    def _add_launch_name_and_id_boost(self, query: dict, launch_name: str, launch_id: int):
        self._add_launch_id_boost(query, launch_id)
        self._add_launch_name_boost(query, launch_name)

    def add_constraints_for_launches_into_query(self, query: dict, launch) -> dict:
        previous_launch_id = getattr(launch, 'previousLaunchId', 0) or 0
        previous_launch_id = int(previous_launch_id)
        analyzer_mode = launch.analyzerConfig.analyzerMode
        launch_name = launch.launchName
        launch_id = launch.launchId
        if analyzer_mode == 'LAUNCH_NAME':
            # Previous launches with the same name
            must = self.create_path(query, ('query', 'bool', 'must'), [])
            must_not = self.create_path(query, ('query', 'bool', 'must_not'), [])
            must.append({'term': {'launch_name': launch_name}})
            must_not.append({'term': {'launch_id': launch_id}})
        elif analyzer_mode == 'CURRENT_AND_THE_SAME_NAME':
            # All launches with the same name
            must = self.create_path(query, ('query', 'bool', 'must'), [])
            must.append({'term': {'launch_name': launch_name}})
            self._add_launch_id_boost(query, launch_id)
        elif analyzer_mode == 'CURRENT_LAUNCH':
            # Just current launch
            must = self.create_path(query, ('query', 'bool', 'must'), [])
            must.append({'term': {'launch_id': launch_id}})
        elif analyzer_mode == 'PREVIOUS_LAUNCH':
            # Just previous launch
            must = self.create_path(query, ('query', 'bool', 'must'), [])
            must.append({'term': {'launch_id': previous_launch_id}})
        elif analyzer_mode == 'ALL':
            # All previous launches
            must_not = self.create_path(query, ('query', 'bool', 'must_not'), [])
            must_not.append({'term': {'launch_id': launch_id}})
        else:
            # Boost launches with the same name and ID, but do not ignore any
            self._add_launch_name_and_id_boost(query, launch_name, launch_id)
        return query

    def add_constraints_for_launches_into_query_suggest(self, query: dict, test_item_info) -> dict:
        previous_launch_id = getattr(test_item_info, 'previousLaunchId', 0) or 0
        previous_launch_id = int(previous_launch_id)
        analyzer_mode = test_item_info.analyzerConfig.analyzerMode
        launch_name = test_item_info.launchName
        launch_id = test_item_info.launchId
        launch_boost = abs(self.search_cfg['BoostLaunch'])
        if analyzer_mode in {'LAUNCH_NAME', 'ALL'}:
            # Previous launches with the same name
            self._add_launch_name_boost(query, launch_name)
            should = self.create_path(query, ('query', 'bool', 'should'), [])
            should.append({'term': {'launch_id': {'value': launch_id, 'boost': 1 / launch_boost}}})
        elif analyzer_mode == 'PREVIOUS_LAUNCH':
            # Just previous launch
            if previous_launch_id:
                self._add_launch_id_boost(query, previous_launch_id)
        else:
            # For:
            # * CURRENT_LAUNCH
            # * CURRENT_AND_THE_SAME_NAME
            # Boost launches with the same name, but do not ignore any
            self._add_launch_name_and_id_boost(query, launch_name, launch_id)
        return query

    def build_more_like_this_query(self,
                                   min_should_match, log_message,
                                   field_name="message", boost=1.0,
                                   override_min_should_match=None):
        """Build more like this query"""
        return utils.build_more_like_this_query(
            min_should_match=min_should_match,
            log_message=log_message,
            field_name=field_name,
            boost=boost,
            override_min_should_match=override_min_should_match,
            max_query_terms=self.search_cfg["MaxQueryTerms"]
        )

    def prepare_restrictions_by_issue_type(self, filter_no_defect=True):
        if filter_no_defect:
            return [
                {"wildcard": {"issue_type": "ti*"}},
                {"wildcard": {"issue_type": "nd*"}}]
        return [{"term": {"issue_type": "ti001"}}]

    def build_common_query(self, log, size=10, filter_no_defect=True):
        issue_type_conditions = self.prepare_restrictions_by_issue_type(
            filter_no_defect=filter_no_defect)
        return {"size": size,
                "sort": ["_score", {"start_time": "desc"}, ],
                "query": {
                    "bool": {
                        "filter": [
                            {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                            {"exists": {"field": "issue_type"}},
                        ],
                        "must_not": issue_type_conditions + [{"term": {"test_item": log["_source"]["test_item"]}}],
                        "should": [
                            {"term": {"test_case_hash": {
                                "value": log["_source"]["test_case_hash"],
                                "boost": abs(self.search_cfg["BoostTestCaseHash"])}}},
                            {"term": {"is_auto_analyzed": {
                                "value": str(self.search_cfg["BoostAA"] > 0).lower(),
                                "boost": abs(self.search_cfg["BoostAA"]), }}},
                        ]
                    }
                }}

    def add_query_with_start_time_decay(self, main_query, start_time):
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
                                    "decay": self.search_cfg["TimeWeightDecay"]
                                }
                            }
                        },
                        {
                            "script_score": {"script": {"source": "0.6"}}
                        }],
                    "score_mode": "max",
                    "boost_mode": "multiply"
                }
            }
        }

    def remove_models(self, model_info):
        try:
            logger.info("Started removing %s models from project %d",
                        model_info["model_type"], model_info["project"])
            deleted_models = self.model_chooser.delete_old_model(
                model_name=model_info["model_type"] + "_model",
                project_id=model_info["project"])
            logger.info("Finished removing %s models from project %d",
                        model_info["model_type"], model_info["project"])
            return deleted_models
        except Exception as err:
            logger.error("Error while removing models.")
            logger.exception(err)
            return 0

    def get_model_info(self, model_info):
        try:
            logger.info("Started getting info for %s model from project %d",
                        model_info["model_type"], model_info["project"])
            model_folder = self.model_chooser.get_model_info(
                model_name=model_info["model_type"] + "_model",
                project_id=model_info["project"])
            logger.info("Finished getting info for %s model from project %d",
                        model_info["model_type"], model_info["project"])
            return {"model_folder": model_folder}
        except Exception as err:
            logger.error("Error while getting info for models.")
            logger.exception(err)
            return ""
