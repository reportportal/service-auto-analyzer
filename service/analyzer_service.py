"""
* Copyright 2019 EPAM Systems
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""
from commons.esclient import EsClient
from utils import utils
from commons.log_preparation import LogPreparation
from boosting_decision_making import weighted_similarity_calculator
from commons import namespace_finder
from commons import model_chooser
import logging
import re

logger = logging.getLogger("analyzerApp.analyzerService")


class AnalyzerService:

    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.es_client = EsClient(app_config=app_config, search_cfg=search_cfg)
        self.log_preparation = LogPreparation()
        self.namespace_finder = namespace_finder.NamespaceFinder(app_config)
        self.model_chooser = model_chooser.ModelChooser(app_config=app_config, search_cfg=search_cfg)
        self.weighted_log_similarity_calculator = None
        if self.search_cfg["SimilarityWeightsFolder"].strip():
            self.weighted_log_similarity_calculator = weighted_similarity_calculator.\
                WeightedSimilarityCalculator(folder=self.search_cfg["SimilarityWeightsFolder"])

    def find_min_should_match_threshold(self, analyzer_config):
        return analyzer_config.minShouldMatch if analyzer_config.minShouldMatch > 0 else\
            int(re.search(r"\d+", self.search_cfg["MinShouldMatch"]).group(0))

    def build_more_like_this_query(self,
                                   min_should_match, log_message,
                                   field_name="message", boost=1.0,
                                   override_min_should_match=None):
        """Build more like this query"""
        return {"more_like_this": {
            "fields":               [field_name],
            "like":                 log_message,
            "min_doc_freq":         1,
            "min_term_freq":        1,
            "minimum_should_match":
                ("5<" + min_should_match) if override_min_should_match is None else override_min_should_match,
            "max_query_terms":      self.search_cfg["MaxQueryTerms"],
            "boost": boost, }}

    def build_query_for_issue_types(self, filter_out_no_defect=True):
        queries_for_issue_types = [
            {"wildcard": {"issue_type": "TI*"}},
            {"wildcard": {"issue_type": "ti*"}}
        ]
        if filter_out_no_defect:
            return queries_for_issue_types + [
                {"wildcard": {"issue_type": "nd*"}},
                {"wildcard": {"issue_type": "ND*"}}
            ]
        return queries_for_issue_types

    def build_common_query(self, log, size=10, filter_out_no_defect=True):
        return {"size": size,
                "sort": ["_score",
                         {"start_time": "desc"}, ],
                "query": {
                    "bool": {
                        "filter": [
                            {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                            {"exists": {"field": "issue_type"}},
                        ],
                        "must_not": self.build_query_for_issue_types(
                            filter_out_no_defect=filter_out_no_defect) + [
                            {"term": {"test_item": log["_source"]["test_item"]}}
                        ],
                        "must": [],
                        "should": [
                            {"term": {"unique_id": {
                                "value": log["_source"]["unique_id"],
                                "boost": abs(self.search_cfg["BoostUniqueID"])}}},
                            {"term": {"test_case_hash": {
                                "value": log["_source"]["test_case_hash"],
                                "boost": abs(self.search_cfg["BoostUniqueID"])}}},
                            {"term": {"is_auto_analyzed": {
                                "value": str(self.search_cfg["BoostAA"] > 0).lower(),
                                "boost": abs(self.search_cfg["BoostAA"]), }}},
                        ]}}}

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
            logger.error(err)
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
            logger.error(err)
            return ""
