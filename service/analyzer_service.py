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
from boosting_decision_making import defect_type_model, custom_defect_type_model
from boosting_decision_making import weighted_similarity_calculator
from commons import minio_client, namespace_finder
import logging
import re

logger = logging.getLogger("analyzerApp.analyzerService")


class AnalyzerService:

    def __init__(self, app_config={}, search_cfg={}):
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.es_client = EsClient(app_config=app_config, search_cfg=search_cfg)
        self.log_preparation = LogPreparation()
        self.weighted_log_similarity_calculator = None
        self.global_defect_type_model = None
        self.namespace_finder = namespace_finder.NamespaceFinder(app_config)
        self.minio_client = minio_client.MinioClient(self.app_config)
        self.initialize_common_models()

    def initialize_common_models(self):
        if self.search_cfg["SimilarityWeightsFolder"].strip():
            self.weighted_log_similarity_calculator = weighted_similarity_calculator.\
                WeightedSimilarityCalculator(folder=self.search_cfg["SimilarityWeightsFolder"])
        if self.search_cfg["GlobalDefectTypeModelFolder"].strip():
            self.global_defect_type_model = defect_type_model.\
                DefectTypeModel(folder=self.search_cfg["GlobalDefectTypeModelFolder"])

    def find_min_should_match_threshold(self, analyzer_config):
        return analyzer_config.minShouldMatch if analyzer_config.minShouldMatch > 0 else\
            int(re.search(r"\d+", self.search_cfg["MinShouldMatch"]).group(0))

    def choose_model(self, project_id, model_name_folder):
        model = None
        if self.minio_client.does_object_exists(project_id, model_name_folder):
            folders = self.minio_client.get_folder_objects(project_id, model_name_folder)
            if len(folders):
                try:
                    model = custom_defect_type_model.CustomDefectTypeModel(
                        self.app_config, project_id, folder=folders[0])
                except Exception as err:
                    logger.error(err)
        return model

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

    def build_common_query(self, log, size=10):
        return {"size": size,
                "sort": ["_score",
                         {"start_time": "desc"}, ],
                "query": {
                    "bool": {
                        "filter": [
                            {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                            {"exists": {"field": "issue_type"}},
                        ],
                        "must_not": [
                            {"wildcard": {"issue_type": "TI*"}},
                            {"wildcard": {"issue_type": "ti*"}},
                            {"wildcard": {"issue_type": "nd*"}},
                            {"wildcard": {"issue_type": "ND*"}},
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
