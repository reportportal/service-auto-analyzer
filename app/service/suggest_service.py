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

import json
from datetime import datetime
from functools import reduce
from time import time

import elasticsearch.helpers

from app.amqp.amqp import AmqpClient
from app.commons import logging, similarity_calculator, object_saving, request_factory, log_merger
from app.commons.esclient import EsClient
from app.commons.model.launch_objects import SuggestAnalysisResult, SearchConfig, ApplicationConfig, TestItemInfo, \
    AnalyzerConf
from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.machine_learning.models import WeightedSimilarityCalculator, BoostingDecisionMaker
from app.machine_learning.suggest_boosting_featurizer import SuggestBoostingFeaturizer
from app.service.analyzer_service import AnalyzerService
from app.utils import utils, text_processing

logger = logging.getLogger("analyzerApp.suggestService")
SPECIAL_FIELDS_BOOST_SCORES = [
    ("detected_message_without_params_extended", 2.0),
    ("only_numbers", 2.0), ("message_params", 2.0), ("urls", 2.0),
    ("paths", 2.0), ("found_exceptions_extended", 8.0),
    ("found_tests_and_methods", 2.0), ("test_item_name", 2.0)
]


class SuggestService(AnalyzerService):
    """The service serves suggestion lists in Make Decision modal."""

    app_config: ApplicationConfig
    search_cfg: SearchConfig
    es_client: EsClient
    namespace_finder: NamespaceFinder
    similarity_model: WeightedSimilarityCalculator

    def __init__(self, model_chooser: ModelChooser, app_config: ApplicationConfig, search_cfg: SearchConfig,
                 es_client: EsClient = None):
        self.app_config = app_config
        self.search_cfg = search_cfg
        super().__init__(model_chooser, search_cfg=self.search_cfg)
        self.es_client = es_client or EsClient(app_config=self.app_config)
        self.suggest_threshold = 0.4
        self.rp_suggest_index_template = 'rp_suggestions_info'
        self.rp_suggest_metrics_index_template = 'rp_suggestions_info_metrics'
        self.namespace_finder = NamespaceFinder(app_config)
        weights_folder = self.search_cfg.SimilarityWeightsFolder
        if not weights_folder:
            raise ValueError('SimilarityWeightsFolder is not set')
        if weights_folder:
            self.similarity_model = WeightedSimilarityCalculator(object_saving.create_filesystem(weights_folder))
            self.similarity_model.load_model()

    def get_config_for_boosting_suggests(self, analyzer_config: AnalyzerConf) -> dict:
        return {
            "max_query_terms": self.search_cfg.MaxQueryTerms,
            "min_should_match": 0.4,
            "min_word_length": self.search_cfg.MinWordLength,
            "filter_min_should_match": [],
            "filter_min_should_match_any": self.choose_fields_to_filter_suggests(analyzer_config.numberOfLogLines),
            "number_of_log_lines": analyzer_config.numberOfLogLines,
            "filter_by_test_case_hash": True,
            "boosting_model": self.search_cfg.SuggestBoostModelFolder,
            "time_weight_decay": self.search_cfg.TimeWeightDecay
        }

    def choose_fields_to_filter_suggests(self, log_lines_num: int) -> list[str]:
        if log_lines_num == -1:
            return [
                "detected_message_extended",
                "detected_message_without_params_extended",
                "detected_message_without_params_and_brackets"]
        return ["message_extended", "message_without_params_extended",
                "message_without_params_and_brackets"]

    def build_suggest_query(self, test_item_info: TestItemInfo, log: dict, size: int = 10,
                            message_field: str = "message", det_mes_field: str = "detected_message",
                            stacktrace_field: str = "stacktrace"):
        min_should_match = "{}%".format(test_item_info.analyzerConfig.minShouldMatch) \
            if test_item_info.analyzerConfig.minShouldMatch > 0 \
            else self.search_cfg.MinShouldMatch
        log_lines = test_item_info.analyzerConfig.numberOfLogLines

        query = self.build_common_query(log, size=size, filter_no_defect=False)
        query = self.add_constraints_for_launches_into_query_suggest(query, test_item_info)

        if log["_source"]["message"].strip():
            query["query"]["bool"]["filter"].append({"term": {"is_merged": False}})
            if log_lines == -1:
                must = utils.create_path(query, ('query', 'bool', 'must'), [])
                must.append(self.build_more_like_this_query("60%",
                                                            log["_source"][det_mes_field],
                                                            field_name=det_mes_field,
                                                            boost=4.0))
                if log["_source"][stacktrace_field].strip():
                    must.append(self.build_more_like_this_query("60%",
                                                                log["_source"][stacktrace_field],
                                                                field_name=stacktrace_field,
                                                                boost=2.0))
                else:
                    query["query"]["bool"]["must_not"].append({"wildcard": {stacktrace_field: "*"}})
            else:
                must = utils.create_path(query, ('query', 'bool', 'must'), [])
                must.append(self.build_more_like_this_query("60%",
                                                            log["_source"][message_field],
                                                            field_name=message_field,
                                                            boost=4.0))
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query("60%",
                                                    log["_source"][stacktrace_field],
                                                    field_name=stacktrace_field,
                                                    boost=1.0))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query("80%",
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs",
                                                boost=0.5))
        else:
            query["query"]["bool"]["filter"].append({"term": {"is_merged": True}})
            query["query"]["bool"]["must_not"].append({"wildcard": {"message": "*"}})
            must = utils.create_path(query, ('query', 'bool', 'must'), [])
            must.append(self.build_more_like_this_query(min_should_match,
                                                        log["_source"]["merged_small_logs"],
                                                        field_name="merged_small_logs",
                                                        boost=2.0))

        utils.append_potential_status_codes(query, log, max_query_terms=self.search_cfg.MaxQueryTerms)

        for field, boost_score in SPECIAL_FIELDS_BOOST_SCORES:
            if log["_source"][field].strip():
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query("1",
                                                    log["_source"][field],
                                                    field_name=field,
                                                    boost=boost_score,
                                                    override_min_should_match="1"))

        return self.add_query_with_start_time_decay(query, log["_source"]["start_time"])

    def query_es_for_suggested_items(self, test_item_info: TestItemInfo, logs: list[dict]):
        full_results = []
        index_name = text_processing.unite_project_name(test_item_info.project, self.app_config.esProjectIndexPrefix)

        for log in logs:
            message = log["_source"]["message"].strip()
            merged_small_logs = log["_source"]["merged_small_logs"].strip()
            if log["_source"]["log_level"] < utils.ERROR_LOGGING_LEVEL or \
                    (not message and not merged_small_logs):
                continue
            queries = []

            for query in [
                self.build_suggest_query(
                    test_item_info, log,
                    message_field="message_extended",
                    det_mes_field="detected_message_extended",
                    stacktrace_field="stacktrace_extended"),
                self.build_suggest_query(
                    test_item_info, log,
                    message_field="message_without_params_extended",
                    det_mes_field="detected_message_without_params_extended",
                    stacktrace_field="stacktrace_extended"),
                self.build_suggest_query(
                    test_item_info, log,
                    message_field="message_without_params_and_brackets",
                    det_mes_field="detected_message_without_params_and_brackets",
                    stacktrace_field="stacktrace_extended")]:
                queries.append("{}\n{}".format(json.dumps({"index": index_name}), json.dumps(query)))

            partial_res = self.es_client.es_client.msearch("\n".join(queries) + "\n")["responses"]
            for ind in range(len(partial_res)):
                full_results.append((log, partial_res[ind]))
        return full_results

    def deduplicate_results(self, gathered_results, scores_by_test_items, test_item_ids):
        _similarity_calculator = similarity_calculator.SimilarityCalculator(
            {
                "max_query_terms": self.search_cfg.MaxQueryTerms,
                "min_word_length": self.search_cfg.MinWordLength,
                "min_should_match": "98%",
                "number_of_log_lines": -1
            },
            similarity_model=self.similarity_model)
        all_pairs_to_check = []
        for i in range(len(gathered_results)):
            for j in range(i + 1, len(gathered_results)):
                test_item_id_first = test_item_ids[gathered_results[i][0]]
                test_item_id_second = test_item_ids[gathered_results[j][0]]
                issue_type1 = scores_by_test_items[test_item_id_first]["mrHit"]["_source"]["issue_type"]
                issue_type2 = scores_by_test_items[test_item_id_second]["mrHit"]["_source"]["issue_type"]
                if issue_type1 != issue_type2:
                    continue
                items_to_compare = {"hits": {"hits": [scores_by_test_items[test_item_id_first]["mrHit"]]}}
                all_pairs_to_check.append((scores_by_test_items[test_item_id_second]["mrHit"],
                                           items_to_compare))
        sim_dict = _similarity_calculator.find_similarity(
            all_pairs_to_check, ["detected_message_with_numbers", "stacktrace", "merged_small_logs"])

        filtered_results = []
        deleted_indices = set()
        for i in range(len(gathered_results)):
            if i in deleted_indices:
                continue
            for j in range(i + 1, len(gathered_results)):
                test_item_id_first = test_item_ids[gathered_results[i][0]]
                test_item_id_second = test_item_ids[gathered_results[j][0]]
                group_id = (str(scores_by_test_items[test_item_id_first]["mrHit"]["_id"]),
                            str(scores_by_test_items[test_item_id_second]["mrHit"]["_id"]))
                if group_id not in sim_dict["detected_message_with_numbers"]:
                    continue
                det_message = sim_dict["detected_message_with_numbers"]
                detected_message_sim = det_message[group_id]
                stacktrace_sim = sim_dict["stacktrace"][group_id]
                merged_logs_sim = sim_dict["merged_small_logs"][group_id]
                if detected_message_sim["similarity"] >= 0.98 and \
                        stacktrace_sim["similarity"] >= 0.98 and merged_logs_sim["similarity"] >= 0.98:
                    deleted_indices.add(j)
            filtered_results.append(gathered_results[i])
        return filtered_results

    def sort_results(self, scores_by_test_items, test_item_ids, predicted_labels_probability):
        gathered_results = []
        for idx, prob in enumerate(predicted_labels_probability):
            test_item_id = test_item_ids[idx]
            gathered_results.append(
                (idx,
                 round(prob[1], 4),
                 scores_by_test_items[test_item_id]["mrHit"]["_source"]["start_time"]))

        gathered_results = sorted(gathered_results, key=lambda x: (x[1], x[2]), reverse=True)
        return self.deduplicate_results(gathered_results, scores_by_test_items, test_item_ids)

    def prepare_not_found_object_info(self, test_item_info, processed_time, model_feature_names: str, model_info):
        return {  # reciprocalRank is not filled for not found results not to count in the metrics dashboard
            "project": test_item_info.project,
            "testItem": test_item_info.testItemId,
            "testItemLogId": "",
            "launchId": test_item_info.launchId,
            "launchName": test_item_info.launchName,
            "issueType": "",
            "relevantItem": "",
            "relevantLogId": "",
            "isMergedLog": False,
            "matchScore": 0.0,
            "resultPosition": -1,
            "modelFeatureNames": model_feature_names,
            "modelFeatureValues": "",
            "modelInfo": model_info,
            "usedLogLines": test_item_info.analyzerConfig.numberOfLogLines,
            "minShouldMatch": self.find_min_should_match_threshold(test_item_info.analyzerConfig),
            "userChoice": 0,
            "processedTime": processed_time,
            "notFoundResults": 100,
            "savedDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "module_version": [self.app_config.appVersion],
            "methodName": "suggestion",
            "clusterId": test_item_info.clusterId
        }

    def get_query_for_test_item_in_cluster(self, test_item_info):
        return {
            "_source": ["test_item"],
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}},
                        {"term": {"is_merged": False}},
                    ],
                    "should": [],
                    "must": [
                        {"term": {"launch_id": test_item_info.launchId}},
                        {"term": {"cluster_id": test_item_info.clusterId}}
                    ]}}}

    def get_query_for_logs_by_test_item(self, test_item_id):
        return {
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}},
                        {"term": {"is_merged": False}},
                        {"term": {"test_item": test_item_id}}]
                }
            }
        }

    def query_logs_for_cluster(self, test_item_info: TestItemInfo, index_name: str) -> tuple[list[dict], int]:
        test_item_id = None
        test_items = self.es_client.es_client.search(
            index_name, body=self.get_query_for_test_item_in_cluster(test_item_info))
        for res in test_items["hits"]["hits"]:
            test_item_id = int(res["_source"]["test_item"])
            break
        if test_item_id is None:
            return [], 0
        logs = []
        for log in elasticsearch.helpers.scan(
                self.es_client.es_client, query=self.get_query_for_logs_by_test_item(test_item_id), index=index_name):
            # clean test item info not to boost by it
            log["_source"]["test_item"] = 0
            log["_source"]["test_case_hash"] = 0
            log["_source"]["unique_id"] = ""
            log["_source"]["test_item_name"] = ""
            logs.append(log)
        return logs, test_item_id

    def prepare_logs_for_suggestions(self, test_item_info: TestItemInfo, index_name: str) -> tuple[list[dict], int]:
        test_item_id_for_suggest = test_item_info.testItemId
        if test_item_info.clusterId != 0:
            prepared_logs, test_item_id_for_suggest = self.query_logs_for_cluster(test_item_info, index_name)
        else:
            unique_logs = text_processing.leave_only_unique_logs(test_item_info.logs)
            prepared_logs = [request_factory.prepare_log_for_suggests(test_item_info, log, index_name)
                             for log in unique_logs if log.logLevel >= utils.ERROR_LOGGING_LEVEL]
        logs, _ = log_merger.decompose_logs_merged_and_without_duplicates(prepared_logs)
        return logs, test_item_id_for_suggest

    def suggest_items(self, test_item_info: TestItemInfo):
        logger.info(f'Started suggesting for test item with id: {test_item_info.testItemId}')
        logger.debug(f'Started suggesting items by request: {test_item_info.json()}')
        logger.info("ES Url %s", text_processing.remove_credentials_from_url(self.es_client.host))
        index_name = text_processing.unite_project_name(test_item_info.project, self.app_config.esProjectIndexPrefix)
        if not self.es_client.index_exists(index_name):
            logger.info("Project %s doesn't exist", index_name)
            logger.info("Finished suggesting for test item with 0 results.")
            return []

        t_start = time()
        results = []
        errors_found = []
        errors_count = 0
        model_info_tags = []
        feature_names = ""
        try:
            logs, test_item_id_for_suggest = self.prepare_logs_for_suggestions(test_item_info, index_name)
            logger.info(f'Number of prepared log search requests for suggestions: {len(logs)}')
            logger.debug(f'Log search requests for suggestions: {json.dumps(logs)}')
            searched_res = self.query_es_for_suggested_items(test_item_info, logs)
            res_num = reduce(lambda a, b: a + b, [len(res[1]['hits']['hits']) for res in searched_res], 0)
            logger.info(f'Found {res_num} items by FTS (KNN)')
            logger.debug(f'Items for suggestions by FTS (KNN): {json.dumps(searched_res)}')

            boosting_config = self.get_config_for_boosting_suggests(test_item_info.analyzerConfig)
            boosting_config["chosen_namespaces"] = self.namespace_finder.get_chosen_namespaces(test_item_info.project)
            # noinspection PyTypeChecker
            _suggest_decision_maker_to_use: BoostingDecisionMaker = self.model_chooser.choose_model(
                test_item_info.project, ModelType.suggestion,
                custom_model_prob=self.search_cfg.ProbabilityForCustomModelSuggestions)

            _boosting_data_gatherer = SuggestBoostingFeaturizer(
                searched_res,
                boosting_config,
                feature_ids=_suggest_decision_maker_to_use.feature_ids,
                weighted_log_similarity_calculator=self.similarity_model)
            # noinspection PyTypeChecker
            _boosting_data_gatherer.set_defect_type_model(
                self.model_chooser.choose_model(test_item_info.project, ModelType.defect_type))
            feature_data, test_item_ids = _boosting_data_gatherer.gather_features_info()
            scores_by_test_items = _boosting_data_gatherer.find_most_relevant_by_type()
            model_info_tags = (_boosting_data_gatherer.get_used_model_info()
                               + _suggest_decision_maker_to_use.get_model_info())
            feature_names = ";".join([str(i) for i in _suggest_decision_maker_to_use.feature_ids])
            if feature_data:
                predicted_labels, predicted_labels_probability = _suggest_decision_maker_to_use.predict(feature_data)
                sorted_results = self.sort_results(
                    scores_by_test_items, test_item_ids, predicted_labels_probability)

                logger.debug("Found %d results for test items ", len(sorted_results))
                for idx, prob, _ in sorted_results:
                    test_item_id = test_item_ids[idx]
                    issue_type = scores_by_test_items[test_item_id]["mrHit"]["_source"]["issue_type"]
                    logger.debug("Test item id %s with issue type %s has probability %.2f",
                                 test_item_id, issue_type, prob)
                processed_time = time() - t_start
                global_idx = 0
                for idx, prob, _ in sorted_results[:self.search_cfg.MaxSuggestionsNumber]:
                    if prob >= self.suggest_threshold:
                        test_item_id = test_item_ids[idx]
                        issue_type = scores_by_test_items[test_item_id]["mrHit"]["_source"]["issue_type"]
                        relevant_log_id = utils.extract_real_id(
                            scores_by_test_items[test_item_id]["mrHit"]["_id"])
                        real_log_id = str(scores_by_test_items[test_item_id]["mrHit"]["_id"])
                        is_merged = real_log_id != str(relevant_log_id)
                        test_item_log_id = utils.extract_real_id(
                            scores_by_test_items[test_item_id]["compared_log"]["_id"])
                        analysis_result = SuggestAnalysisResult(
                            project=test_item_info.project,
                            testItem=test_item_id_for_suggest,
                            testItemLogId=test_item_log_id,
                            launchId=test_item_info.launchId,
                            launchName=test_item_info.launchName,
                            launchNumber=test_item_info.launchNumber,
                            issueType=issue_type,
                            relevantItem=test_item_id,
                            relevantLogId=relevant_log_id,
                            isMergedLog=is_merged,
                            matchScore=round(prob, 2) * 100,
                            esScore=round(scores_by_test_items[test_item_id]["mrHit"]["_score"], 2),
                            esPosition=scores_by_test_items[test_item_id]["mrHit"]["es_pos"],
                            modelFeatureNames=feature_names,
                            modelFeatureValues=";".join([str(feature) for feature in feature_data[idx]]),
                            modelInfo=";".join(model_info_tags),
                            resultPosition=global_idx,
                            usedLogLines=test_item_info.analyzerConfig.numberOfLogLines,
                            minShouldMatch=self.find_min_should_match_threshold(
                                test_item_info.analyzerConfig),
                            processedTime=processed_time,
                            clusterId=test_item_info.clusterId,
                            methodName="suggestion")
                        results.append(analysis_result)
                        logger.debug(analysis_result)
                        global_idx += 1
            else:
                logger.debug("There are no results for test item %s", test_item_info.testItemId)
        except Exception as exc:
            logger.exception(exc)
            errors_found.append(utils.extract_exception(exc))
            errors_count += 1
        results_to_share = {test_item_info.launchId: {
            "not_found": int(len(results) == 0), "items_to_process": 1,
            "processed_time": time() - t_start, "found_items": len(results),
            "launch_id": test_item_info.launchId, "launch_name": test_item_info.launchName,
            "project_id": test_item_info.project, "method": "suggest",
            "gather_date": datetime.now().strftime("%Y-%m-%d"),
            "gather_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "number_of_log_lines": test_item_info.analyzerConfig.numberOfLogLines,
            "model_info": model_info_tags,
            "module_version": [self.app_config.appVersion],
            "min_should_match": self.find_min_should_match_threshold(
                test_item_info.analyzerConfig),
            "errors": errors_found,
            "errors_count": errors_count}}
        if not results:
            self.es_client.create_index_for_stats_info(self.rp_suggest_metrics_index_template)
            self.es_client._bulk_index([{
                "_index": self.rp_suggest_metrics_index_template,
                "_source": self.prepare_not_found_object_info(
                    test_item_info, time() - t_start, feature_names, model_info_tags)
            }])
        if self.app_config.amqpUrl:
            amqp_client = AmqpClient(self.app_config)
            amqp_client.send_to_inner_queue("stats_info", json.dumps(results_to_share))
            if results:
                for model_type in [ModelType.suggestion, ModelType.auto_analysis]:
                    amqp_client.send_to_inner_queue(
                        'train_models',
                        TrainInfo(model_type=model_type, project=test_item_info.project,
                                  gathered_metric_total=len(results)).json())
            amqp_client.close()
        logger.debug("Stats info %s", results_to_share)
        logger.info("Processed the test item. It took %.2f sec.", time() - t_start)
        logger.info("Finished suggesting for test item with %d results.", len(results))
        return results
