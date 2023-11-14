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
from queue import Queue
from threading import Thread
from time import time, sleep

from app.amqp.amqp import AmqpClient
from app.boosting_decision_making import boosting_featurizer
from app.commons import logging
from app.commons.esclient import EsClient
from app.commons.launch_objects import AnalysisResult, BatchLogInfo, AnalysisCandidate, SuggestAnalysisResult
from app.commons.model_chooser import ModelType
from app.commons.namespace_finder import NamespaceFinder
from app.commons.similarity_calculator import SimilarityCalculator
from app.service.analyzer_service import AnalyzerService
from app.utils import utils, text_processing

logger = logging.getLogger("analyzerApp.autoAnalyzerService")
EARLY_FINISH = False


class AutoAnalyzerService(AnalyzerService):
    es_client: EsClient
    namespace_finder: NamespaceFinder

    def __init__(self, model_chooser, app_config=None, search_cfg=None, es_client: EsClient = None):
        self.app_config = app_config or {}
        self.search_cfg = search_cfg or {}
        super().__init__(model_chooser, search_cfg=self.search_cfg)
        self.es_client = es_client or EsClient(app_config=self.app_config, search_cfg=self.search_cfg)
        self.namespace_finder = NamespaceFinder(app_config)

    def get_config_for_boosting(self, analyzer_config):
        min_should_match = self.find_min_should_match_threshold(analyzer_config) / 100
        return {
            "max_query_terms": self.search_cfg["MaxQueryTerms"],
            "min_should_match": min_should_match,
            "min_word_length": self.search_cfg["MinWordLength"],
            "filter_min_should_match_any": [],
            "filter_min_should_match": self.choose_fields_to_filter_strict(
                analyzer_config.numberOfLogLines, min_should_match),
            "number_of_log_lines": analyzer_config.numberOfLogLines,
            "filter_by_test_case_hash": True,
            "boosting_model": self.search_cfg["BoostModelFolder"],
            "filter_by_all_logs_should_be_similar": analyzer_config.allMessagesShouldMatch,
            "time_weight_decay": self.search_cfg["TimeWeightDecay"]
        }

    def choose_fields_to_filter_strict(self, log_lines, min_should_match):
        fields = [
            "detected_message", "message", "potential_status_codes"] \
            if log_lines == -1 else ["message", "potential_status_codes"]
        if min_should_match > 0.99:
            fields.append("found_tests_and_methods")
        return fields

    def get_min_should_match_setting(self, launch):
        return "{}%".format(launch.analyzerConfig.minShouldMatch) \
            if launch.analyzerConfig.minShouldMatch > 0 else self.search_cfg["MinShouldMatch"]

    def build_analyze_query(self, launch, log, size=10):
        """Build analyze query"""
        min_should_match = self.get_min_should_match_setting(launch)

        query = self.build_common_query(log, size=size)
        query = self.add_constraints_for_launches_into_query(query, launch)

        if log["_source"]["message"].strip():
            log_lines = launch.analyzerConfig.numberOfLogLines
            query["query"]["bool"]["filter"].append({"term": {"is_merged": False}})
            if log_lines == -1:
                query["query"]["bool"]["must"].append(
                    self.build_more_like_this_query(min_should_match,
                                                    log["_source"]["detected_message"],
                                                    field_name="detected_message",
                                                    boost=4.0))
                if log["_source"]["stacktrace"].strip():
                    query["query"]["bool"]["must"].append(
                        self.build_more_like_this_query(min_should_match,
                                                        log["_source"]["stacktrace"],
                                                        field_name="stacktrace",
                                                        boost=2.0))
                else:
                    query["query"]["bool"]["must_not"].append({"wildcard": {"stacktrace": "*"}})
            else:
                query["query"]["bool"]["must"].append(
                    self.build_more_like_this_query(min_should_match,
                                                    log["_source"]["message"],
                                                    field_name="message",
                                                    boost=4.0))
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query("80%",
                                                    log["_source"]["detected_message"],
                                                    field_name="detected_message",
                                                    boost=2.0))
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query("60%",
                                                    log["_source"]["stacktrace"],
                                                    field_name="stacktrace", boost=1.0))
            query["query"]["bool"]["should"].append(
                self.build_more_like_this_query("80%",
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs",
                                                boost=0.5))
        else:
            query["query"]["bool"]["filter"].append({"term": {"is_merged": True}})
            query["query"]["bool"]["must_not"].append({"wildcard": {"message": "*"}})
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query(min_should_match,
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs",
                                                boost=2.0))

        if log["_source"]["found_exceptions"].strip():
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query("1",
                                                log["_source"]["found_exceptions"],
                                                field_name="found_exceptions",
                                                boost=8.0,
                                                override_min_should_match="1"))
        for field, boost_score in [
            ("detected_message_without_params_extended", 2.0),
            ("only_numbers", 2.0), ("potential_status_codes", 8.0),
            ("found_tests_and_methods", 2), ("test_item_name", 2.0)
        ]:
            if log["_source"][field].strip():
                query["query"]["bool"]["should"].append(
                    self.build_more_like_this_query("1",
                                                    log["_source"][field],
                                                    field_name=field,
                                                    boost=boost_score,
                                                    override_min_should_match="1"))

        return self.add_query_with_start_time_decay(query, log["_source"]["start_time"])

    def build_query_with_no_defect(self, launch, log, size=10):
        min_should_match = self.get_min_should_match_setting(launch)
        query = {
            "size": size,
            "sort": ["_score",
                     {"start_time": "desc"}],
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": utils.ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}}
                    ],
                    "must_not": [
                        {"term": {"issue_type": "ti001"}},
                        {"term": {"test_item": log["_source"]["test_item"]}}
                    ],
                    "must": [
                        {"term": {"test_case_hash": log["_source"]["test_case_hash"]}}
                    ],
                    "should": []
                }}}
        query = self.add_constraints_for_launches_into_query(query, launch)
        if log["_source"]["message"].strip():
            query["query"]["bool"]["filter"].append({"term": {"is_merged": False}})
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query(min_should_match,
                                                log["_source"]["message"],
                                                field_name="message"))
        else:
            query["query"]["bool"]["filter"].append({"term": {"is_merged": True}})
            query["query"]["bool"]["must_not"].append({"wildcard": {"message": "*"}})
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query(min_should_match,
                                                log["_source"]["merged_small_logs"],
                                                field_name="merged_small_logs"))
        if log["_source"]["found_exceptions"].strip():
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query("1",
                                                log["_source"]["found_exceptions"],
                                                field_name="found_exceptions",
                                                boost=8.0,
                                                override_min_should_match="1"))
        utils.append_potential_status_codes(query, log, max_query_terms=self.search_cfg["MaxQueryTerms"])
        return self.add_query_with_start_time_decay(query, log["_source"]["start_time"])

    def leave_only_similar_logs(self, candidates_with_no_defect, boosting_config):
        new_results = []
        for log_info, search_res in candidates_with_no_defect:
            no_defect_candidate_exists = False
            for log in search_res["hits"]["hits"]:
                if log["_source"]["issue_type"][:2].lower() in ["nd", "ti"]:
                    no_defect_candidate_exists = True
            new_search_res = []
            _similarity_calculator = SimilarityCalculator(
                boosting_config,
                weighted_similarity_calculator=self.weighted_log_similarity_calculator)
            if no_defect_candidate_exists:
                _similarity_calculator.find_similarity(
                    [(log_info, search_res)],
                    ["message", "merged_small_logs"])
                for obj in search_res["hits"]["hits"]:
                    group_id = (obj["_id"], log_info["_id"])
                    if group_id in _similarity_calculator.similarity_dict["message"]:
                        sim_val = _similarity_calculator.similarity_dict["message"][group_id]
                        if sim_val["both_empty"]:
                            sim_val = _similarity_calculator.similarity_dict["merged_small_logs"][group_id]
                        threshold = boosting_config["min_should_match"]
                        if not sim_val["both_empty"] and sim_val["similarity"] >= threshold:
                            new_search_res.append(obj)
            new_results.append((log_info, {"hits": {"hits": new_search_res}}))
        return new_results

    def filter_by_all_logs_should_be_similar(self, candidates_with_no_defect, boosting_config):
        if boosting_config["filter_by_all_logs_should_be_similar"]:
            test_item_stats = {}
            for log_info, search_res in candidates_with_no_defect:
                for log in search_res["hits"]["hits"]:
                    if log["_source"]["test_item"] not in test_item_stats:
                        test_item_stats[log["_source"]["test_item"]] = 0
                    test_item_stats[log["_source"]["test_item"]] += 1
            new_results = []
            for log_info, search_res in candidates_with_no_defect:
                new_search_res = []
                for log in search_res["hits"]["hits"]:
                    if log["_source"]["test_item"] in test_item_stats:
                        if test_item_stats[log["_source"]["test_item"]] == len(candidates_with_no_defect):
                            new_search_res.append(log)
                new_results.append((log_info, {"hits": {"hits": new_search_res}}))
            return new_results
        return candidates_with_no_defect

    def find_relevant_with_no_defect(self, candidates_with_no_defect, boosting_config):
        candidates_with_no_defect = self.leave_only_similar_logs(
            candidates_with_no_defect, boosting_config)
        candidates_with_no_defect = self.filter_by_all_logs_should_be_similar(
            candidates_with_no_defect, boosting_config)
        for log_info, search_res in candidates_with_no_defect:
            latest_type = None
            latest_item = None
            latest_date = None
            for obj in search_res["hits"]["hits"]:
                logger.debug("%s %s %s", obj["_source"]["start_time"], obj["_source"]["issue_type"],
                             obj["_source"]["test_item"])
                start_time = datetime.strptime(obj["_source"]["start_time"], '%Y-%m-%d %H:%M:%S')
                if latest_date is None or latest_date < start_time:
                    latest_type = obj["_source"]["issue_type"]
                    latest_item = obj
                    latest_date = start_time
            if latest_type and latest_type[:2].lower() in ["nd", "ti"]:
                return [(log_info, {"hits": {"hits": [latest_item]}})]
        return []

    def _send_result_to_queue(self, test_item_dict, batches, batch_logs):
        t_start = time()
        partial_res = self.es_client.es_client.msearch("\n".join(batches) + "\n")["responses"]
        avg_time_processed = (time() - t_start) / (len(partial_res) if partial_res else 1)
        for test_item_id in test_item_dict:
            candidates = []
            candidates_with_no_defect = []
            time_processed = 0.0
            batch_log_info = None
            for idx in test_item_dict[test_item_id]:
                batch_log_info = batch_logs[idx]
                if batch_log_info.query_type == "without no defect":
                    candidates.append(
                        (batch_log_info.log_info, partial_res[idx]))
                if batch_log_info.query_type == "with no defect":
                    candidates_with_no_defect.append(
                        (batch_log_info.log_info, partial_res[idx]))
                time_processed += avg_time_processed
            if batch_log_info:
                self.queue.put(AnalysisCandidate(
                    analyzerConfig=batch_log_info.analyzerConfig,
                    testItemId=batch_log_info.testItemId,
                    project=batch_log_info.project,
                    launchId=batch_log_info.launchId,
                    launchName=batch_log_info.launchName,
                    launchNumber=batch_log_info.launchNumber,
                    timeProcessed=time_processed,
                    candidates=candidates,
                    candidatesWithNoDefect=candidates_with_no_defect
                ))

    def _query_elasticsearch(self, launches, max_batch_size=30):
        t_start = time()
        batches = []
        batch_logs = []
        index_in_batch = 0
        test_item_dict = {}
        batch_size = 5
        n_first_blocks = 3
        test_items_number_to_process = 0
        try:
            for launch in launches:
                index_name = text_processing.unite_project_name(
                    str(launch.project), self.app_config["esProjectIndexPrefix"])
                if not self.es_client.index_exists(index_name):
                    continue
                if test_items_number_to_process >= self.search_cfg["MaxAutoAnalysisItemsToProcess"]:
                    logger.info("Only first %d test items were taken",
                                self.search_cfg["MaxAutoAnalysisItemsToProcess"])
                    break
                if EARLY_FINISH:
                    logger.info("Early finish from analyzer before timeout")
                    break
                for test_item in launch.testItems:
                    if test_items_number_to_process >= self.search_cfg["MaxAutoAnalysisItemsToProcess"]:
                        logger.info("Only first %d test items were taken",
                                    self.search_cfg["MaxAutoAnalysisItemsToProcess"])
                        break
                    if EARLY_FINISH:
                        logger.info("Early finish from analyzer before timeout")
                        break
                    unique_logs = text_processing.leave_only_unique_logs(test_item.logs)
                    prepared_logs = [self.log_preparation._prepare_log(launch, test_item, log, index_name)
                                     for log in unique_logs if log.logLevel >= utils.ERROR_LOGGING_LEVEL]
                    results, _ = self.log_merger.decompose_logs_merged_and_without_duplicates(prepared_logs)

                    for log in results:
                        message = log["_source"]["message"].strip()
                        merged_logs = log["_source"]["merged_small_logs"].strip()
                        if log["_source"]["log_level"] < utils.ERROR_LOGGING_LEVEL or \
                                (not message and not merged_logs):
                            continue
                        for query_type, query in [
                            ("without no defect", self.build_analyze_query(launch, log)),
                            ("with no defect", self.build_query_with_no_defect(launch, log))
                        ]:
                            full_query = "{}\n{}".format(
                                json.dumps({"index": index_name}), json.dumps(query))
                            batches.append(full_query)
                            batch_logs.append(BatchLogInfo(
                                analyzerConfig=launch.analyzerConfig,
                                testItemId=test_item.testItemId,
                                log_info=log,
                                query_type=query_type,
                                project=launch.project,
                                launchId=launch.launchId,
                                launchName=launch.launchName,
                                launchNumber=launch.launchNumber
                            ))
                            if test_item.testItemId not in test_item_dict:
                                test_item_dict[test_item.testItemId] = []
                            test_item_dict[test_item.testItemId].append(index_in_batch)
                            index_in_batch += 1
                    if n_first_blocks <= 0:
                        batch_size = max_batch_size
                    if len(batches) >= batch_size:
                        n_first_blocks -= 1
                        self._send_result_to_queue(test_item_dict, batches, batch_logs)
                        batches = []
                        batch_logs = []
                        test_item_dict = {}
                        index_in_batch = 0
                    test_items_number_to_process += 1
            if len(batches) > 0:
                self._send_result_to_queue(test_item_dict, batches, batch_logs)

        except Exception as err:
            logger.error("Error in ES query")
            logger.error(err)
        self.finished_queue.put("Finished")
        logger.info("Es queries finished %.2f s.", time() - t_start)

    @utils.ignore_warnings
    def analyze_logs(self, launches):
        global EARLY_FINISH
        cnt_launches = len(launches)
        logger.info("Started analysis for %d launches", cnt_launches)
        logger.info("ES Url %s", text_processing.remove_credentials_from_url(self.es_client.host))
        self.queue = Queue()
        self.finished_queue = Queue()
        defect_type_model_to_use = {}
        es_query_thread = Thread(target=self._query_elasticsearch, args=(launches,))
        es_query_thread.daemon = True
        es_query_thread.start()
        analyzed_results_for_index = []
        try:
            results = []
            t_start = time()
            del launches

            cnt_items_to_process = 0
            results_to_share = {}
            chosen_namespaces = {}
            while self.finished_queue.empty() or not self.queue.empty():
                if (self.search_cfg["AutoAnalysisTimeout"] - (
                        time() - t_start)) <= 5:  # check whether we are running out of time # noqa
                    EARLY_FINISH = True
                    break
                if self.queue.empty():
                    sleep(0.1)
                    continue
                else:
                    analyzer_candidates = self.queue.get()
                try:
                    project_id = analyzer_candidates.project
                    launch_id = analyzer_candidates.launchId
                    if launch_id not in results_to_share:
                        results_to_share[launch_id] = {
                            "not_found": 0, "items_to_process": 0, "processed_time": 0,
                            "launch_id": launch_id,
                            "launch_name": analyzer_candidates.launchName,
                            "project_id": project_id,
                            "method": "auto_analysis",
                            "gather_date": datetime.now().strftime("%Y-%m-%d"),
                            "gather_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "number_of_log_lines": analyzer_candidates.analyzerConfig.numberOfLogLines,
                            "min_should_match": self.find_min_should_match_threshold(
                                analyzer_candidates.analyzerConfig),
                            "model_info": set(),
                            "module_version": [self.app_config["appVersion"]],
                            "errors": [],
                            "errors_count": 0}

                    t_start_item = time()
                    cnt_items_to_process += 1
                    results_to_share[launch_id]["items_to_process"] += 1
                    results_to_share[launch_id]["processed_time"] += analyzer_candidates.timeProcessed
                    boosting_config = self.get_config_for_boosting(analyzer_candidates.analyzerConfig)

                    if project_id not in chosen_namespaces:
                        chosen_namespaces[project_id] = self.namespace_finder.get_chosen_namespaces(
                            project_id)
                    boosting_config["chosen_namespaces"] = chosen_namespaces[project_id]
                    _boosting_decision_maker = self.model_chooser.choose_model(
                        project_id, ModelType.AUTO_ANALYSIS_MODEL,
                        custom_model_prob=self.search_cfg["ProbabilityForCustomModelAutoAnalysis"])
                    features_dict_objects = _boosting_decision_maker.features_dict_with_saved_objects
                    if project_id not in defect_type_model_to_use:
                        defect_type_model_to_use[project_id] = self.model_chooser.choose_model(
                            project_id, ModelType.DEFECT_TYPE_MODEL)

                    relevant_with_no_defect_candidate = self.find_relevant_with_no_defect(
                        analyzer_candidates.candidatesWithNoDefect, boosting_config)

                    candidates_to_check = []
                    if relevant_with_no_defect_candidate:
                        candidates_to_check.append(relevant_with_no_defect_candidate)
                    candidates_to_check.append(analyzer_candidates.candidates)

                    found_result = False

                    for candidates in candidates_to_check:
                        boosting_data_gatherer = boosting_featurizer.BoostingFeaturizer(
                            candidates,
                            boosting_config,
                            feature_ids=_boosting_decision_maker.get_feature_ids(),
                            weighted_log_similarity_calculator=self.weighted_log_similarity_calculator,
                            features_dict_with_saved_objects=features_dict_objects)
                        boosting_data_gatherer.set_defect_type_model(defect_type_model_to_use[project_id])
                        feature_data, issue_type_names = boosting_data_gatherer.gather_features_info()
                        model_info_tags = (boosting_data_gatherer.get_used_model_info() +
                                           _boosting_decision_maker.get_model_info())
                        results_to_share[launch_id]["model_info"].update(model_info_tags)

                        if len(feature_data) > 0:

                            predicted_labels, predicted_labels_probability = \
                                _boosting_decision_maker.predict(feature_data)

                            scores_by_issue_type = boosting_data_gatherer.scores_by_issue_type

                            for i in range(len(issue_type_names)):
                                logger.debug(
                                    "Most relevant item with issue type %s has id %s",
                                    issue_type_names[i],
                                    boosting_data_gatherer.
                                    scores_by_issue_type[issue_type_names[i]]["mrHit"]["_id"])
                                logger.debug(
                                    "Issue type %s has label %d and probability %.3f for features %s",
                                    issue_type_names[i],
                                    predicted_labels[i],
                                    predicted_labels_probability[i][1],
                                    feature_data[i])

                            predicted_issue_type, prob, global_idx = utils.choose_issue_type(
                                predicted_labels,
                                predicted_labels_probability,
                                issue_type_names,
                                boosting_data_gatherer.scores_by_issue_type)

                            if predicted_issue_type:
                                chosen_type = scores_by_issue_type[predicted_issue_type]
                                relevant_item = chosen_type["mrHit"]["_source"]["test_item"]
                                analysis_result = AnalysisResult(testItem=analyzer_candidates.testItemId,
                                                                 issueType=predicted_issue_type,
                                                                 relevantItem=relevant_item)
                                relevant_log_id = utils.extract_real_id(chosen_type["mrHit"]["_id"])
                                test_item_log_id = utils.extract_real_id(chosen_type["compared_log"]["_id"])
                                analyzed_results_for_index.append(SuggestAnalysisResult(
                                    project=analyzer_candidates.project,
                                    testItem=analyzer_candidates.testItemId,
                                    testItemLogId=test_item_log_id,
                                    launchId=analyzer_candidates.launchId,
                                    launchName=analyzer_candidates.launchName,
                                    launchNumber=analyzer_candidates.launchNumber,
                                    issueType=predicted_issue_type,
                                    relevantItem=relevant_item,
                                    relevantLogId=relevant_log_id,
                                    isMergedLog=chosen_type["compared_log"]["_source"]["is_merged"],
                                    matchScore=round(prob * 100, 2),
                                    esScore=round(chosen_type["mrHit"]["_score"], 2),
                                    esPosition=chosen_type["mrHit"]["es_pos"],
                                    modelFeatureNames=";".join(_boosting_decision_maker.get_feature_names()),
                                    modelFeatureValues=";".join(
                                        [str(feature) for feature in feature_data[global_idx]]),
                                    modelInfo=";".join(model_info_tags),
                                    resultPosition=0,
                                    usedLogLines=analyzer_candidates.analyzerConfig.numberOfLogLines,
                                    minShouldMatch=self.find_min_should_match_threshold(
                                        analyzer_candidates.analyzerConfig),
                                    processedTime=time() - t_start_item,
                                    methodName="auto_analysis",
                                    userChoice=1))  # default choice in AA, user will change via defect change

                                results.append(analysis_result)
                                found_result = True
                                logger.debug(analysis_result)
                            else:
                                logger.debug("Test item %s has no relevant items",
                                             analyzer_candidates.testItemId)
                        else:
                            logger.debug("There are no results for test item %s",
                                         analyzer_candidates.testItemId)

                        if found_result:
                            break

                    if not found_result:
                        results_to_share[launch_id]["not_found"] += 1
                    results_to_share[launch_id]["processed_time"] += (time() - t_start_item)
                except Exception as err:
                    logger.error(err)
                    if launch_id in results_to_share:
                        results_to_share[launch_id]["errors"].append(
                            utils.extract_exception(err))
                        results_to_share[launch_id]["errors_count"] += 1
            if "amqpUrl" in self.app_config and self.app_config["amqpUrl"].strip() and analyzed_results_for_index:
                AmqpClient(self.app_config["amqpUrl"]).send_to_inner_queue(
                    self.app_config["exchangeName"], "index_suggest_info",
                    json.dumps([_info.dict() for _info in analyzed_results_for_index]))
                for launch_id in results_to_share:
                    results_to_share[launch_id]["model_info"] = list(
                        results_to_share[launch_id]["model_info"])
                AmqpClient(self.app_config["amqpUrl"]).send_to_inner_queue(
                    self.app_config["exchangeName"], "stats_info", json.dumps(results_to_share))
        except Exception as err:
            logger.error(err)
        es_query_thread.join()
        EARLY_FINISH = False
        self.queue = Queue()
        self.finished_queue = Queue()
        logger.debug("Stats info %s", results_to_share)
        logger.info("Processed %d test items. It took %.2f sec.", cnt_items_to_process, time() - t_start)
        logger.info("Finished analysis for %d launches with %d results.", cnt_launches, len(results))
        return results
