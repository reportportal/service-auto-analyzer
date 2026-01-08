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
from collections import defaultdict
from datetime import datetime
from functools import reduce
from time import time
from typing import Any, Optional

from app.amqp.amqp import AmqpClient
from app.commons import esclient, log_merger, logging, request_factory
from app.commons.esclient import EsClient
from app.commons.model.launch_objects import (
    ERROR_LOGGING_LEVEL,
    AnalysisCandidate,
    AnalysisResult,
    AnalyzerConf,
    ApplicationConfig,
    BatchLogInfo,
    Launch,
    SearchConfig,
    SuggestAnalysisResult,
    TestItem,
)
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.similarity_calculator import SimilarityCalculator
from app.ml.predictor import AutoAnalysisPredictor, PredictionResult
from app.service.analyzer_service import AnalyzerService
from app.utils import text_processing, utils

LOGGER = logging.getLogger("analyzerApp.autoAnalyzerService")

SPECIAL_FIELDS_BOOST_SCORES = [
    ("detected_message_without_params_extended", 2.0),
    ("only_numbers", 2.0),
    ("potential_status_codes", 8.0),
    ("found_tests_and_methods", 2),
    ("test_item_name", 2.0),
]


def _choose_issue_type(prediction_results: list[PredictionResult]) -> Optional[PredictionResult]:
    """Choose the best issue type from a list of prediction results.

    :param list[PredictionResult] prediction_results: List of PredictionResult objects to choose from
    :return: The PredictionResult with the highest probability, or None if no positive predictions
    """
    if not prediction_results:
        return None

    best_result = None
    max_prob = 0.0
    max_val_start_time = None

    for result in prediction_results:
        if result.label == 1:
            start_time = result.data["mrHit"]["_source"]["start_time"]
            predicted_prob = round(result.probability[1], 4)

            if (predicted_prob > max_prob) or (
                (predicted_prob == max_prob)  # noqa
                and (max_val_start_time is None or start_time > max_val_start_time)
            ):
                max_prob = predicted_prob
                best_result = result
                max_val_start_time = start_time
    return best_result


def _prepare_logs(launch: Launch, test_item: TestItem, index_name: str) -> list[dict[str, Any]]:
    unique_logs = text_processing.leave_only_unique_logs(test_item.logs)
    prepared_logs = [
        request_factory.prepare_log(launch, test_item, log, index_name)
        for log in unique_logs
        if log.logLevel >= ERROR_LOGGING_LEVEL
    ]
    results, _ = log_merger.decompose_logs_merged_and_without_duplicates(prepared_logs)
    return results


class AutoAnalyzerService(AnalyzerService):
    app_config: ApplicationConfig
    es_client: EsClient
    namespace_finder: NamespaceFinder

    def __init__(
        self,
        model_chooser: ModelChooser,
        app_config: ApplicationConfig,
        search_cfg: SearchConfig,
        es_client: Optional[EsClient] = None,
    ) -> None:
        super().__init__(model_chooser, search_cfg=search_cfg)
        self.app_config = app_config
        self.es_client = es_client or EsClient(app_config=self.app_config)
        self.namespace_finder = NamespaceFinder(app_config)

    def get_config_for_boosting(self, analyzer_config: AnalyzerConf) -> dict[str, Any]:
        min_should_match = self.find_min_should_match_threshold(analyzer_config) / 100
        return {
            "max_query_terms": self.search_cfg.MaxQueryTerms,
            "min_should_match": min_should_match,
            "min_word_length": self.search_cfg.MinWordLength,
            "filter_min_should_match_any": [],
            "filter_min_should_match": self.choose_fields_to_filter_strict(
                analyzer_config.numberOfLogLines, min_should_match
            ),
            "number_of_log_lines": analyzer_config.numberOfLogLines,
            "filter_by_test_case_hash": True,
            "boosting_model": self.search_cfg.BoostModelFolder,
            "filter_by_all_logs_should_be_similar": analyzer_config.allMessagesShouldMatch,
            "time_weight_decay": self.search_cfg.TimeWeightDecay,
        }

    def choose_fields_to_filter_strict(self, log_lines: int, min_should_match: float) -> list[str]:
        fields = ["detected_message", "message"] if log_lines == -1 else ["message"]
        if min_should_match > 0.99:
            fields.append("found_tests_and_methods")
        return fields

    def get_min_should_match_setting(self, launch: Launch) -> str:
        if launch.analyzerConfig.minShouldMatch > 0:
            return f"{launch.analyzerConfig.minShouldMatch}%"
        return self.search_cfg.MinShouldMatch

    def build_analyze_query(self, launch: Launch, log: dict[str, Any], size: int = 10) -> dict[str, Any]:
        """Build query to get similar log entries for the given log entry.

        This query is used to find similar log entries for the given log entry, the results of this query then will be
        used to find the most relevant log entry with the issue type, and then in the Gradient Boosting model to
        predict the issue type for the given log entry.
        """
        min_should_match = self.get_min_should_match_setting(launch)

        query = self.build_common_query(log, size=size)
        query = self.add_constraints_for_launches_into_query(query, launch)

        must = utils.create_path(query, ("query", "bool", "must"), [])
        must_not = utils.create_path(query, ("query", "bool", "must_not"), [])
        should = utils.create_path(query, ("query", "bool", "should"), [])
        filter_ = utils.create_path(query, ("query", "bool", "filter"), [])
        if log["_source"]["message"].strip():
            log_lines = launch.analyzerConfig.numberOfLogLines
            filter_.append({"term": {"is_merged": False}})
            if log_lines == -1:
                must.append(
                    self.build_more_like_this_query(
                        min_should_match, log["_source"]["detected_message"], field_name="detected_message", boost=4.0
                    )
                )
                if log["_source"]["stacktrace"].strip():
                    must.append(
                        self.build_more_like_this_query(
                            min_should_match, log["_source"]["stacktrace"], field_name="stacktrace", boost=2.0
                        )
                    )
                else:
                    must_not.append({"wildcard": {"stacktrace": "*"}})
            else:
                must.append(
                    self.build_more_like_this_query(
                        min_should_match, log["_source"]["message"], field_name="message", boost=4.0
                    )
                )
                should.append(
                    self.build_more_like_this_query(
                        "80%", log["_source"]["detected_message"], field_name="detected_message", boost=2.0
                    )
                )
                should.append(
                    self.build_more_like_this_query(
                        "60%", log["_source"]["stacktrace"], field_name="stacktrace", boost=1.0
                    )
                )
            should.append(
                self.build_more_like_this_query(
                    "80%", log["_source"]["merged_small_logs"], field_name="merged_small_logs", boost=0.5
                )
            )
        else:
            filter_.append({"term": {"is_merged": True}})
            must_not.append({"wildcard": {"message": "*"}})
            must.append(
                self.build_more_like_this_query(
                    min_should_match, log["_source"]["merged_small_logs"], field_name="merged_small_logs", boost=2.0
                )
            )

        if log["_source"]["found_exceptions"].strip():
            must.append(
                self.build_more_like_this_query(
                    "1",
                    log["_source"]["found_exceptions"],
                    field_name="found_exceptions",
                    boost=8.0,
                    override_min_should_match="1",
                )
            )
        for field, boost_score in SPECIAL_FIELDS_BOOST_SCORES:
            if log["_source"][field].strip():
                should.append(
                    self.build_more_like_this_query(
                        "1", log["_source"][field], field_name=field, boost=boost_score, override_min_should_match="1"
                    )
                )

        return self.add_query_with_start_time_decay(query, log["_source"]["start_time"])

    def build_query_with_no_defect(self, launch: Launch, log: dict[str, Any], size: int = 10) -> dict[str, Any]:
        min_should_match = self.get_min_should_match_setting(launch)
        query: dict[str, Any] = {
            "size": size,
            "sort": ["_score", {"start_time": "desc"}],
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"log_level": {"gte": ERROR_LOGGING_LEVEL}}},
                        {"exists": {"field": "issue_type"}},
                    ],
                    "must_not": [
                        {"term": {"issue_type": "ti001"}},
                        {"term": {"test_item": log["_source"]["test_item"]}},
                    ],
                    "must": [{"term": {"test_case_hash": log["_source"]["test_case_hash"]}}],
                    "should": [],
                }
            },
        }
        query = self.add_constraints_for_launches_into_query(query, launch)
        if log["_source"]["message"].strip():
            query["query"]["bool"]["filter"].append({"term": {"is_merged": False}})
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query(min_should_match, log["_source"]["message"], field_name="message")
            )
        else:
            query["query"]["bool"]["filter"].append({"term": {"is_merged": True}})
            query["query"]["bool"]["must_not"].append({"wildcard": {"message": "*"}})
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query(
                    min_should_match, log["_source"]["merged_small_logs"], field_name="merged_small_logs"
                )
            )
        if log["_source"]["found_exceptions"].strip():
            query["query"]["bool"]["must"].append(
                self.build_more_like_this_query(
                    "1",
                    log["_source"]["found_exceptions"],
                    field_name="found_exceptions",
                    boost=8.0,
                    override_min_should_match="1",
                )
            )
        utils.append_potential_status_codes(query, log, max_query_terms=self.search_cfg.MaxQueryTerms)
        return self.add_query_with_start_time_decay(query, log["_source"]["start_time"])

    def leave_only_similar_logs(
        self, candidates_with_no_defect: list[tuple[dict[str, Any], dict[str, Any]]], boosting_config: dict[str, Any]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        new_results = []
        for log_info, search_res in candidates_with_no_defect:
            no_defect_candidate_exists = False
            for log in search_res["hits"]["hits"]:
                if log["_source"]["issue_type"][:2].lower() in ["nd", "ti"]:
                    no_defect_candidate_exists = True
            new_search_res = []
            _similarity_calculator = SimilarityCalculator()
            if no_defect_candidate_exists:
                sim_dict = _similarity_calculator.find_similarity(
                    [(log_info, search_res)], ["message", "merged_small_logs"]
                )
                for obj in search_res["hits"]["hits"]:
                    group_id = (str(obj["_id"]), str(log_info["_id"]))
                    if group_id in sim_dict["message"]:
                        sim_val = sim_dict["message"][group_id]
                        if sim_val.both_empty:
                            sim_val = sim_dict["merged_small_logs"][group_id]
                        threshold = boosting_config["min_should_match"]
                        if not sim_val.both_empty and sim_val.similarity >= threshold:
                            new_search_res.append(obj)
            new_results.append((log_info, {"hits": {"hits": new_search_res}}))
        return new_results

    def filter_by_all_logs_should_be_similar(
        self, candidates_with_no_defect: list[tuple[dict[str, Any], dict[str, Any]]], boosting_config: dict[str, Any]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
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

    def find_relevant_with_no_defect(
        self, candidates_with_no_defect: list[tuple[dict[str, Any], dict[str, Any]]], boosting_config: dict[str, Any]
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        candidates_with_no_defect = self.leave_only_similar_logs(candidates_with_no_defect, boosting_config)
        candidates_with_no_defect = self.filter_by_all_logs_should_be_similar(
            candidates_with_no_defect, boosting_config
        )
        for log_info, search_res in candidates_with_no_defect:
            latest_type = None
            latest_item = None
            latest_date = None
            for obj in search_res["hits"]["hits"]:
                LOGGER.debug(
                    "%s %s %s", obj["_source"]["start_time"], obj["_source"]["issue_type"], obj["_source"]["test_item"]
                )
                start_time = datetime.strptime(obj["_source"]["start_time"], "%Y-%m-%d %H:%M:%S")
                if latest_date is None or latest_date < start_time:
                    latest_type = obj["_source"]["issue_type"]
                    latest_item = obj
                    latest_date = start_time
            if latest_type and latest_type[:2].lower() in ["nd", "ti"]:
                return [(log_info, {"hits": {"hits": [latest_item]}})]
        return []

    def _process_batch(
        self,
        test_item_dict: dict[int, list[int]],
        batches: list[str],
        batch_logs: list[BatchLogInfo],
    ) -> list[AnalysisCandidate]:
        t_start = time()
        search_results = self.es_client.es_client.msearch(body="\n".join(batches) + "\n")
        partial_res = search_results["responses"] if search_results else []
        if not partial_res:
            LOGGER.warning("No search results for batches")
            return []
        res_num = reduce(lambda a, b: a + b, [len(res["hits"]["hits"]) for res in partial_res], 0)
        LOGGER.info(f"Found {res_num} items by FTS (KNN)")
        LOGGER.debug(f"Items for analysis by FTS (KNN): {json.dumps(search_results)}")

        avg_time_processed = (time() - t_start) / (len(partial_res) if partial_res else 1)
        analysis_candidates = []
        for test_item_id in test_item_dict:
            candidates = []
            candidates_with_no_defect = []
            time_processed = 0.0
            batch_log_info = None
            for idx in test_item_dict[test_item_id]:
                batch_log_info = batch_logs[idx]
                if batch_log_info.query_type == "without no defect":
                    candidates.append((batch_log_info.log_info, partial_res[idx]))
                if batch_log_info.query_type == "with no defect":
                    candidates_with_no_defect.append((batch_log_info.log_info, partial_res[idx]))
                time_processed += avg_time_processed
            if batch_log_info:
                analysis_candidates.append(
                    AnalysisCandidate(
                        analyzerConfig=batch_log_info.analyzerConfig,
                        testItemId=batch_log_info.testItemId,
                        project=batch_log_info.project,
                        launchId=batch_log_info.launchId,
                        launchName=batch_log_info.launchName,
                        launchNumber=batch_log_info.launchNumber,
                        timeProcessed=time_processed,
                        candidates=candidates,
                        candidatesWithNoDefect=candidates_with_no_defect,
                    )
                )
        return analysis_candidates

    def _should_stop_processing(self, test_items_processed: int) -> bool:
        """Check if we've reached the maximum number of test items to process."""
        if test_items_processed >= self.search_cfg.MaxAutoAnalysisItemsToProcess:
            LOGGER.info("Only first %d test items were taken", self.search_cfg.MaxAutoAnalysisItemsToProcess)
            return True
        return False

    def _get_analysis_candidates(
        self,
        launches: list[Launch],
        max_batch_size: int = 30,
    ) -> list[AnalysisCandidate]:
        t_start = time()
        all_candidates = []
        batches = []
        batch_logs = []
        index_in_batch = 0
        test_item_dict = defaultdict(list)
        batch_size = 5
        n_first_blocks = 3
        test_items_number_to_process = 0

        try:
            for launch in launches:
                index_name = esclient.get_index_name(
                    launch.project, self.app_config.esProjectIndexPrefix, "rp_log_item"
                )
                if not self.es_client.index_exists(index_name):
                    continue

                for test_item in launch.testItems:
                    if self._should_stop_processing(test_items_number_to_process):
                        break

                    logs = _prepare_logs(launch, test_item, index_name)
                    for log in logs:
                        message = log["_source"]["message"].strip()
                        merged_logs = log["_source"]["merged_small_logs"].strip()
                        if log["_source"]["log_level"] < ERROR_LOGGING_LEVEL or (not message and not merged_logs):
                            continue
                        for query_type, query in [
                            ("without no defect", self.build_analyze_query(launch, log)),
                            ("with no defect", self.build_query_with_no_defect(launch, log)),
                        ]:
                            full_query = "{}\n{}".format(json.dumps({"index": index_name}), json.dumps(query))
                            batches.append(full_query)
                            batch_logs.append(
                                BatchLogInfo(
                                    analyzerConfig=launch.analyzerConfig,
                                    testItemId=test_item.testItemId,
                                    log_info=log,
                                    query_type=query_type,
                                    project=launch.project,
                                    launchId=launch.launchId,
                                    launchName=launch.launchName,
                                    launchNumber=launch.launchNumber,
                                )
                            )
                            test_item_dict[test_item.testItemId].append(index_in_batch)
                            index_in_batch += 1
                    if n_first_blocks <= 0:
                        batch_size = max_batch_size
                    if len(batches) >= batch_size:
                        n_first_blocks -= 1
                        batch_candidates = self._process_batch(dict(test_item_dict), batches, batch_logs)
                        all_candidates.extend(batch_candidates)
                        batches = []
                        batch_logs = []
                        test_item_dict.clear()
                        index_in_batch = 0
                    test_items_number_to_process += 1

            # Process remaining batches
            if len(batches) > 0:
                batch_candidates = self._process_batch(dict(test_item_dict), batches, batch_logs)
                all_candidates.extend(batch_candidates)

        except Exception as exc:
            LOGGER.exception("Error in ES query", exc_info=exc)

        LOGGER.info("Es queries finished %.2f s.", time() - t_start)
        return all_candidates

    @utils.ignore_warnings
    def analyze_logs(self, launches: list[Launch]) -> list[AnalysisResult]:
        cnt_launches = len(launches)
        LOGGER.info(f"Started analysis for {cnt_launches} launches")

        analyzed_results_for_index = []
        t_start = time()
        results = []
        results_to_share: dict[int, dict[str, Any]] = {}
        cnt_items_to_process = 0
        chosen_namespaces = {}

        try:
            # Generate all analysis candidates using batch processing
            all_candidates = self._get_analysis_candidates(launches)

            # Group by Project ID and Launch ID
            all_candidates_by_launch_and_project: dict[tuple[int, int], list[AnalysisCandidate]] = defaultdict(list)
            for candidate in all_candidates:
                all_candidates_by_launch_and_project[candidate.project, candidate.launchId].append(candidate)

            # Process candidates sequentially
            for (project_id, launch_id), analyzer_candidates in all_candidates_by_launch_and_project.items():
                if not analyzer_candidates:
                    LOGGER.info(f"No candidates found for project {project_id}, launch {launch_id}")
                    continue

                # Create predictor for auto analysis
                analyzer_config = analyzer_candidates[0].analyzerConfig  # Same for all candidates in the launch
                boosting_config = self.get_config_for_boosting(analyzer_config)
                predictor = AutoAnalysisPredictor(
                    model_chooser=self.model_chooser,
                    project_id=project_id,
                    boosting_config=boosting_config,
                    custom_model_prob=self.search_cfg.ProbabilityForCustomModelAutoAnalysis,
                    hash_source=launch_id,
                )

                for analyzer_candidate in analyzer_candidates:
                    try:
                        if launch_id not in results_to_share:
                            results_to_share[launch_id] = {
                                "not_found": 0,
                                "items_to_process": 0,
                                "processed_time": 0,
                                "launch_id": launch_id,
                                "launch_name": analyzer_candidate.launchName,
                                "project_id": project_id,
                                "method": "auto_analysis",
                                "gather_date": datetime.now().strftime("%Y-%m-%d"),
                                "gather_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "number_of_log_lines": analyzer_candidate.analyzerConfig.numberOfLogLines,
                                "min_should_match": self.find_min_should_match_threshold(
                                    analyzer_candidate.analyzerConfig
                                ),
                                "model_info": [],
                                "module_version": [self.app_config.appVersion],
                                "errors": [],
                                "errors_count": 0,
                            }

                        t_start_item = time()
                        cnt_items_to_process += 1
                        results_to_share[launch_id]["items_to_process"] += 1
                        results_to_share[launch_id]["processed_time"] += analyzer_candidate.timeProcessed

                        if project_id not in chosen_namespaces:
                            chosen_namespaces[project_id] = self.namespace_finder.get_chosen_namespaces(project_id)
                        boosting_config["chosen_namespaces"] = chosen_namespaces[project_id]

                        relevant_with_no_defect_candidate = self.find_relevant_with_no_defect(
                            analyzer_candidate.candidatesWithNoDefect, boosting_config
                        )

                        candidates_to_check = []
                        if relevant_with_no_defect_candidate:
                            candidates_to_check.append(relevant_with_no_defect_candidate)
                        candidates_to_check.append(analyzer_candidate.candidates)

                        found_result = False

                        for candidates in candidates_to_check:
                            # Use predictor for the complete prediction workflow
                            prediction_results = predictor.predict(candidates)

                            if not prediction_results:
                                LOGGER.debug(f"There are no results for test item {analyzer_candidate.testItemId}")
                                continue

                            # Get model info tags from the first result (same for all results)
                            model_info_tags = prediction_results[0].model_info_tags
                            new_model_info_tags = set(results_to_share[launch_id]["model_info"])
                            new_model_info_tags.update(model_info_tags)
                            results_to_share[launch_id]["model_info"] = list(new_model_info_tags)

                            # Debug logging
                            for result in prediction_results:
                                test_item_id = analyzer_candidate.testItemId
                                relevant_test_item_id = result.data["mrHit"]["_source"]["test_item"]
                                log_id = result.data["mrHit"]["_id"]
                                LOGGER.debug(
                                    f"Most relevant ID for item '{test_item_id}' is '{relevant_test_item_id}', it has "
                                    f"issue type '{result.identity}', log ID '{log_id}' and probability: "
                                    + str(round(result.probability[1], 4))
                                )

                            best = _choose_issue_type(prediction_results)
                            if not best:
                                LOGGER.debug(f"Test item {analyzer_candidate.testItemId} has no relevant items")
                                continue

                            predicted_issue_type = best.identity
                            prob = round(best.probability[1], 4)
                            chosen_type = best.data
                            feature_names = None
                            feature_values = None
                            if best.feature_info:
                                feature_names = ";".join([str(f_id) for f_id in best.feature_info.feature_ids])
                                feature_values = ";".join([str(f) for f in best.feature_info.feature_data])

                            relevant_item = chosen_type["mrHit"]["_source"]["test_item"]
                            analysis_result = AnalysisResult(
                                testItem=analyzer_candidate.testItemId,
                                issueType=predicted_issue_type,
                                relevantItem=relevant_item,
                            )
                            relevant_log_id = utils.extract_real_id(chosen_type["mrHit"]["_id"])
                            test_item_log_id = utils.extract_real_id(chosen_type["compared_log"]["_id"])
                            analyzed_results_for_index.append(
                                SuggestAnalysisResult(
                                    project=analyzer_candidate.project,
                                    testItem=analyzer_candidate.testItemId,
                                    testItemLogId=test_item_log_id,
                                    launchId=analyzer_candidate.launchId,
                                    launchName=analyzer_candidate.launchName,
                                    launchNumber=analyzer_candidate.launchNumber,
                                    issueType=predicted_issue_type,
                                    relevantItem=relevant_item,
                                    relevantLogId=relevant_log_id,
                                    isMergedLog=chosen_type["compared_log"]["_source"]["is_merged"],
                                    matchScore=round(prob * 100, 2),
                                    esScore=round(chosen_type["mrHit"]["_score"], 2),
                                    esPosition=best.original_position,
                                    modelFeatureNames=feature_names,
                                    modelFeatureValues=feature_values,
                                    modelInfo=";".join(model_info_tags),
                                    resultPosition=0,
                                    usedLogLines=analyzer_candidate.analyzerConfig.numberOfLogLines,
                                    minShouldMatch=self.find_min_should_match_threshold(
                                        analyzer_candidate.analyzerConfig
                                    ),
                                    processedTime=time() - t_start_item,
                                    methodName="auto_analysis",
                                    userChoice=1,
                                )
                            )  # default choice in AA, user will change via defect change

                            results.append(analysis_result)
                            found_result = True
                            LOGGER.debug(analysis_result)

                            if found_result:
                                break

                        if not found_result:
                            results_to_share[launch_id]["not_found"] += 1
                        results_to_share[launch_id]["processed_time"] += time() - t_start_item

                    except Exception as exc:
                        LOGGER.exception(
                            f"Unable to process candidate for analysis {analyzer_candidate.testItemId}", exc_info=exc
                        )
                        if launch_id in results_to_share:
                            results_to_share[launch_id]["errors"].append(utils.extract_exception(exc))
                            results_to_share[launch_id]["errors_count"] += 1

            # Send results to AMQP if configured
            if self.app_config.amqpUrl and analyzed_results_for_index:
                amqp_client = AmqpClient(self.app_config)
                amqp_client.send_to_inner_queue(
                    "index_suggest_info", json.dumps([_info.model_dump() for _info in analyzed_results_for_index])
                )
                for launch_id in results_to_share:
                    results_to_share[launch_id]["model_info"] = list(results_to_share[launch_id]["model_info"])
                amqp_client.send_to_inner_queue("stats_info", json.dumps(results_to_share))
                amqp_client.close()

        except Exception as exc:
            LOGGER.exception("Unable to process analysis candidates", exc_info=exc)

        LOGGER.debug(f"Stats info: {json.dumps(results_to_share)}")
        LOGGER.info(f"Processed {cnt_items_to_process} test items. It took {time() - t_start:.2f} sec.")
        LOGGER.info(f"Finished analysis for {cnt_launches} launches with {len(results)} results.")
        return results
