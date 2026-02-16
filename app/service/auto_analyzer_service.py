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

import json
from collections import defaultdict
from datetime import datetime
from time import time
from typing import Any, Optional

from app.amqp.amqp import AmqpClient
from app.commons import logging, request_factory
from app.commons.model import LogItemIndexData
from app.commons.model.db import Hit
from app.commons.model.launch_objects import (
    ERROR_LOGGING_LEVEL,
    AnalysisCandidate,
    AnalysisResult,
    AnalyzerConf,
    ApplicationConfig,
    Launch,
    SearchConfig,
    SuggestAnalysisResult,
    TestItem,
)
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.os_client import OsClient
from app.ml.predictor import AutoAnalysisPredictor
from app.service.analyzer_service import AnalyzerService
from app.utils import utils
from app.utils.os_migration import (
    bucket_sort_logs_by_similarity,
    build_search_results,
    extract_inner_hit_logs,
    get_request_logs,
)

LOGGER = logging.getLogger("analyzerApp.autoAnalyzerService")

SPECIAL_FIELDS_BOOST_SCORES = [
    ("detected_message_without_params_extended", 2.0),
    ("only_numbers", 2.0),
    ("potential_status_codes", 8.0),
    ("found_tests_and_methods", 2.0),
    ("test_item_name", 2.0),
]

INNER_HITS_SOURCE = [
    "logs.log_id",
    "logs.log_time",
    "logs.log_level",
    "logs.cluster_id",
    "logs.cluster_message",
    "logs.cluster_with_numbers",
    "logs.original_message",
    "logs.message",
    "logs.message_extended",
    "logs.message_without_params_extended",
    "logs.message_without_params_and_brackets",
    "logs.detected_message",
    "logs.detected_message_with_numbers",
    "logs.detected_message_extended",
    "logs.detected_message_without_params_extended",
    "logs.detected_message_without_params_and_brackets",
    "logs.stacktrace",
    "logs.stacktrace_extended",
    "logs.only_numbers",
    "logs.potential_status_codes",
    "logs.found_exceptions",
    "logs.found_exceptions_extended",
    "logs.found_tests_and_methods",
    "logs.urls",
    "logs.paths",
    "logs.message_params",
    "logs.whole_message",
]

TEST_ITEM_SOURCE_FIELDS = [
    "test_item_id",
    "test_item_name",
    "unique_id",
    "test_case_hash",
    "launch_id",
    "launch_name",
    "issue_type",
    "is_auto_analyzed",
    "start_time",
]

UNSHORTENED_MESSAGE_FIELDS = [
    "detected_message_without_params_extended",
    "detected_message_extended",
]

SHORTENED_MESSAGE_FIELDS = [
    "message",
    "detected_message",
]


def choose_fields_to_filter_strict(log_lines: int, min_should_match: float) -> list[str]:
    fields = list(UNSHORTENED_MESSAGE_FIELDS if log_lines == -1 else SHORTENED_MESSAGE_FIELDS)
    if min_should_match > 0.99:
        fields.append("found_tests_and_methods")
    return fields


def prepare_request_logs_for_launch(
    launch: Launch,
) -> list[tuple[TestItem, list]]:
    prepared = request_factory.prepare_test_items(
        launch,
        number_of_logs_to_index=launch.analyzerConfig.numberOfLogsToIndex,
        minimal_log_level=launch.analyzerConfig.minimumLogLevel,
        similarity_threshold_to_drop=launch.analyzerConfig.similarityThresholdToDrop,
    )
    source_test_items = {item.testItemId: item for item in launch.testItems}
    request_logs_by_test_item: list[tuple[TestItem, list]] = []

    for prepared_item in prepared:
        try:
            test_item_id = int(prepared_item.test_item_id)
        except (TypeError, ValueError):
            continue
        source_test_item = source_test_items.get(test_item_id)
        if source_test_item is None:
            continue
        request_logs = get_request_logs(prepared_item, issue_type="")
        if not request_logs:
            continue
        request_logs_by_test_item.append((source_test_item, request_logs))

    return request_logs_by_test_item


class AutoAnalyzerService(AnalyzerService):
    app_config: ApplicationConfig
    os_client: OsClient
    namespace_finder: NamespaceFinder
    model_chooser: ModelChooser

    def __init__(
        self,
        model_chooser: ModelChooser,
        app_config: ApplicationConfig,
        search_cfg: SearchConfig,
        os_client: Optional[OsClient] = None,
    ) -> None:
        super().__init__(search_cfg=search_cfg)
        self.model_chooser = model_chooser
        self.app_config = app_config
        self.os_client = os_client or OsClient(app_config=self.app_config)
        self.namespace_finder = NamespaceFinder(app_config)

    def _get_config_for_boosting(self, analyzer_config: AnalyzerConf) -> dict[str, Any]:
        min_should_match = self.find_min_should_match_threshold(analyzer_config) / 100
        return {
            "max_query_terms": self.search_cfg.MaxQueryTerms,
            "min_should_match": min_should_match,
            "min_word_length": self.search_cfg.MinWordLength,
            "filter_min_should_match_any": [],
            "filter_min_should_match": choose_fields_to_filter_strict(
                analyzer_config.numberOfLogLines, min_should_match
            ),
            "number_of_log_lines": analyzer_config.numberOfLogLines,
            "filter_by_test_case_hash": True,
            "boosting_model": self.search_cfg.BoostModelFolder,
            "filter_by_all_logs_should_be_similar": analyzer_config.allMessagesShouldMatch,
            "time_weight_decay": self.search_cfg.TimeWeightDecay,
        }

    def _get_min_should_match_setting(self, launch: Launch) -> str:
        if launch.analyzerConfig.minShouldMatch > 0:
            return f"{launch.analyzerConfig.minShouldMatch}%"
        return self.search_cfg.MinShouldMatch

    def _build_nested_analyze_query(
        self,
        launch: Launch,
        request_log,
        size: int = 10,
    ) -> dict[str, Any]:
        min_should_match = self._get_min_should_match_setting(launch)
        log_lines = launch.analyzerConfig.numberOfLogLines
        nested_should: list[dict[str, Any]] = []

        for message_field in choose_fields_to_filter_strict(
            log_lines, self.find_min_should_match_threshold(launch.analyzerConfig) / 100
        ):
            field_value = getattr(request_log, message_field, "").strip()
            if field_value:
                nested_should.append(
                    utils.build_more_like_this_query(
                        min_should_match,
                        field_value,
                        field_name=f"logs.{message_field}",
                        boost=4.0,
                        max_query_terms=self.search_cfg.MaxQueryTerms,
                    )
                )

        stacktrace_text = request_log.stacktrace_extended.strip()
        if stacktrace_text:
            stacktrace_boost = 2.0 if log_lines == -1 else 1.0
            nested_should.append(
                utils.build_more_like_this_query(
                    min_should_match,
                    stacktrace_text,
                    field_name="logs.stacktrace_extended",
                    boost=stacktrace_boost,
                    max_query_terms=self.search_cfg.MaxQueryTerms,
                )
            )

        found_exceptions = request_log.found_exceptions.strip()
        if found_exceptions:
            nested_should.append(
                utils.build_more_like_this_query(
                    "1",
                    found_exceptions,
                    field_name="logs.found_exceptions",
                    boost=8.0,
                    override_min_should_match="1",
                    max_query_terms=self.search_cfg.MaxQueryTerms,
                )
            )

        potential_status_codes = request_log.potential_status_codes.strip()
        if potential_status_codes:
            number_of_codes = str(len(set(potential_status_codes.split())))
            nested_should.append(
                utils.build_more_like_this_query(
                    "1",
                    potential_status_codes,
                    field_name="logs.potential_status_codes",
                    boost=8.0,
                    override_min_should_match=number_of_codes,
                    max_query_terms=self.search_cfg.MaxQueryTerms,
                )
            )

        for field_name, boost_score in SPECIAL_FIELDS_BOOST_SCORES:
            field_value = getattr(request_log, field_name, "").strip()
            if field_value:
                target_field = field_name if field_name == "test_item_name" else f"logs.{field_name}"
                nested_should.append(
                    utils.build_more_like_this_query(
                        "1",
                        field_value,
                        field_name=target_field,
                        boost=boost_score,
                        override_min_should_match="1",
                        max_query_terms=self.search_cfg.MaxQueryTerms,
                    )
                )

        if not nested_should:
            return {}

        nested_query = {
            "nested": {
                "path": "logs",
                "score_mode": launch.analyzerConfig.searchScoreMode,
                "query": {"bool": {"should": nested_should}},
                "inner_hits": {
                    "size": 5,
                    "_source": INNER_HITS_SOURCE,
                },
            }
        }

        # Issue type restrictions for analysis: do not include TI and ND
        issue_type_conditions = utils.prepare_restrictions_by_issue_type(filter_no_defect=True)

        query: dict[str, Any] = {
            "size": size,
            "sort": ["_score", {"start_time": "desc"}],
            "_source": TEST_ITEM_SOURCE_FIELDS,
            "query": {
                "bool": {
                    "filter": [{"exists": {"field": "issue_type"}}],
                    "must_not": issue_type_conditions + [{"term": {"test_item_id": str(request_log.test_item)}}],
                    "must": [nested_query],
                    "should": [
                        {
                            "term": {
                                "test_case_hash": {
                                    "value": request_log.test_case_hash,
                                    "boost": abs(self.search_cfg.BoostTestCaseHash),
                                }
                            }
                        },
                    ],
                }
            },
        }
        utils.append_aa_ma_boosts(query, self.search_cfg)
        query = self.add_constraints_for_launches_into_query(query, launch)
        return self.add_query_with_start_time_decay(query, request_log.start_time)

    def _query_candidates_for_test_item(
        self,
        launch: Launch,
        request_logs: list,
    ) -> list[tuple[LogItemIndexData, list[Hit[LogItemIndexData]]]]:
        all_queries: list[dict[str, Any]] = []
        query_request_logs = []
        for request_log in request_logs:
            if request_log.log_level < ERROR_LOGGING_LEVEL:
                continue
            if not request_log.message.strip():
                continue
            query = self._build_nested_analyze_query(launch, request_log)
            if not query:
                continue
            all_queries.append({})
            all_queries.append(query)
            query_request_logs.append(request_log)

        if not all_queries:
            return []

        seen_test_item_ids: set[str] = set()
        unique_hits = []
        for hit in self.os_client.msearch(launch.project, all_queries):
            test_item_id = hit.source.test_item_id
            if test_item_id in seen_test_item_ids:
                continue
            seen_test_item_ids.add(test_item_id)
            unique_hits.append(hit)

        if not unique_hits:
            return []

        found_log_hits = extract_inner_hit_logs(unique_hits)
        if not found_log_hits:
            return []

        buckets = bucket_sort_logs_by_similarity(query_request_logs, found_log_hits)
        return build_search_results(query_request_logs, buckets)

    def _should_stop_processing(self, test_items_processed: int) -> bool:
        if test_items_processed >= self.search_cfg.MaxAutoAnalysisItemsToProcess:
            LOGGER.info("Only first %d test items were taken", self.search_cfg.MaxAutoAnalysisItemsToProcess)
            return True
        return False

    def _get_analysis_candidates(
        self,
        launches: list[Launch],
    ) -> tuple[list[AnalysisCandidate], dict[str, TestItem]]:
        t_start = time()
        all_candidates: list[AnalysisCandidate] = []
        request_log_to_test_item: dict[str, TestItem] = {}
        processed_items = 0

        for launch in launches:
            request_logs_by_test_item = prepare_request_logs_for_launch(launch)
            for source_test_item, request_logs in request_logs_by_test_item:
                if self._should_stop_processing(processed_items):
                    LOGGER.info("Collected %d candidates for analysis", len(all_candidates))
                    LOGGER.info("Os queries finished %.2f s.", time() - t_start)
                    return all_candidates, request_log_to_test_item
                for request_log in request_logs:
                    request_log_to_test_item[request_log.log_id] = source_test_item

                item_start = time()
                search_results = self._query_candidates_for_test_item(launch, request_logs)
                all_candidates.append(
                    AnalysisCandidate(
                        analyzerConfig=launch.analyzerConfig,
                        testItemId=source_test_item.testItemId,
                        project=launch.project,
                        launchId=launch.launchId,
                        launchName=launch.launchName,
                        launchNumber=launch.launchNumber,
                        timeProcessed=time() - item_start,
                        candidates=search_results,
                        candidatesWithNoDefect=[],
                    )
                )
                processed_items += 1

        LOGGER.info("Os queries finished %.2f s.", time() - t_start)
        return all_candidates, request_log_to_test_item

    @utils.ignore_warnings
    def analyze_logs(self, launches: list[Launch]) -> list[AnalysisResult]:
        cnt_launches = len(launches)
        LOGGER.info(f"Started analysis for {cnt_launches} launches")

        analyzed_results_for_index: list[SuggestAnalysisResult] = []
        t_start = time()
        results: list[AnalysisResult] = []
        results_to_share: dict[int, dict[str, Any]] = {}
        cnt_items_to_process = 0
        chosen_namespaces: dict[int, dict[str, int]] = {}

        try:
            all_candidates, request_log_to_test_item = self._get_analysis_candidates(launches)
            all_candidates_by_launch_and_project: dict[tuple[int, int], list[AnalysisCandidate]] = defaultdict(list)
            for candidate in all_candidates:
                all_candidates_by_launch_and_project[candidate.project, candidate.launchId].append(candidate)

            for (project_id, launch_id), analyzer_candidates in all_candidates_by_launch_and_project.items():
                if not analyzer_candidates:
                    LOGGER.info(f"No candidates found for project {project_id}, launch {launch_id}")
                    continue

                analyzer_config = analyzer_candidates[0].analyzerConfig
                boosting_config = self._get_config_for_boosting(analyzer_config)
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

                        prediction_results = predictor.predict(analyzer_candidate.candidates)
                        if not prediction_results:
                            LOGGER.debug(f"There are no results for test item {analyzer_candidate.testItemId}")
                            results_to_share[launch_id]["not_found"] += 1
                            results_to_share[launch_id]["processed_time"] += time() - t_start_item
                            continue

                        model_info_tags = prediction_results[0].model_info_tags
                        new_model_info_tags = set(results_to_share[launch_id]["model_info"])
                        new_model_info_tags.update(model_info_tags)
                        results_to_share[launch_id]["model_info"] = list(new_model_info_tags)

                        positive_predictions = [result for result in prediction_results if result.label == 1]
                        if not positive_predictions:
                            LOGGER.debug(f"Test item {analyzer_candidate.testItemId} has no positive predictions")
                            results_to_share[launch_id]["not_found"] += 1
                            results_to_share[launch_id]["processed_time"] += time() - t_start_item
                            continue

                        grouped_predictions = utils.group_predictions_by_test_item(positive_predictions)
                        ranked_predictions = utils.score_and_rank_test_items(grouped_predictions)
                        if not ranked_predictions:
                            LOGGER.debug(f"Test item {analyzer_candidate.testItemId} has no ranked predictions")
                            results_to_share[launch_id]["not_found"] += 1
                            results_to_share[launch_id]["processed_time"] += time() - t_start_item
                            continue

                        weighted_score, best = ranked_predictions[0]
                        chosen_type = best.data
                        compared_log = chosen_type["compared_log"]
                        source_test_item = request_log_to_test_item.get(compared_log.log_id)
                        analyzed_test_item_id = (
                            source_test_item.testItemId if source_test_item else analyzer_candidate.testItemId
                        )

                        feature_names = None
                        feature_values = None
                        if best.feature_info:
                            feature_names = ";".join([str(f_id) for f_id in best.feature_info.feature_ids])
                            feature_values = ";".join([str(f) for f in best.feature_info.feature_data])

                        predicted_issue_type = best.identity
                        relevant_item = chosen_type["mrHit"].source.test_item
                        relevant_log_id = utils.extract_real_id(chosen_type["mrHit"].id)
                        test_item_log_id = utils.extract_real_id(compared_log.log_id)

                        analysis_result = AnalysisResult(
                            testItem=analyzed_test_item_id,
                            issueType=predicted_issue_type,
                            relevantItem=relevant_item,
                        )
                        analyzed_results_for_index.append(
                            SuggestAnalysisResult(
                                project=analyzer_candidate.project,
                                testItem=analyzed_test_item_id,
                                testItemLogId=test_item_log_id,
                                launchId=analyzer_candidate.launchId,
                                launchName=analyzer_candidate.launchName,
                                launchNumber=analyzer_candidate.launchNumber,
                                issueType=predicted_issue_type,
                                relevantItem=relevant_item,
                                relevantLogId=relevant_log_id,
                                isMergedLog=False,
                                matchScore=round(weighted_score * 100, 2),
                                esScore=round(chosen_type["mrHit"].score, 2),
                                esPosition=best.original_position,
                                modelFeatureNames=feature_names,
                                modelFeatureValues=feature_values,
                                modelInfo=";".join(model_info_tags),
                                resultPosition=0,
                                usedLogLines=analyzer_candidate.analyzerConfig.numberOfLogLines,
                                minShouldMatch=self.find_min_should_match_threshold(analyzer_candidate.analyzerConfig),
                                processedTime=time() - t_start_item,
                                methodName="auto_analysis",
                                userChoice=1,
                            )
                        )
                        results.append(analysis_result)
                        LOGGER.debug(analysis_result)
                        results_to_share[launch_id]["processed_time"] += time() - t_start_item
                    except Exception as exc:
                        LOGGER.exception(
                            f"Unable to process candidate for analysis {analyzer_candidate.testItemId}", exc_info=exc
                        )
                        if launch_id in results_to_share:
                            results_to_share[launch_id]["errors"].append(utils.extract_exception(exc))
                            results_to_share[launch_id]["errors_count"] += 1

            for launch_id in results_to_share:
                results_to_share[launch_id]["model_info"] = list(results_to_share[launch_id]["model_info"])

            if self.app_config.amqpUrl and analyzed_results_for_index:
                amqp_client = AmqpClient(self.app_config)
                amqp_client.send_to_inner_queue(
                    "index_suggest_info", json.dumps([_info.model_dump() for _info in analyzed_results_for_index])
                )
                amqp_client.close()
        except Exception as exc:
            LOGGER.exception("Unable to process analysis candidates", exc_info=exc)

        LOGGER.debug(f"Stats info: {json.dumps(results_to_share)}")
        LOGGER.info(f"Processed {cnt_items_to_process} test items. It took {time() - t_start:.2f} sec.")
        LOGGER.info(f"Finished analysis for {cnt_launches} launches with {len(results)} results.")
        return results
