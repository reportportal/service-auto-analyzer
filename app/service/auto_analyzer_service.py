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
    AnalysisCandidate,
    AnalysisResult,
    AnalyzerConf,
    ApplicationConfig,
    Launch,
    SearchConfig,
    TestItem,
)
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.os_client import OsClient
from app.ml.predictor import AutoAnalysisPredictor, PredictionResult
from app.service.analyzer_service import AnalyzerService
from app.utils import utils
from app.utils.os_migration import (
    construct_analysis_query,
    extract_inner_hit_logs,
    get_request_logs,
)

LOGGER = logging.getLogger("analyzerApp.autoAnalyzerService")

LOG_FIELDS_BOOST_SCORES = [
    ("detected_message_without_params_extended", utils.BOOST_SUPPORTING),
    ("only_numbers", utils.BOOST_SUPPORTING),
    ("found_tests_and_methods", utils.BOOST_SUPPORTING),
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
) -> list[tuple[TestItem, list[LogItemIndexData]]]:
    prepared = request_factory.prepare_test_items(
        launch,
        number_of_logs_to_index=launch.analyzerConfig.numberOfLogsToIndex,
        minimal_log_level=launch.analyzerConfig.minimumLogLevel,
        similarity_threshold_to_drop=launch.analyzerConfig.similarityThresholdToDrop,
    )
    source_test_items = {item.testItemId: item for item in launch.testItems}
    request_logs_by_test_item: list[tuple[TestItem, list[LogItemIndexData]]] = []

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


def to_analysis_result(candidate: AnalysisCandidate, result: PredictionResult) -> AnalysisResult:
    relevant_item = result.data.mrHit.source.test_item
    analysis_result = AnalysisResult(
        testItem=candidate.testItemId,
        issueType=result.identity,
        relevantItem=relevant_item,
    )
    return analysis_result


def log_query_results(start_time: float | int, all_candidates: list[AnalysisCandidate]):
    LOGGER.info("Collected %d candidates for analysis", len(all_candidates))
    LOGGER.info("Os queries finished %.2f s.", time() - start_time)


class AutoAnalyzerService(AnalyzerService):
    app_config: ApplicationConfig
    os_client: OsClient
    namespace_finder: NamespaceFinder
    model_chooser: ModelChooser
    amqp_client: Optional[AmqpClient]

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
        if self.app_config.amqpUrl:
            self.amqp_client = AmqpClient(self.app_config)

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
        request_log: LogItemIndexData,
        size: int = 10,
    ) -> dict[str, Any]:
        min_should_match = self._get_min_should_match_setting(launch)
        log_lines = launch.analyzerConfig.numberOfLogLines
        nested_must: list[dict[str, Any]] = []
        nested_should: list[dict[str, Any]] = []

        strict_fields = choose_fields_to_filter_strict(
            log_lines, self.find_min_should_match_threshold(launch.analyzerConfig) / 100
        )
        for position, message_field in enumerate(strict_fields):
            field_value = getattr(request_log, message_field, "").strip()
            if field_value:
                clause = utils.build_more_like_this_query(
                    min_should_match,
                    field_value,
                    field_name=f"logs.{message_field}",
                    boost=utils.BOOST_MESSAGE,
                    max_query_terms=self.search_cfg.MaxQueryTerms,
                )
                # The primary message field is required: it narrows the child candidate set
                # instead of leaving the nested query a pure disjunction.
                (nested_must if position == 0 else nested_should).append(clause)

        stacktrace_text = request_log.stacktrace_extended.strip()
        if stacktrace_text:
            stacktrace_boost = utils.BOOST_SUPPORTING if log_lines == -1 else utils.BOOST_NEUTRAL
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
                    boost=utils.BOOST_ERROR_IDENTITY,
                    override_min_should_match="1",
                    max_query_terms=self.search_cfg.MaxQueryTerms,
                )
            )

        nested_should.extend(
            utils.build_status_codes_queries(
                request_log.potential_status_codes,
                field_name="logs.potential_status_codes",
                boost=utils.BOOST_ERROR_IDENTITY,
            )
        )

        for field_name, boost_score in LOG_FIELDS_BOOST_SCORES:
            if field_name in strict_fields:
                # Already added above with a higher boost; a second identical clause is pure cost.
                continue
            field_value = getattr(request_log, field_name, "").strip()
            if field_value:
                nested_should.append(
                    utils.build_more_like_this_query(
                        "1",
                        field_value,
                        field_name=f"logs.{field_name}",
                        boost=boost_score,
                        override_min_should_match="1",
                        max_query_terms=self.search_cfg.MaxQueryTerms,
                    )
                )

        if not nested_must and not nested_should:
            return {}

        query = construct_analysis_query(
            request_log,
            nested_must,
            nested_should,
            launch.analyzerConfig.searchScoreMode,
            size,
            self.search_cfg.BoostTestCaseHash,
            self.search_cfg.MaxQueryTerms,
            True,
        )
        utils.append_aa_ma_boosts(query, self.search_cfg)
        query = self.add_constraints_for_launches_into_query(query, launch)
        return self.add_query_with_start_time_decay(query, request_log.start_time)

    def _query_candidates_for_launch(
        self,
        launch: Launch,
        request_logs_by_test_item: list[tuple[TestItem, list[LogItemIndexData]]],
        max_batch_size: Optional[int] = None,
    ) -> list[list[tuple[LogItemIndexData, list[Hit[LogItemIndexData]]]]]:
        """Query candidates for a whole launch, batching queries across test items.

        One `msearch` per test item leaves OpenSearch with only a handful of queries to run
        concurrently and pays a full round trip each time. Batching keeps the request count
        proportional to the query count instead of the test item count.

        Batch size is a memory trade, not just a round-trip one: every query in a batch holds its
        full response until the batch is consumed, and a Test Item response carries its matched
        logs. Past roughly 20 queries the round-trip saving flattens while the memory held per
        request keeps growing linearly, so `AnalysisQueryBatchSize` defaults there.

        :param launch: Launch being analyzed
        :param request_logs_by_test_item: Prepared request logs grouped by their test item
        :param max_batch_size: Queries per `msearch`; defaults to `AnalysisQueryBatchSize`
        :return: Search results per test item, aligned with `request_logs_by_test_item`
        """
        if max_batch_size is None:
            max_batch_size = self.search_cfg.AnalysisQueryBatchSize
        plan: list[tuple[int, LogItemIndexData]] = []
        all_queries: list[dict[str, Any]] = []
        for item_index, (_test_item, request_logs) in enumerate(request_logs_by_test_item):
            for request_log in request_logs:
                if not request_log.message.strip():
                    continue
                query = self._build_nested_analyze_query(launch, request_log)
                if not query:
                    continue
                plan.append((item_index, request_log))
                all_queries.append({})
                all_queries.append(query)

        results: list[list[tuple[LogItemIndexData, list[Hit[LogItemIndexData]]]]] = [
            [] for _ in request_logs_by_test_item
        ]

        # Each query contributes a header and a body to `all_queries`, so a chunk of `chunk`
        # entries corresponds to `chunk // 2` entries of `plan`. Responses are consumed as they
        # are yielded and reduced to their inner-hit logs straight away, so only the current
        # chunk's Test Item models are held at a time.
        chunk = max_batch_size * 2
        for offset in range(0, len(all_queries), chunk):
            planned = plan[offset // 2 : (offset + chunk) // 2]
            consumed = 0
            for (item_index, request_log), hits in zip(
                planned, self.os_client.msearch_grouped(launch.project, all_queries[offset : offset + chunk])
            ):
                consumed += 1
                seen_test_item_ids: set[str] = set()
                unique_hits = []
                for hit in hits:
                    test_item_id = hit.source.test_item_id
                    if test_item_id in seen_test_item_ids:
                        continue
                    seen_test_item_ids.add(test_item_id)
                    unique_hits.append(hit)
                found_log_hits = extract_inner_hit_logs(unique_hits)
                if found_log_hits:
                    results[item_index].append((request_log, found_log_hits))
            if consumed < len(planned):
                LOGGER.warning(
                    "Got %d responses for %d queries in project %s, %d request logs were not analyzed",
                    consumed,
                    len(planned),
                    str(launch.project),
                    len(planned) - consumed,
                )
        return results

    def _should_stop_processing(self, test_items_processed: int) -> bool:
        if test_items_processed >= self.search_cfg.MaxAutoAnalysisItemsToProcess:
            LOGGER.info("Only first %d test items were taken", self.search_cfg.MaxAutoAnalysisItemsToProcess)
            return True
        return False

    def _get_analysis_candidates(
        self,
        launches: list[Launch],
    ) -> list[AnalysisCandidate]:
        t_start = time()
        all_candidates: list[AnalysisCandidate] = []
        processed_items = 0

        for launch in launches:
            preparation_start = time()
            request_logs_by_test_item = prepare_request_logs_for_launch(launch)
            LOGGER.info(
                "Finished preparing request logs for launch '%s', took: %.2f s.",
                str(launch.launchId),
                time() - preparation_start,
            )

            if self._should_stop_processing(processed_items):
                log_query_results(t_start, all_candidates)
                return all_candidates
            remaining = self.search_cfg.MaxAutoAnalysisItemsToProcess - processed_items
            request_logs_by_test_item = request_logs_by_test_item[:remaining]

            item_start = time()
            results_per_test_item = self._query_candidates_for_launch(launch, request_logs_by_test_item)
            mean_processing_time = (time() - item_start) / max(len(request_logs_by_test_item), 1)

            for (source_test_item, request_logs), search_results in zip(
                request_logs_by_test_item, results_per_test_item, strict=True
            ):
                all_candidates.append(
                    AnalysisCandidate(
                        analyzerConfig=launch.analyzerConfig,
                        testItemId=source_test_item.testItemId,
                        project=launch.project,
                        launchId=launch.launchId,
                        launchName=launch.launchName,
                        launchNumber=launch.launchNumber,
                        timeProcessed=mean_processing_time,
                        candidates=search_results,
                        candidatesWithNoDefect=[],
                    )
                )
                processed_items += 1

        log_query_results(t_start, all_candidates)
        return all_candidates

    @utils.ignore_warnings
    def analyze_logs(self, launches: list[Launch]) -> None:
        cnt_launches = len(launches)
        LOGGER.info(f"Started analysis for {cnt_launches} launches")

        t_start = time()
        results: list[AnalysisResult] = []
        results_to_share: dict[int, dict[str, Any]] = {}
        cnt_items_to_process = 0
        chosen_namespaces: dict[int, dict[str, int]] = {}
        results_per_project: dict[int, int] = defaultdict(lambda: 0)

        try:
            all_candidates = self._get_analysis_candidates(launches)
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

                        _, best = ranked_predictions[0]

                        results_per_project[project_id] = results_per_project[project_id] + 1
                        analysis_result = to_analysis_result(analyzer_candidate, best)
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

        except Exception as exc:
            LOGGER.exception("Unable to process analysis candidates", exc_info=exc)

        LOGGER.debug(f"Stats info: {json.dumps(results_to_share)}")
        LOGGER.info(f"Processed {cnt_items_to_process} test items. It took {time() - t_start:.2f} sec.")
        LOGGER.info(f"Finished analysis for {cnt_launches} launches with {len(results)} results.")
        if self.amqp_client:
            self.amqp_client.publish_response(json.dumps(results))
