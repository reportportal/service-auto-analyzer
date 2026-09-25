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
from app.commons.model.db import Hit
from app.commons.model.launch_objects import (
    AnalysisCandidate,
    AnalysisResult,
    AnalyzerConf,
    ApplicationConfig,
    Launch,
    SearchConfig,
)
from app.commons.model.test_item_index import TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.os_client import OsClient
from app.commons.query_builder import AutoAnalysisQueryBuilder
from app.ml.predictor import AutoAnalysisPredictor, PredictionResult
from app.service.analyzer_service import AnalyzerService
from app.utils import utils

LOGGER = logging.getLogger("analyzerApp.autoAnalyzerService")


def prepare_request_items_for_launch(launch: Launch) -> list[TestItemIndexData]:
    """Prepare request Test Items of the launch, skipping Test Items without logs to analyze by.

    :param launch: Launch to analyze
    :return: Request Test Items with logs
    """
    prepared = request_factory.prepare_test_items(
        launch,
        number_of_logs_to_index=launch.analyzerConfig.numberOfLogsToIndex,
        minimal_log_level=launch.analyzerConfig.minimumLogLevel,
        similarity_threshold_to_drop=launch.analyzerConfig.similarityThresholdToDrop,
    )
    return [prepared_item for prepared_item in prepared if prepared_item.logs]


def to_analysis_result(candidate: AnalysisCandidate, result: PredictionResult) -> AnalysisResult:
    return AnalysisResult(
        testItem=candidate.testItemId,
        issueType=result.data.mrHit.source.issue_type or "",
        relevantItem=int(result.data.mrHit.source.test_item_id),
    )


def choose_best_prediction(prediction_results: list[PredictionResult]) -> Optional[PredictionResult]:
    """Choose the positive prediction with the highest probability, the first found one on ties.

    :param prediction_results: Predictions for found Test Items
    :return: The best positive prediction, or None if there are no positive predictions
    """
    positive_predictions = [result for result in prediction_results if result.label == 1]
    if not positive_predictions:
        return None
    return max(positive_predictions, key=lambda result: (result.probability[1], -result.original_position))


def log_query_results(start_time: float | int, all_candidates: list[AnalysisCandidate]):
    LOGGER.info("Collected %d candidates for analysis", len(all_candidates))
    LOGGER.info("Os queries finished %.2f s.", time() - start_time)


class AutoAnalyzerService(AnalyzerService):
    app_config: ApplicationConfig
    os_client: OsClient
    namespace_finder: NamespaceFinder
    model_chooser: ModelChooser
    amqp_client: Optional[AmqpClient]
    query_builder: AutoAnalysisQueryBuilder

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
        self.query_builder = AutoAnalysisQueryBuilder(self.search_cfg)
        self.amqp_client = None
        if self.app_config.amqpUrl:
            self.amqp_client = AmqpClient(self.app_config)

    def _get_config_for_boosting(self, analyzer_config: AnalyzerConf) -> dict[str, Any]:
        return {
            "max_query_terms": self.search_cfg.MaxQueryTerms,
            "min_should_match": self.find_min_should_match_threshold(analyzer_config) / 100,
            "min_word_length": self.search_cfg.MinWordLength,
            "number_of_log_lines": analyzer_config.numberOfLogLines,
            "boosting_model": self.search_cfg.BoostModelFolder,
            "time_weight_decay": self.search_cfg.TimeWeightDecay,
        }

    def _get_min_should_match_setting(self, launch: Launch) -> str:
        if launch.analyzerConfig.minShouldMatch > 0:
            return f"{launch.analyzerConfig.minShouldMatch}%"
        return self.search_cfg.MinShouldMatch

    def _build_item_query(self, launch: Launch, request_item: TestItemIndexData) -> dict[str, Any]:
        """Build a query to search for Test Items similar to the request one by all its logs.

        :param launch: Launch being analyzed
        :param request_item: Request Test Item
        :return: OpenSearch query, or an empty dict if the request Test Item has no logs to search by
        """
        query = self.query_builder.build(
            request_item,
            number_of_log_lines=launch.analyzerConfig.numberOfLogLines,
            min_should_match=self._get_min_should_match_setting(launch),
            min_logs_to_match="100%" if launch.analyzerConfig.allMessagesShouldMatch else "1",
            filter_no_defect=True,
        )
        if not query:
            return {}
        query = self.add_constraints_for_launches_into_query(query, launch)
        return self.add_query_with_start_time_decay(query, request_item.start_time)

    def _query_candidates_for_launch(
        self,
        launch: Launch,
        request_items: list[TestItemIndexData],
        max_batch_size: Optional[int] = None,
    ) -> list[tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]]:
        """Query candidates for a whole launch, one query per request Test Item, batching queries across Test Items.

        One `msearch` per test item leaves OpenSearch with only a handful of queries to run
        concurrently and pays a full round trip each time. Batching keeps the request count
        proportional to the query count instead of the test item count.

        Batch size is a memory trade, not just a round-trip one: every query in a batch holds its
        full response until the batch is consumed, and a Test Item response carries its matched
        logs. Past roughly 20 queries the round-trip saving flattens while the memory held per
        request keeps growing linearly, so `AnalysisQueryBatchSize` defaults there.

        :param launch: Launch being analyzed
        :param request_items: Prepared request Test Items
        :param max_batch_size: Queries per `msearch`; defaults to `AnalysisQueryBatchSize`
        :return: Request Test Items with the Test Items found for them, aligned with `request_items`
        """
        if max_batch_size is None:
            max_batch_size = self.search_cfg.AnalysisQueryBatchSize
        plan: list[int] = []
        all_queries: list[dict[str, Any]] = []
        for item_index, request_item in enumerate(request_items):
            query = self._build_item_query(launch, request_item)
            if not query:
                continue
            plan.append(item_index)
            all_queries.append({})
            all_queries.append(query)

        results: list[tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]] = [
            (request_item, []) for request_item in request_items
        ]

        # Each query contributes a header and a body to `all_queries`, so a chunk of `chunk`
        # entries corresponds to `chunk // 2` entries of `plan`.
        chunk = max_batch_size * 2
        for offset in range(0, len(all_queries), chunk):
            planned = plan[offset // 2 : (offset + chunk) // 2]
            consumed = 0
            for item_index, hits in zip(
                planned, self.os_client.msearch_grouped(launch.project, all_queries[offset : offset + chunk])
            ):
                consumed += 1
                results[item_index] = (request_items[item_index], hits)
            if consumed < len(planned):
                LOGGER.warning(
                    "Got %d responses for %d queries in project %s, %d test items were not analyzed",
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
            request_items = prepare_request_items_for_launch(launch)
            LOGGER.info(
                "Finished preparing request test items for launch '%s', took: %.2f s.",
                str(launch.launchId),
                time() - preparation_start,
            )

            if self._should_stop_processing(processed_items):
                log_query_results(t_start, all_candidates)
                return all_candidates
            remaining = self.search_cfg.MaxAutoAnalysisItemsToProcess - processed_items
            request_items = request_items[:remaining]

            item_start = time()
            results_per_test_item = self._query_candidates_for_launch(launch, request_items)
            mean_processing_time = (time() - item_start) / max(len(request_items), 1)

            for search_results in results_per_test_item:
                all_candidates.append(
                    AnalysisCandidate(
                        analyzerConfig=launch.analyzerConfig,
                        testItemId=int(search_results[0].test_item_id),
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

                        best = choose_best_prediction(prediction_results)
                        if best is None:
                            LOGGER.debug(f"Test item {analyzer_candidate.testItemId} has no positive predictions")
                            results_to_share[launch_id]["not_found"] += 1
                            results_to_share[launch_id]["processed_time"] += time() - t_start_item
                            continue

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
            self.amqp_client.publish_response(json.dumps([res.model_dump() for res in results]))
