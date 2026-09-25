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
import traceback
from time import time
from typing import Any, Optional

from app.commons import logging, request_factory, similarity_calculator
from app.commons.model.db import Hit
from app.commons.model.launch_objects import (
    AnalyzerConf,
    ApplicationConfig,
    Launch,
    SearchConfig,
    SimilarityResult,
    SuggestAnalysisResult,
    TestItem,
    TestItemInfo,
)
from app.commons.model.test_item_index import LogData, TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.os_client import OsClient
from app.commons.query_builder import TEST_ITEM_SOURCE_FIELDS, SuggestQueryBuilder, best_log_match
from app.ml.predictor import PREDICTION_CLASSES, PredictionResult
from app.service.analyzer_service import AnalyzerService
from app.utils import utils

LOGGER = logging.getLogger("analyzerApp.suggestService")

SIMILARITY_THRESHOLD = 0.98
SIMILARITY_FIELDS = ["detected_message_with_numbers", "stacktrace", "whole_message"]


def _build_launch_from_test_item_info(test_item_info: TestItemInfo) -> Launch:
    """Build a Launch object from TestItemInfo for use with prepare_test_items.

    :param test_item_info: Source test item info
    :return: Launch object wrapping the test item
    """
    test_item = TestItem(
        testItemId=test_item_info.testItemId,
        isAutoAnalyzed=False,
        uniqueId=test_item_info.uniqueId,
        testCaseHash=test_item_info.testCaseHash,
        testItemName=test_item_info.testItemName,
        logs=test_item_info.logs,
    )
    return Launch(
        launchId=test_item_info.launchId,
        project=test_item_info.project,
        launchName=test_item_info.launchName,
        launchNumber=test_item_info.launchNumber,
        analyzerConfig=test_item_info.analyzerConfig,
        testItems=[test_item],
    )


def _get_best_matched_log(result: PredictionResult) -> Optional[LogData]:
    log_match = best_log_match(result.data.mrHit)
    return log_match[1].source if log_match else None


def _create_similarity_dict(
    prediction_results: list[PredictionResult],
) -> dict[str, dict[tuple[str, str], SimilarityResult]]:
    """Create a similarity dictionary for comparing best matched logs of prediction results."""
    _similarity_calculator = similarity_calculator.SimilarityCalculator()
    matched_logs = [_get_best_matched_log(result) for result in prediction_results]
    all_pairs_to_check: list[tuple[LogData, list[LogData]]] = []
    for i, result_first in enumerate(prediction_results):
        first_log = matched_logs[i]
        if first_log is None:
            continue
        for j in range(i + 1, len(prediction_results)):
            second_log = matched_logs[j]
            if second_log is None:
                continue
            if result_first.data.mrHit.source.issue_type != prediction_results[j].data.mrHit.source.issue_type:
                continue
            all_pairs_to_check.append((second_log, [first_log]))
    return _similarity_calculator.find_similarity(all_pairs_to_check, SIMILARITY_FIELDS)


def _filter_by_similarity(
    prediction_results: list[PredictionResult],
    sim_dict: dict[str, dict[tuple[str, str], SimilarityResult]],
) -> list[PredictionResult]:
    """Filter prediction results by removing highly similar duplicates."""
    matched_logs = [_get_best_matched_log(result) for result in prediction_results]
    filtered_results = []
    deleted_indices: set[int] = set()
    for i in range(len(prediction_results)):
        if i in deleted_indices:
            continue
        first_log = matched_logs[i]
        for j in range(i + 1, len(prediction_results)):
            second_log = matched_logs[j]
            if first_log is None or second_log is None:
                continue
            group_id = (str(first_log.log_id), str(second_log.log_id))
            if group_id not in sim_dict["detected_message_with_numbers"]:
                continue
            detected_message_sim = sim_dict["detected_message_with_numbers"][group_id]
            stacktrace_sim = sim_dict["stacktrace"][group_id]
            whole_message_sim = sim_dict["whole_message"][group_id]
            if (
                (detected_message_sim.both_empty or detected_message_sim.similarity >= SIMILARITY_THRESHOLD)
                and (stacktrace_sim.both_empty or stacktrace_sim.similarity >= SIMILARITY_THRESHOLD)
                and (whole_message_sim.both_empty or whole_message_sim.similarity >= SIMILARITY_THRESHOLD)
            ):
                deleted_indices.add(j)
        filtered_results.append(prediction_results[i])
    return filtered_results


def deduplicate_results(
    prediction_results: list[PredictionResult],
) -> list[PredictionResult]:
    """Deduplicate prediction results by removing items with highly similar best matched logs."""
    sim_dict = _create_similarity_dict(prediction_results)
    filtered_results = _filter_by_similarity(prediction_results, sim_dict)
    return filtered_results


def rank_predictions(prediction_results: list[PredictionResult]) -> list[PredictionResult]:
    """Sort predictions by probability descending, keeping the search order on ties.

    :param prediction_results: Predictions for found Test Items
    :return: Sorted predictions
    """
    return sorted(
        prediction_results, key=lambda result: (result.probability[1], -result.original_position), reverse=True
    )


def _sort_logs(test_item: TestItemIndexData) -> list[LogData]:
    return sorted(
        test_item.logs or [],
        key=lambda log: log.log_order if log.log_order is not None else utils.safe_int(log.log_id),
    )


class SuggestService(AnalyzerService):
    """The service serves suggestion lists in Make Decision modal."""

    app_config: ApplicationConfig
    search_cfg: SearchConfig
    os_client: OsClient
    namespace_finder: NamespaceFinder
    model_chooser: ModelChooser
    query_builder: SuggestQueryBuilder

    def __init__(
        self,
        model_chooser: ModelChooser,
        app_config: ApplicationConfig,
        search_cfg: SearchConfig,
        os_client: Optional[OsClient] = None,
    ):
        self.model_chooser = model_chooser
        self.app_config = app_config
        self.search_cfg = search_cfg
        super().__init__(search_cfg=self.search_cfg)
        self.os_client = os_client or OsClient(app_config=self.app_config)
        self.suggest_threshold = 0.4
        self.namespace_finder = NamespaceFinder(app_config)
        self.query_builder = SuggestQueryBuilder(self.search_cfg)

    def _get_config_for_boosting_suggests(self, analyzer_config: AnalyzerConf) -> dict:
        return {
            "max_query_terms": self.search_cfg.MaxQueryTerms,
            "min_should_match": 0.4,
            "min_word_length": self.search_cfg.MinWordLength,
            "number_of_log_lines": analyzer_config.numberOfLogLines,
            "boosting_model": self.search_cfg.SuggestBoostModelFolder,
            "time_weight_decay": self.search_cfg.TimeWeightDecay,
        }

    def _build_item_query(self, test_item_info: TestItemInfo, request_item: TestItemIndexData) -> dict[str, Any]:
        """Build a query to search for Test Items similar to the request one by all its logs.

        :param test_item_info: The test item being analyzed
        :param request_item: Request Test Item
        :return: Complete OpenSearch query dictionary, or empty dict if no text to match
        """
        if test_item_info.analyzerConfig.minShouldMatch > 0:
            min_should_match = f"{test_item_info.analyzerConfig.minShouldMatch}%"
        else:
            min_should_match = self.search_cfg.MinShouldMatch
        query = self.query_builder.build(
            request_item,
            number_of_log_lines=test_item_info.analyzerConfig.numberOfLogLines,
            min_should_match=min_should_match,
            min_logs_to_match="1",
            filter_no_defect=False,
        )
        if not query:
            return {}
        query = self.add_constraints_for_launches_into_query_suggest(query, test_item_info)
        return self.add_query_with_start_time_decay(query, request_item.start_time)

    def _query_suggested_items(
        self,
        test_item_info: TestItemInfo,
        request_item: TestItemIndexData,
    ) -> tuple[TestItemIndexData, list[Hit[TestItemIndexData]]]:
        """Query OpenSearch for Test Items similar to the request one by all its logs.

        :param test_item_info: The test item being analyzed
        :param request_item: Request Test Item
        :return: Request Test Item and found Test Items
        """
        query = self._build_item_query(test_item_info, request_item)
        if not query:
            return request_item, []
        for hits in self.os_client.msearch_grouped(test_item_info.project, [{}, query]):
            return request_item, hits
        return request_item, []

    def _prepare_request_data(self, test_item_info: TestItemInfo) -> tuple[Optional[TestItemIndexData], int]:
        """Prepare request Test Item for suggestion search.

        For normal case: builds a Launch and calls prepare_test_items. For cluster case: queries OpenSearch for the
        test item in the cluster and uses it with identity fields cleared.

        :param test_item_info: The test item to prepare data for
        :return: Tuple of (request Test Item or None if there are no logs to search by, test_item_id_for_suggest)
        """
        test_item_id_for_suggest = test_item_info.testItemId
        if test_item_info.clusterId != 0:
            # Cluster case: find test item in cluster
            query: dict[str, Any] = {
                "_source": TEST_ITEM_SOURCE_FIELDS + ["logs"],
                "size": 1,
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"launch_id": str(test_item_info.launchId)}},
                            {
                                "nested": {
                                    "path": "logs",
                                    "query": {"term": {"logs.cluster_id": str(test_item_info.clusterId)}},
                                }
                            },
                        ],
                        "filter": [
                            {"exists": {"field": "issue_type"}},
                        ],
                    }
                },
            }
            found_test_item: Optional[TestItemIndexData] = None
            for hit in self.os_client.search(test_item_info.project, query):
                found_test_item = hit.source
                test_item_id_for_suggest = utils.safe_int(found_test_item.test_item_id)
                break
            if found_test_item is None or not found_test_item.logs:
                return None, 0
            # Clear identity fields to prevent boosting
            request_item = found_test_item.model_copy(
                update={
                    "test_item_id": "0",
                    "test_item_name": None,
                    "test_case_hash": None,
                    "unique_id": None,
                    "logs": _sort_logs(found_test_item),
                }
            )
            return request_item, test_item_id_for_suggest
        else:
            # Normal case
            launch = _build_launch_from_test_item_info(test_item_info)
            prepared_items = request_factory.prepare_test_items(launch)
            if not prepared_items or not prepared_items[0].logs:
                return None, test_item_id_for_suggest
            return prepared_items[0], test_item_id_for_suggest

    def suggest_items(self, test_item_info: TestItemInfo) -> list[SuggestAnalysisResult]:
        """Suggest issue types for a test item based on similar historical items.

        :param test_item_info: The test item to suggest issue types for
        :return: List of suggestion results sorted by probability
        """
        LOGGER.info(f"Started suggesting for test item with id: {test_item_info.testItemId}")
        LOGGER.debug(f"Started suggesting items by request: {test_item_info.model_dump_json()}")

        t_start = time()
        results: list[SuggestAnalysisResult] = []
        errors_found: list[str] = []
        errors_count = 0
        feature_names: Optional[str] = None
        try:
            request_item, test_item_id_for_suggest = self._prepare_request_data(test_item_info)
            prediction_results: list[PredictionResult] = []
            if request_item is None:
                LOGGER.info(f"There are no logs to search by for test item {test_item_info.testItemId}")
            else:
                LOGGER.info(f"Number of prepared logs for suggestion search: {len(request_item.logs or [])}")
                LOGGER.debug(f"Test item search request for suggestions: {request_item.model_dump_json()}")
                searched_res = self._query_suggested_items(test_item_info, request_item)
                LOGGER.info(f"Found {len(searched_res[1])} items by FTS (KNN)")
                LOGGER.debug(
                    "Items for suggestions by FTS (KNN): " + json.dumps([hit.model_dump() for hit in searched_res[1]])
                )

                boosting_config = self._get_config_for_boosting_suggests(test_item_info.analyzerConfig)
                boosting_config["chosen_namespaces"] = self.namespace_finder.get_chosen_namespaces(
                    test_item_info.project
                )

                predictor_class = PREDICTION_CLASSES[self.search_cfg.MlModelForSuggestions]
                # Create predictor for suggestions
                predictor = predictor_class(
                    model_chooser=self.model_chooser,
                    project_id=test_item_info.project,
                    boosting_config=boosting_config,
                    custom_model_prob=self.search_cfg.ProbabilityForCustomModelSuggestions,
                    hash_source=test_item_info.launchId,
                )

                # Use predictor for the complete prediction workflow
                prediction_results = predictor.predict(searched_res)

            if request_item is not None and prediction_results:
                ranked = rank_predictions(prediction_results)

                # Deduplicate the ranked results
                unique_results = deduplicate_results(ranked)

                LOGGER.debug(f"Found {len(unique_results)} results for test items.")
                for result in unique_results:
                    prob = result.probability[1]
                    identity = result.identity
                    issue_type = result.data.mrHit.source.issue_type
                    LOGGER.debug(f"Test item '{identity}' with issue type '{issue_type}' has probability {prob:.2f}")

                processed_time = time() - t_start

                request_logs = request_item.logs or []
                for pos_idx, result in enumerate(unique_results[: self.search_cfg.MaxSuggestionsNumber]):
                    score = result.probability[1]
                    if score < self.suggest_threshold:
                        continue
                    log_match = best_log_match(result.data.mrHit)
                    if log_match is None:
                        LOGGER.debug(f"Test item '{result.identity}' has no matched logs")
                        continue
                    request_log_index, relevant_log = log_match
                    feature_values = None
                    if result.feature_info:
                        feature_names = ";".join([str(f_id) for f_id in result.feature_info.feature_ids])
                        feature_values = ";".join([str(f) for f in result.feature_info.feature_data])

                    analysis_result = SuggestAnalysisResult(
                        project=test_item_info.project,
                        testItem=test_item_id_for_suggest,
                        testItemLogId=utils.extract_real_id(request_logs[request_log_index].log_id),
                        launchId=test_item_info.launchId,
                        launchName=test_item_info.launchName,
                        launchNumber=test_item_info.launchNumber,
                        issueType=result.data.mrHit.source.issue_type or "",
                        relevantItem=int(result.data.mrHit.source.test_item_id),
                        relevantLogId=utils.extract_real_id(relevant_log.source.log_id),
                        isMergedLog=False,
                        matchScore=round(score * 100, 2),
                        esScore=round(result.data.mrHit.score or 0.0, 2),
                        esPosition=result.original_position,
                        modelFeatureNames=feature_names,
                        modelFeatureValues=feature_values,
                        modelInfo=";".join(result.model_info_tags),
                        resultPosition=pos_idx,
                        usedLogLines=test_item_info.analyzerConfig.numberOfLogLines,
                        minShouldMatch=self.find_min_should_match_threshold(test_item_info.analyzerConfig),
                        processedTime=processed_time,
                        clusterId=test_item_info.clusterId,
                        methodName="suggestion",
                    )
                    results.append(analysis_result)
                    LOGGER.debug(analysis_result)
            else:
                LOGGER.debug(f"There are no results for test item {test_item_info.testItemId}")
        except Exception as exc:
            traceback.print_exc()
            LOGGER.exception(exc)
            errors_found.append(utils.extract_exception(exc))
            errors_count += 1
        results_to_share = [res.model_dump(exclude={"modelFeatureNames", "modelFeatureValues"}) for res in results]
        LOGGER.debug(f"Results: {json.dumps(results_to_share)}")
        LOGGER.info(f"Processed the test item. It took {time() - t_start:.2f} sec. Errors: {errors_count}")
        LOGGER.info(f"Finished suggesting for test item with {len(results)} results.")
        return results
