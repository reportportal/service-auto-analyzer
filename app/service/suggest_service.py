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
from datetime import datetime
from time import time
from typing import Any, Optional

from app.amqp.amqp import AmqpClient
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
from app.commons.model.log_item_index import LogItemIndexData
from app.commons.model.ml import ModelType, TrainInfo
from app.commons.model.test_item_index import TestItemIndexData
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.os_client import OsClient
from app.ml.predictor import PREDICTION_CLASSES, PredictionResult
from app.service.analyzer_service import AnalyzerService
from app.utils import utils
from app.utils.os_migration import (
    bucket_sort_logs_by_similarity,
    build_search_results,
    extract_inner_hit_logs,
    get_request_logs,
)

LOGGER = logging.getLogger("analyzerApp.suggestService")

SIMILARITY_THRESHOLD = 0.98

LOG_FIELDS_BOOST_SCORES = [
    ("detected_message_without_params_extended", 2.0),
    ("only_numbers", 2.0),
    ("message_params", 2.0),
    ("urls", 2.0),
    ("paths", 2.0),
    ("found_exceptions_extended", 8.0),
    ("found_tests_and_methods", 2.0),
]

TEST_ITEM_FIELDS_BOOST_SCORES = [
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

UNSHORTENED_MESSAGE_FIELDS = [
    "detected_message_extended",
    "detected_message_without_params_extended",
    "detected_message_without_params_and_brackets",
]

SHORTENED_MESSAGE_FIELDS = [
    "message_extended",
    "message_without_params_extended",
    "message_without_params_and_brackets",
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


def _create_similarity_dict(
    prediction_results: list[PredictionResult],
) -> dict[str, dict[tuple[str, str], SimilarityResult]]:
    """Create a similarity dictionary for comparing prediction results."""
    _similarity_calculator = similarity_calculator.SimilarityCalculator()
    all_pairs_to_check = []
    for i, result_first in enumerate(prediction_results):
        for j in range(i + 1, len(prediction_results)):
            result_second = prediction_results[j]
            issue_type1 = result_first.data["mrHit"].source.issue_type
            issue_type2 = result_second.data["mrHit"].source.issue_type
            if issue_type1 != issue_type2:
                continue
            items_to_compare = [result_first.data["mrHit"]]
            all_pairs_to_check.append((result_second.data["mrHit"].source, items_to_compare))
    sim_dict = _similarity_calculator.find_similarity(
        all_pairs_to_check, ["detected_message_with_numbers", "stacktrace", "whole_message"]
    )
    return sim_dict


def _filter_by_similarity(
    prediction_results: list[PredictionResult],
    sim_dict: dict[str, dict[tuple[str, str], SimilarityResult]],
) -> list[PredictionResult]:
    """Filter prediction results by removing highly similar duplicates."""
    filtered_results = []
    deleted_indices: set[int] = set()
    for i in range(len(prediction_results)):
        if i in deleted_indices:
            continue
        for j in range(i + 1, len(prediction_results)):
            result_first = prediction_results[i]
            result_second = prediction_results[j]
            group_id = (
                str(result_first.data["mrHit"].id),
                str(result_second.data["mrHit"].id),
            )
            if group_id not in sim_dict["detected_message_with_numbers"]:
                continue
            det_message = sim_dict["detected_message_with_numbers"]
            detected_message_sim = det_message[group_id]
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
    """Deduplicate prediction results by removing highly similar items."""
    sim_dict = _create_similarity_dict(prediction_results)
    filtered_results = _filter_by_similarity(prediction_results, sim_dict)
    return filtered_results


def choose_fields_to_filter_suggests(log_lines_num: int) -> list[str]:
    return UNSHORTENED_MESSAGE_FIELDS if log_lines_num == -1 else SHORTENED_MESSAGE_FIELDS


class SuggestService(AnalyzerService):
    """The service serves suggestion lists in Make Decision modal."""

    app_config: ApplicationConfig
    search_cfg: SearchConfig
    os_client: OsClient
    namespace_finder: NamespaceFinder
    model_chooser: ModelChooser

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

    def _get_config_for_boosting_suggests(self, analyzer_config: AnalyzerConf) -> dict:
        return {
            "max_query_terms": self.search_cfg.MaxQueryTerms,
            "min_should_match": 0.4,
            "min_word_length": self.search_cfg.MinWordLength,
            "filter_min_should_match": [],
            "filter_min_should_match_any": choose_fields_to_filter_suggests(analyzer_config.numberOfLogLines),
            "number_of_log_lines": analyzer_config.numberOfLogLines,
            "filter_by_test_case_hash": True,
            "boosting_model": self.search_cfg.SuggestBoostModelFolder,
            "time_weight_decay": self.search_cfg.TimeWeightDecay,
        }

    def _build_nested_suggest_query(
        self,
        test_item_info: TestItemInfo,
        request_log: LogItemIndexData,
        size: int = 10,
    ) -> dict[str, Any]:
        """Build a test-item-centric nested query for suggestions.

        Constructs a query with nested log matching using more_like_this on various
        field variants, special field boosts, inner_hits, and time decay.

        :param test_item_info: The test item being analyzed
        :param request_log: The request log to find similar items for
        :param size: Maximum number of results per query
        :return: Complete OpenSearch query dictionary, or empty dict if no text to match
        """
        if test_item_info.analyzerConfig.minShouldMatch > 0:
            min_should_match = f"{test_item_info.analyzerConfig.minShouldMatch}%"
        else:
            min_should_match = self.search_cfg.MinShouldMatch
        log_lines = test_item_info.analyzerConfig.numberOfLogLines

        nested_should: list[dict[str, Any]] = []

        # Build more_like_this clauses for all 3 field variants
        message_fields = choose_fields_to_filter_suggests(log_lines)

        for message_field in message_fields:
            message_text = getattr(request_log, message_field, "").strip()
            if message_text:
                nested_should.append(
                    utils.build_more_like_this_query(
                        min_should_match,
                        message_text,
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

        # Add special log field boosts
        for field, boost_score in LOG_FIELDS_BOOST_SCORES:
            field_value = getattr(request_log, field, "").strip()
            if field_value:
                nested_should.append(
                    utils.build_more_like_this_query(
                        "1",
                        field_value,
                        field_name=f"logs.{field}",
                        boost=boost_score,
                        override_min_should_match="1",
                        max_query_terms=self.search_cfg.MaxQueryTerms,
                    )
                )

        # Add potential status codes
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

        if not nested_should:
            return {}

        nested_query: dict[str, Any] = {
            "nested": {
                "path": "logs",
                "score_mode": test_item_info.analyzerConfig.searchScoreMode,
                "query": {"bool": {"should": nested_should}},
                "inner_hits": {
                    "size": 5,
                    "_source": INNER_HITS_SOURCE,
                },
            }
        }

        # Issue type restrictions for suggestions: do not include TI
        issue_type_conditions = utils.prepare_restrictions_by_issue_type(filter_no_defect=False)

        outer_should: list[dict[str, Any]] = [
            {
                "term": {
                    "test_case_hash": {
                        "value": test_item_info.testCaseHash,
                        "boost": abs(self.search_cfg.BoostTestCaseHash),
                    }
                }
            },
        ]

        # Add special test item field boosts
        for field, boost_score in TEST_ITEM_FIELDS_BOOST_SCORES:
            field_value = getattr(request_log, field, "").strip()
            if field_value:
                outer_should.append(
                    utils.build_more_like_this_query(
                        "1",
                        field_value,
                        field_name=field,
                        boost=boost_score,
                        override_min_should_match="1",
                        max_query_terms=self.search_cfg.MaxQueryTerms,
                    )
                )

        query: dict[str, Any] = {
            "size": size,
            "sort": ["_score", {"start_time": "desc"}],
            "_source": TEST_ITEM_SOURCE_FIELDS,
            "query": {
                "bool": {
                    "filter": [{"exists": {"field": "issue_type"}}],
                    "must_not": issue_type_conditions + [{"term": {"test_item_id": str(test_item_info.testItemId)}}],
                    "must": [nested_query],
                    "should": outer_should,
                }
            },
        }

        utils.append_aa_ma_boosts(query, self.search_cfg)
        query = self.add_constraints_for_launches_into_query_suggest(query, test_item_info)
        return self.add_query_with_start_time_decay(query, request_log.start_time)

    def _query_suggested_items(
        self,
        test_item_info: TestItemInfo,
        request_logs: list[LogItemIndexData],
    ) -> list[tuple[LogItemIndexData, list[Hit[LogItemIndexData]]]]:
        """Query OpenSearch for suggested items using test-item-centric nested queries.

        Builds one nested query per request log, sends them via msearch, extracts
        inner hit logs, and aligns them to request logs using bucket sorting.

        :param test_item_info: The test item being analyzed
        :param request_logs: List of request logs to search for
        :return: Search results as list of (request_log, found_log_hits) tuples
        """
        all_queries: list[dict[str, Any]] = []

        for request_log in request_logs:
            message = request_log.message.strip()
            if not message:
                continue
            query = self._build_nested_suggest_query(test_item_info, request_log)
            if not query:
                continue
            all_queries.append({})
            all_queries.append(query)

        if not all_queries:
            return []

        # Execute msearch and deduplicate by test_item_id
        seen_test_item_ids: set[str] = set()
        unique_hits: list[Hit[TestItemIndexData]] = []
        for hit in self.os_client.msearch(test_item_info.project, all_queries):
            test_item_id = hit.source.test_item_id
            if test_item_id not in seen_test_item_ids:
                seen_test_item_ids.add(test_item_id)
                unique_hits.append(hit)

        if not unique_hits:
            return []

        # Extract inner hit logs (convert nested LogData to LogItemIndexData)
        found_log_hits = extract_inner_hit_logs(unique_hits)

        if not found_log_hits:
            return []

        # Align found logs to request logs using bucket sorting
        buckets = bucket_sort_logs_by_similarity(request_logs, found_log_hits)
        return build_search_results(request_logs, buckets)

    def _prepare_not_found_object_info(
        self,
        test_item_info: TestItemInfo,
        processed_time: float,
        model_feature_names: Optional[str],
        model_info: Optional[list[str]],
    ) -> dict:
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
            "clusterId": test_item_info.clusterId,
        }

    def _prepare_request_data(self, test_item_info: TestItemInfo) -> tuple[list[LogItemIndexData], int]:
        """Prepare request logs for suggestion search.

        For normal case: builds a Launch, calls prepare_test_items, and converts to
        request logs. For cluster case: queries OpenSearch for the test item in the
        cluster and converts its logs to request logs with identity fields cleared.

        :param test_item_info: The test item to prepare data for
        :return: Tuple of (request_logs, test_item_id_for_suggest)
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
                test_item_id_for_suggest = int(found_test_item.test_item_id)
                break
            if found_test_item is None:
                return [], 0
            request_logs = get_request_logs(found_test_item, issue_type="")
            # Clear identity fields to prevent boosting
            for log in request_logs:
                log.test_item_name = ""
                log.test_case_hash = 0
                log.unique_id = ""
                log.test_item = 0
            return request_logs, test_item_id_for_suggest
        else:
            # Normal case
            launch = _build_launch_from_test_item_info(test_item_info)
            prepared_items = request_factory.prepare_test_items(launch)
            if not prepared_items:
                return [], test_item_id_for_suggest
            test_item = prepared_items[0]
            request_logs = get_request_logs(test_item, issue_type="")
            return request_logs, test_item_id_for_suggest

    def suggest_items(self, test_item_info: TestItemInfo) -> list[SuggestAnalysisResult]:
        """Suggest issue types for a test item based on similar historical items.

        :param test_item_info: The test item to suggest issue types for
        :return: List of suggestion results sorted by central-weighted score
        """
        LOGGER.info(f"Started suggesting for test item with id: {test_item_info.testItemId}")
        LOGGER.debug(f"Started suggesting items by request: {test_item_info.model_dump_json()}")

        t_start = time()
        results: list[SuggestAnalysisResult] = []
        errors_found: list[str] = []
        errors_count = 0
        model_info_tags: list[str] = []
        feature_names: Optional[str] = None
        try:
            request_logs, test_item_id_for_suggest = self._prepare_request_data(test_item_info)
            LOGGER.info(f"Number of prepared log search requests for suggestions: {len(request_logs)}")
            LOGGER.debug(
                f"Log search requests for suggestions: {json.dumps([item.model_dump() for item in request_logs])}"
            )
            searched_res = self._query_suggested_items(test_item_info, request_logs)
            res_num = sum(len(hits) for _, hits in searched_res)
            LOGGER.info(f"Found {res_num} items by FTS (KNN)")
            LOGGER.debug(
                "Items for suggestions by FTS (KNN): "
                + json.dumps([(item[0].model_dump(), [res.model_dump() for res in item[1]]) for item in searched_res])
            )

            boosting_config = self._get_config_for_boosting_suggests(test_item_info.analyzerConfig)
            boosting_config["chosen_namespaces"] = self.namespace_finder.get_chosen_namespaces(test_item_info.project)

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

            if prediction_results:
                # Extract model info tags (same for all results)
                model_info_tags = prediction_results[0].model_info_tags

                # Group predictions by test item and calculate central-weighted scores
                grouped = utils.group_predictions_by_test_item(prediction_results)
                ranked = utils.score_and_rank_test_items(grouped)

                # Extract the most significant results (one per test item)
                representative_results = [result for _, result in ranked]

                # Deduplicate the representative results
                unique_results = deduplicate_results(representative_results)

                LOGGER.debug(f"Found {len(unique_results)} results for test items.")
                for result in unique_results:
                    prob = result.probability[1]
                    identity = result.identity
                    issue_type = result.data["mrHit"].source.issue_type
                    LOGGER.debug(f"Test item '{identity}' with issue type '{issue_type}' has probability {prob:.2f}")

                processed_time = time() - t_start

                # Build a lookup from representative result to its weighted score
                score_by_identity: dict[str, float] = {}
                for weighted_avg, result in ranked:
                    score_by_identity[result.identity] = weighted_avg

                for pos_idx, result in enumerate(unique_results[: self.search_cfg.MaxSuggestionsNumber]):
                    weighted_score = score_by_identity.get(result.identity, result.probability[1])
                    if weighted_score >= self.suggest_threshold:
                        feature_values = None
                        if result.feature_info:
                            feature_names = ";".join([str(f_id) for f_id in result.feature_info.feature_ids])
                            feature_values = ";".join([str(f) for f in result.feature_info.feature_data])

                        issue_type = result.data["mrHit"].source.issue_type
                        relevant_log_id = utils.extract_real_id(result.data["mrHit"].id)
                        test_item_log_id = utils.extract_real_id(result.data["compared_log"].log_id)
                        test_item_id = result.data["mrHit"].source.test_item
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
                            isMergedLog=False,
                            matchScore=round(weighted_score, 2) * 100,
                            esScore=round(result.data["mrHit"].score, 2),
                            esPosition=result.original_position,
                            modelFeatureNames=feature_names,
                            modelFeatureValues=feature_values,
                            modelInfo=";".join(model_info_tags),
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
        results_to_share = {
            test_item_info.launchId: {
                "not_found": int(len(results) == 0),
                "items_to_process": 1,
                "processed_time": time() - t_start,
                "found_items": len(results),
                "launch_id": test_item_info.launchId,
                "launch_name": test_item_info.launchName,
                "project_id": test_item_info.project,
                "method": "suggest",
                "gather_date": datetime.now().strftime("%Y-%m-%d"),
                "gather_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "number_of_log_lines": test_item_info.analyzerConfig.numberOfLogLines,
                "model_info": model_info_tags,
                "module_version": [self.app_config.appVersion],
                "min_should_match": self.find_min_should_match_threshold(test_item_info.analyzerConfig),
                "errors": errors_found,
                "errors_count": errors_count,
            }
        }
        if self.app_config.amqpUrl:
            amqp_client = AmqpClient(self.app_config)
            if results:
                for model_type in [ModelType.suggestion, ModelType.auto_analysis]:
                    amqp_client.send_to_inner_queue(
                        "train_models",
                        TrainInfo(
                            model_type=model_type, project=test_item_info.project, gathered_metric_total=len(results)
                        ).model_dump_json(),
                    )
            amqp_client.close()
        LOGGER.debug(f"Stats info: {json.dumps(results_to_share)}")
        LOGGER.info(f"Processed the test item. It took {time() - t_start:.2f} sec.")
        LOGGER.info(f"Finished suggesting for test item with {len(results)} results.")
        return results
